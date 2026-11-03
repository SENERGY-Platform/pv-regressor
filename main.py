"""
   Copyright 2022 InfAI (CC SES)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

"""PV power forecasting operator using operator-lib's external ML lifecycle."""

import datetime
import math

import mlflow
import pandas as pd
from mlflow import MlflowClient
from operator_lib.operator_lib import OperatorLib
from operator_lib.util import Config, Inputs, MLOperator, Outputs, Selector, logger
from operator_lib.util.helpers import provide_historic_data

from algo.training import train_on_ray


WEATHER_FIELDS = [
    "instant_air_temperature",
    "instant_relative_humidity",
    "instant_ultraviolet_index_clear_sky",
    "1_hours_precipitation_amount",
    "instant_cloud_area_fraction",
    "weather_time",
    "forecasted_for",
]


# Fixed in code rather than config: every config field becomes a value flows
# can set, and these are not meant to be changed per deployment.
TRAINING_DURATION_DAYS = 90
RETRAINING_INTERVAL_HOURS = 24
FAILED_TRAINING_RETRY_HOURS = 6
MIN_TRAINING_HOURS = 168
# Must equal max_forecasts of import-yr-forecast.
FORECAST_BATCH_SIZE = 48

# Training sees one row per forecast vintage (up to 48 per hour), so the
# previous 5000 trees would take hours per fit.
ESTIMATOR_PARAMS = {
    "n_estimators": 500,
    "max_depth": 64,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
    "random_state": 42,
}


class CustomConfig(Config):
    lat: float = 51.34
    long: float = 12.36
    # A field of the library, declared again so that flows can set it.
    logger_level: str = "info"


class PVInputs(Inputs):
    # 1_hours_precipitation_amount is no identifier, so the fields are given as
    # the annotations mapping instead of class annotations.
    __annotations__ = {
        "instant_air_temperature": float,
        "instant_relative_humidity": float,
        "instant_ultraviolet_index_clear_sky": int,
        "1_hours_precipitation_amount": float,
        "instant_cloud_area_fraction": float,
        "weather_time": str,
        "forecasted_for": str,
        "energy_time": str,
        "energy": float,
    }


class ForecastOutputs(Outputs):
    output: float
    timestamp: str
    initial_phase: str


class Operator(MLOperator):
    configType = CustomConfig
    inputs = PVInputs
    outputs = ForecastOutputs

    selectors = [
        Selector({"name": "weather_func", "args": WEATHER_FIELDS}),
        Selector({"name": "power_func", "args": ["energy_time", "energy"]}),
    ]

    def init(self, *args, **kwargs):
        # MLOperator.init may immediately call train(), so state used by train must
        # exist before calling it.
        self.weather_batches = {}
        self.next_retraining_at = None
        self.initial_phase_until = None
        self.output_sequence = 0
        super().init(*args, **kwargs)

        if self.next_retraining_at is None:
            self.schedule_from_production_model()

    def schedule_retraining(self, hours):
        self.next_retraining_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=hours)

    def schedule_from_production_model(self):
        """Preserve the training cadence across operator restarts."""
        try:
            version = MlflowClient().get_model_version_by_alias(self.model_id, "production")
            created_at = datetime.datetime.fromtimestamp(
                version.creation_timestamp / 1000.0,
                tz=datetime.timezone.utc,
            )
            self.next_retraining_at = created_at + datetime.timedelta(hours=RETRAINING_INTERVAL_HOURS)
        except Exception:
            self.schedule_retraining(RETRAINING_INTERVAL_HOURS)

    def train(self, model, mlflow_logger):
        """Fetch both historic input streams and train on a Ray worker."""
        try:
            duration = datetime.timedelta(days=TRAINING_DURATION_DAYS)
            dataset_refs = provide_historic_data(duration, require_full_duration=False)

            mlflow_logger.log_params({
                "training_duration_days": TRAINING_DURATION_DAYS,
                "min_training_hours": MIN_TRAINING_HOURS,
                "latitude": float(self.config.lat),
                "longitude": float(self.config.long),
                **{f"model_{key}": value for key, value in ESTIMATOR_PARAMS.items()},
            })

            trained_model, report = train_on_ray(
                dataset_refs=dataset_refs,
                latitude=float(self.config.lat),
                longitude=float(self.config.long),
                min_training_hours=MIN_TRAINING_HOURS,
                estimator_params=ESTIMATOR_PARAMS,
            )
            mlflow_logger.log_dict(report, "training/report.json")
            numeric_metrics = {
                key: value for key, value in report.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            mlflow_logger.log_metrics(numeric_metrics)
        except Exception:
            # Raising would stop the operator (or abort init). Keep the current
            # model and retry later instead.
            logger.exception("Training failed, retrying in %s hours", FAILED_TRAINING_RETRY_HOURS)
            self.schedule_retraining(FAILED_TRAINING_RETRY_HOURS)
            return None

        if trained_model is None:
            logger.info("Not enough matched PV/weather history for training: %s", report)
            # Matched history grows by about one hour per hour of incoming data.
            missing_hours = max(0, MIN_TRAINING_HOURS - report["matched_hours"])
            self.initial_phase_until = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=missing_hours)
            self.schedule_retraining(FAILED_TRAINING_RETRY_HOURS)
            return None

        self.schedule_retraining(RETRAINING_INTERVAL_HOURS)
        return trained_model

    def need_retraining(self, model):
        if self.next_retraining_at is None:
            return False
        due = datetime.datetime.now(datetime.timezone.utc) >= self.next_retraining_at
        if due and mlflow.active_run() is not None:
            # operator-lib leaves a run open when a training attempt returns no
            # model. Close it immediately before the next attempt starts.
            mlflow.end_run(status="FINISHED")
        return due

    def initial_phase_message(self):
        """Tell consumers that no model exists yet and, if known, for how long."""
        if self.initial_phase_until is None:
            return "Die Anwendung befindet sich noch in der Initialisierungsphase"
        # A model can appear no earlier than the next training attempt.
        until = self.initial_phase_until
        if self.next_retraining_at is not None:
            until = max(until, self.next_retraining_at)
        remaining = until - datetime.datetime.now(datetime.timezone.utc)
        hours = max(1, math.ceil(remaining.total_seconds() / 3600))
        return f"Die Anwendung befindet sich noch für ca. {hours} Stunden in der Initialisierungsphase"

    def infer(self, model, data, selector, device_id, timestamp):
        if selector == "power_func":
            # PV values are read from the historic platform source by train().
            return None, None, None
        if selector != "weather_func":
            return None, None, None

        weather_time = pd.to_datetime(data["weather_time"], utc=True, errors="coerce")
        if pd.isna(weather_time):
            logger.warning("Ignoring weather value with invalid weather_time")
            return None, None, None

        batch_key = weather_time.isoformat()
        batch = self.weather_batches.setdefault(batch_key, [])
        batch.append(dict(data))
        expected_size = FORECAST_BATCH_SIZE
        if len(batch) < expected_size:
            return None, None, None

        # Remove before prediction so a malformed batch cannot remain stuck.
        forecast_rows = self.weather_batches.pop(batch_key)[:expected_size]
        if model is None:
            # Without a model there is no value, but consumers still learn why.
            logger.info("No production model available yet")
            predictions = [None] * len(forecast_rows)
            initial_phase = self.initial_phase_message()
        else:
            frame = pd.DataFrame(forecast_rows)
            predictions = [float(max(0.0, value)) for value in model.predict(frame)]
            initial_phase = ""

        self.output_sequence += 1
        result = []
        for row, prediction in zip(forecast_rows, predictions):
            target = pd.to_datetime(row["forecasted_for"], utc=True)
            output_time = target + pd.Timedelta(self.output_sequence, unit="us")
            result.append({
                "timestamp": output_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "output": prediction,
                "initial_phase": initial_phase,
            })
        # operator-lib gives all results of a run one timestamp. None leaves the
        # Kafka timestamp to the producer; the forecast time is in the payload.
        return None, result, None


if __name__ == "__main__":
    OperatorLib(Operator(), name="pv-regressor-operator", git_info_file="git_commit")
