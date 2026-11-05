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
from operator_lib.util import Config, Inputs, MLOperator, Outputs, Selector, clock, logger
from operator_lib.util.helpers import provide_historic_data

from algo.data import HOUR, WEATHER_VALUE_COLUMNS, as_utc, has_weather_values, lead_hours, period_start
from algo.training import train_on_ray


WEATHER_FIELDS = [
    *WEATHER_VALUE_COLUMNS,
    "forecasted_for",
    "issued_at",
    "lead_days",
]


# Fixed in code rather than config: every config field becomes a value flows
# can set, and these are not meant to be changed per deployment.
TRAINING_DURATION_DAYS = 90
RETRAINING_INTERVAL_HOURS = 24
FAILED_TRAINING_RETRY_HOURS = 6
MIN_TRAINING_HOURS = 168
# Outputs for one hour are offset by this many microseconds minus the lead in
# seconds, so a later forecast sorts after an earlier one, also across restarts.
# Above the import's longest lead of 7 days and below one second.
LEAD_OFFSET_SECONDS = 8 * 24 * 3600

# Training sees one row per forecast and lead, so the previous 5000 trees
# would take hours per fit.
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
    shortwave_radiation: float
    direct_radiation: float
    diffuse_radiation: float
    direct_normal_irradiance: float
    cloud_cover: float
    temperature_2m: float
    relative_humidity_2m: float
    precipitation: float
    wind_speed_10m: float
    forecasted_for: str
    issued_at: str
    lead_days: int
    energy_time: str
    energy: float


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
        # A PV device without its own time field; history then uses the stored time.
        Selector({"name": "power_untimed_func", "args": ["energy"]}),
    ]

    def init(self, *args, **kwargs):
        # MLOperator.init may immediately call train(), so state used by train must
        # exist before calling it.
        self.next_retraining_at = None
        self.initial_phase_until = None
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
        if selector != "weather_func":
            # PV values are read from the historic platform source by train().
            return None, None, None

        frame = pd.DataFrame([dict(data)])
        forecasted_for = as_utc(frame["forecasted_for"]).iloc[0] if "forecasted_for" in frame else pd.NaT
        if pd.isna(forecasted_for):
            logger.warning("Ignoring weather value with invalid forecasted_for")
            return None, None, None
        if not has_weather_values(frame).iloc[0]:
            return None, None, None
        # The prediction is the mean power of [start, start + 1 h).
        start = period_start(frame["forecasted_for"]).iloc[0]
        if start + HOUR <= clock.now():
            return None, None, None

        if model is None:
            # Without a model there is no value, but consumers still learn why.
            logger.info("No production model available yet")
            prediction = None
            initial_phase = self.initial_phase_message()
        else:
            prediction = float(max(0.0, model.predict(frame)[0]))
            initial_phase = ""

        output_time = start + pd.Timedelta(microseconds=sequence_offset_us(lead_hours(frame).iloc[0]))
        # None leaves the Kafka timestamp to the producer; the forecast hour, up to
        # days ahead, is in the payload. A list, because the evaluation metric would
        # score a single dict at the message time instead of the forecast hour.
        return None, [{
            "timestamp": output_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "output": prediction,
            "initial_phase": initial_phase,
        }], None


def sequence_offset_us(lead):
    """Microseconds added to the output time; larger for a shorter lead, 0 without one."""
    if pd.isna(lead):
        return 0
    return min(999_999, max(0, LEAD_OFFSET_SECONDS - int(round(lead * 3600))))


if __name__ == "__main__":
    OperatorLib(Operator(), name="pv-regressor-operator", git_info_file="git_commit")
