"""
   Copyright 2026 InfAI (CC SES)

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

"""Ray entry point for externally executed model training."""

import math
import ray
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from algo.data import build_features, build_training_frame, identify_input_frames
from algo.model import PVForecastModel


@ray.remote
def train_remote(dataset_handles, latitude, longitude, min_training_hours, estimator_params):
    frames = [dataset.to_pandas(limit=None) for dataset in dataset_handles]
    weather, power = identify_input_frames(frames)
    joined = build_training_frame(weather, power)
    unique_hours = int(joined["target_hour"].nunique()) if not joined.empty else 0
    report = {
        "weather_rows": int(len(weather)),
        "power_rows": int(len(power)),
        "matched_rows": int(len(joined)),
        "matched_hours": unique_hours,
        "minimum_training_hours": int(min_training_hours),
    }
    if unique_hours < min_training_hours:
        report["status"] = "insufficient_data"
        return None, report

    features = build_features(joined, latitude, longitude)
    target = joined["energy"].astype(float)
    valid = features.notna().any(axis=1) & target.notna()
    features, target, joined = features.loc[valid], target.loc[valid], joined.loc[valid]

    estimator = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", GradientBoostingRegressor(**estimator_params)),
    ])

    # Keep every forecast for one target hour on the same side of the split.
    hours = joined["target_hour"].drop_duplicates().sort_values().tolist()
    split_index = max(1, int(len(hours) * 0.8))
    if split_index < len(hours):
        validation_hours = set(hours[split_index:])
        validation_mask = joined["target_hour"].isin(validation_hours)
        evaluation_model = clone(estimator).fit(features.loc[~validation_mask], target.loc[~validation_mask])
        predicted = evaluation_model.predict(features.loc[validation_mask])
        actual = target.loc[validation_mask]
        report.update({
            "validation_rows": int(validation_mask.sum()),
            "validation_mae": float(mean_absolute_error(actual, predicted)),
            "validation_rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
            "validation_r2": float(r2_score(actual, predicted)) if len(actual) > 1 else 0.0,
        })

    estimator.fit(features, target)
    report["status"] = "trained"
    report["training_rows"] = int(len(features))
    return PVForecastModel(estimator, latitude, longitude), report


def train_on_ray(dataset_refs, latitude, longitude, min_training_hours, estimator_params):
    """Resolve source tasks and execute preprocessing and fitting off-process."""
    dataset_handles = ray.get(dataset_refs)
    return ray.get(train_remote.remote(
        dataset_handles,
        latitude,
        longitude,
        min_training_hours,
        estimator_params,
    ))
