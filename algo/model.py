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

"""MLflow-compatible PV forecast model."""

import datetime

import mlflow.pyfunc
import numpy as np

from algo.data import build_features


class PVForecastModel(mlflow.pyfunc.PythonModel):
    def __init__(self, estimator, latitude, longitude):
        self.estimator = estimator
        self.latitude = latitude
        self.longitude = longitude
        self.trained_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def predict(self, context, model_input, params=None):
        features = build_features(model_input, self.latitude, self.longitude)
        predictions = self.estimator.predict(features)
        # The sun position is deterministic and prevents positive night forecasts.
        return np.where(features["solar_elevation"].to_numpy() <= 0, 0.0, predictions)
