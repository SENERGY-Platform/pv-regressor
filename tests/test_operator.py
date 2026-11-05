"""
   Copyright (c) 2026 InfAI (CC SES)

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

import datetime
import types

import numpy as np
import pytest
from operator_lib.util import clock, get_selector

from algo.data import FEATURE_COLUMNS, WEATHER_VALUE_COLUMNS
from algo.model import PVForecastModel
from main import WEATHER_FIELDS, Operator, PVInputs


NOW = datetime.datetime(2026, 6, 1, 0, 0, tzinfo=datetime.timezone.utc)


class StubEstimator:
    def __init__(self, value):
        self.value = value
        self.features = None

    def predict(self, features):
        self.features = features
        return np.full(len(features), self.value)


class StubPyFunc:
    """Stands in for mlflow's PyFuncModel, which calls predict(None, data)."""

    def __init__(self, value):
        self.estimator = StubEstimator(value)
        self.python_model = PVForecastModel(self.estimator, 51.34, 12.36)

    def predict(self, data):
        return self.python_model.predict(None, data)


@pytest.fixture
def operator():
    clock.set_fixed(NOW)
    op = Operator()
    op.next_retraining_at = None
    op.initial_phase_until = None
    yield op
    clock.set_fixed(None)


def message(forecasted_for="2026-06-02T12:00:00Z", lead_days=1, **values):
    data = {name: 100.0 for name in WEATHER_VALUE_COLUMNS}
    data.update(values)
    valid = datetime.datetime.fromisoformat(forecasted_for.replace("Z", "+00:00"))
    data["forecasted_for"] = forecasted_for
    data["issued_at"] = (valid - datetime.timedelta(days=lead_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    data["lead_days"] = lead_days
    return data


def infer(op, data, model=None, selector="weather_func"):
    return op.infer(model, data, selector, None, NOW)


def test_without_model_answers_null_with_initial_phase(operator):
    dt, result, model = infer(operator, message())

    assert dt is None and model is None
    assert len(result) == 1
    assert result[0]["output"] is None
    assert result[0]["initial_phase"].startswith("Die Anwendung befindet sich")
    assert result[0]["timestamp"].startswith("2026-06-02T11:00:00.")
    assert result[0]["timestamp"].endswith("Z")


def test_with_model_predicts_one_row_for_the_preceding_hour(operator):
    model = StubPyFunc(250.0)

    _, result, _ = infer(operator, message(), model)

    assert result == [{
        "timestamp": "2026-06-02T11:00:00.604800Z",
        "output": 250.0,
        "initial_phase": "",
    }]
    assert list(model.estimator.features.columns) == FEATURE_COLUMNS


def test_negative_prediction_is_clipped(operator):
    _, result, _ = infer(operator, message(), StubPyFunc(-5.0))

    assert result[0]["output"] == 0.0


def test_later_forecast_for_the_same_hour_sorts_after_the_earlier(operator):
    _, early, _ = infer(operator, message(lead_days=2), StubPyFunc(1.0))
    _, late, _ = infer(operator, message(lead_days=1), StubPyFunc(1.0))

    assert early[0]["timestamp"] < late[0]["timestamp"]
    assert early[0]["timestamp"][:19] == late[0]["timestamp"][:19] == "2026-06-02T11:00:00"


def test_message_without_any_weather_value_yields_nothing(operator):
    data = message(**{name: None for name in WEATHER_VALUE_COLUMNS})

    assert infer(operator, data, StubPyFunc(1.0)) == (None, None, None)


def test_message_with_some_values_missing_is_predicted(operator):
    data = message(direct_normal_irradiance=None, temperature_2m=None)

    _, result, _ = infer(operator, data, StubPyFunc(7.0))

    assert result[0]["output"] == 7.0


def test_invalid_forecasted_for_yields_nothing(operator):
    data = message()
    data["forecasted_for"] = "not a time"

    assert infer(operator, data) == (None, None, None)


@pytest.mark.parametrize("forecasted_for, answered", [
    ("2026-06-01T00:00:00Z", False),  # hour [23:00, 00:00) just ended
    ("2026-06-01T01:00:00Z", True),   # hour [00:00, 01:00) just began
])
def test_hours_already_over_are_skipped(operator, forecasted_for, answered):
    _, result, _ = infer(operator, message(forecasted_for))

    assert (result is not None) == answered


def test_power_selectors_are_ignored(operator):
    # A complete weather message, so only the selector decides.
    data = {**message(), "energy": 1.0, "energy_time": "2026-06-01T00:00:00Z"}
    for selector in ("power_func", "power_untimed_func"):
        assert infer(operator, data, StubPyFunc(1.0), selector) == (None, None, None)


def mappings(*dests):
    return [types.SimpleNamespace(dest=dest) for dest in dests]


def test_selectors_match_the_mapped_fields():
    assert get_selector(mappings(*WEATHER_FIELDS), Operator.selectors) == "weather_func"
    assert get_selector(mappings("energy_time", "energy"), Operator.selectors) == "power_func"
    assert get_selector(mappings("energy"), Operator.selectors) == "power_untimed_func"


def test_declared_inputs_are_the_selector_fields():
    assert PVInputs.names() == WEATHER_FIELDS + ["energy_time", "energy"]
