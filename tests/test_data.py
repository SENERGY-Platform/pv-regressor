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

import numpy as np
import pandas as pd
import pytest
from astral import Observer
from astral.sun import elevation

from algo.data import (
    FEATURE_COLUMNS,
    WEATHER_VALUE_COLUMNS,
    build_features,
    build_training_frame,
    identify_input_frames,
)


LAT, LONG = 51.34, 12.36


def forecast(forecasted_for, issued_at="auto", lead_days=1, **values):
    row = {name: 1.0 for name in WEATHER_VALUE_COLUMNS}
    row.update(values)
    forecasted = pd.Timestamp(forecasted_for)
    row["forecasted_for"] = forecasted_for
    row["issued_at"] = (
        (forecasted - pd.Timedelta(days=lead_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        if issued_at == "auto" else issued_at
    )
    row["lead_days"] = lead_days
    return row


def power(*samples):
    return pd.DataFrame([{"energy_time": at, "energy": value} for at, value in samples])


def test_forecast_joins_power_of_the_preceding_hour():
    weather = pd.DataFrame([forecast("2026-06-01T12:00:00Z")])
    measured = power(
        ("2026-06-01T10:59:59Z", 999.0),
        ("2026-06-01T11:00:00Z", 100.0),
        ("2026-06-01T11:59:59.999Z", 300.0),
        ("2026-06-01T12:00:00Z", 999.0),
    )

    joined = build_training_frame(weather, measured)

    assert len(joined) == 1
    assert joined.loc[0, "target_hour"] == pd.Timestamp("2026-06-01T11:00:00Z")
    assert joined.loc[0, "energy"] == pytest.approx(200.0)


def test_power_bucket_uses_time_without_energy_time():
    weather = pd.DataFrame([forecast("2026-06-01T12:00:00Z")])
    # Timescale returns tz-naive UTC.
    measured = pd.DataFrame({
        "time": [datetime.datetime(2026, 6, 1, 11, 30), datetime.datetime(2026, 6, 1, 12, 30)],
        "energy": [50.0, 999.0],
    })

    joined = build_training_frame(weather, measured)

    assert joined["energy"].tolist() == [50.0]


def test_offset_timestamps_are_normalised_to_utc():
    weather = pd.DataFrame([forecast("2026-06-01T14:00:00+02:00", issued_at="2026-05-31T14:00:00+02:00")])
    measured = power(("2026-06-01T11:15:00Z", 70.0), ("2026-06-01T13:15:00Z", 999.0))

    joined = build_training_frame(weather, measured)

    assert joined["energy"].tolist() == [70.0]
    assert joined["lead_hours"].tolist() == [24.0]


def test_lead_hours_per_forecast():
    weather = pd.DataFrame([
        forecast("2026-06-01T12:00:00Z", lead_days=1),
        forecast("2026-06-01T12:00:00Z", lead_days=2),
    ])
    measured = power(("2026-06-01T11:30:00Z", 10.0))

    joined = build_training_frame(weather, measured)

    assert sorted(joined["lead_hours"].tolist()) == [24.0, 48.0]


def test_lead_hours_fall_back_to_lead_days_then_time():
    rows = [
        forecast("2026-06-01T12:00:00Z", issued_at=None, lead_days=2),
        forecast("2026-06-01T13:00:00Z", issued_at=None, lead_days=None),
    ]
    frame = pd.DataFrame(rows)
    frame["time"] = [datetime.datetime(2026, 5, 1, 0, 0), datetime.datetime(2026, 6, 1, 7, 0)]

    features = build_features(frame, LAT, LONG)

    assert features["lead_hours"].tolist() == [48.0, 6.0]


def test_exact_duplicate_rows_count_once():
    row = forecast("2026-06-01T12:00:00Z", shortwave_radiation=400.0)
    weather = pd.DataFrame([row, row, dict(row)])
    measured = power(("2026-06-01T11:30:00Z", 10.0))

    joined = build_training_frame(weather, measured)

    assert len(joined) == 1
    assert joined.loc[0, "shortwave_radiation"] == 400.0


def test_rows_without_any_weather_value_are_dropped_and_partial_rows_kept():
    weather = pd.DataFrame([
        forecast("2026-06-01T12:00:00Z", **{name: None for name in WEATHER_VALUE_COLUMNS}),
        forecast("2026-06-01T13:00:00Z", direct_normal_irradiance=None, precipitation=None),
    ])
    measured = power(("2026-06-01T11:30:00Z", 10.0), ("2026-06-01T12:30:00Z", 20.0))

    joined = build_training_frame(weather, measured)

    assert joined["forecasted_for"].tolist() == [pd.Timestamp("2026-06-01T13:00:00Z")]
    assert np.isnan(joined.loc[0, "direct_normal_irradiance"])
    assert joined.loc[0, "temperature_2m"] == 1.0


def test_forecast_issued_after_its_valid_time_is_dropped():
    weather = pd.DataFrame([forecast("2026-06-01T12:00:00Z", issued_at="2026-06-01T13:00:00Z")])
    measured = power(("2026-06-01T11:30:00Z", 10.0))

    assert build_training_frame(weather, measured).empty


def test_solar_elevation_is_taken_in_the_middle_of_the_described_hour():
    frame = pd.DataFrame([forecast("2026-06-01T06:00:00Z")])

    features = build_features(frame, LAT, LONG)

    expected = elevation(
        Observer(latitude=LAT, longitude=LONG),
        datetime.datetime(2026, 6, 1, 5, 30, tzinfo=datetime.timezone.utc),
    )
    assert features.loc[0, "solar_elevation"] == pytest.approx(expected)


def test_training_and_inference_share_the_feature_contract():
    message = forecast("2026-06-01T12:00:00Z", lead_days=2, cloud_cover=40, temperature_2m=None)
    joined = build_training_frame(pd.DataFrame([message]), power(("2026-06-01T11:30:00Z", 10.0)))

    from_training = build_features(joined, LAT, LONG)
    from_message = build_features(pd.DataFrame([message]), LAT, LONG)

    assert list(from_training.columns) == FEATURE_COLUMNS
    assert list(from_message.columns) == FEATURE_COLUMNS
    pd.testing.assert_frame_equal(from_training.reset_index(drop=True), from_message, check_dtype=False)


def test_identify_input_frames_by_columns():
    weather = pd.DataFrame([forecast("2026-06-01T12:00:00Z")])
    measured = pd.DataFrame({"time": [datetime.datetime(2026, 6, 1)], "energy": [1.0]})

    assert identify_input_frames([measured, weather]) == (weather, measured)
    with pytest.raises(ValueError):
        identify_input_frames([measured])
