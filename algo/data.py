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

"""Creation of leak-free, hourly PV/weather training examples."""

import numpy as np
import pandas as pd
from astral import Observer
from astral.sun import elevation


WEATHER_VALUE_COLUMNS = [
    "instant_air_temperature",
    "instant_relative_humidity",
    "instant_ultraviolet_index_clear_sky",
    "1_hours_precipitation_amount",
    "instant_cloud_area_fraction",
]

FEATURE_COLUMNS = [
    "solar_elevation",
    *WEATHER_VALUE_COLUMNS,
    "lead_hours",
]


def identify_input_frames(frames):
    """Identify inputs by mapped columns rather than deployment topic order."""
    weather = power = None
    for frame in frames:
        columns = set(frame.columns)
        if set(WEATHER_VALUE_COLUMNS + ["forecasted_for"]).issubset(columns):
            weather = frame
        if "energy" in columns:
            power = frame
    if weather is None or power is None:
        raise ValueError("Historic inputs must contain one weather dataset and one PV energy dataset")
    return weather, power


def as_utc(series):
    return pd.to_datetime(series, utc=True, errors="coerce")


def build_training_frame(weather, power):
    """Join forecast vintages to the measured mean power of their target hour."""
    weather = weather.copy()
    power = power.copy()

    weather["forecasted_for"] = as_utc(weather["forecasted_for"])
    if "weather_time" in weather:
        weather["issued_at"] = as_utc(weather["weather_time"])
    else:
        weather["issued_at"] = pd.NaT
    if "time" in weather:
        weather["issued_at"] = weather["issued_at"].fillna(as_utc(weather["time"]))

    if "energy_time" in power:
        power["measured_at"] = as_utc(power["energy_time"])
    else:
        power["measured_at"] = pd.NaT
    if "time" in power:
        power["measured_at"] = power["measured_at"].fillna(as_utc(power["time"]))

    for column in WEATHER_VALUE_COLUMNS:
        weather[column] = pd.to_numeric(weather[column], errors="coerce")
    power["energy"] = pd.to_numeric(power["energy"], errors="coerce")

    weather = weather.dropna(subset=["forecasted_for", "issued_at"] + WEATHER_VALUE_COLUMNS)
    power = power.dropna(subset=["measured_at", "energy"])
    weather = weather[weather["issued_at"] <= weather["forecasted_for"]]
    weather["target_hour"] = weather["forecasted_for"].dt.floor("h")
    power["target_hour"] = power["measured_at"].dt.floor("h")

    # Multiple measurements within an hour represent power samples, hence mean.
    hourly_power = power.groupby("target_hour", as_index=False)["energy"].mean()
    weather = weather.groupby(
        ["issued_at", "forecasted_for", "target_hour"], as_index=False
    )[WEATHER_VALUE_COLUMNS].mean()
    joined = weather.merge(hourly_power, on="target_hour", how="inner", validate="many_to_one")
    joined["lead_hours"] = (
        joined["forecasted_for"] - joined["issued_at"]
    ).dt.total_seconds() / 3600.0
    joined = joined[joined["lead_hours"] >= 0]
    return joined.sort_values(["target_hour", "issued_at"]).reset_index(drop=True)


def build_features(frame, latitude, longitude):
    """Build the exact feature contract shared by training and inference."""
    result = pd.DataFrame(index=frame.index)
    target_times = as_utc(frame["forecasted_for"])
    observer = Observer(latitude=latitude, longitude=longitude)
    result["solar_elevation"] = [
        elevation(observer, value.to_pydatetime()) if not pd.isna(value) else np.nan
        for value in target_times
    ]
    for column in WEATHER_VALUE_COLUMNS:
        result[column] = pd.to_numeric(frame[column], errors="coerce")

    issued_at = as_utc(frame["weather_time"]) if "weather_time" in frame else as_utc(frame["issued_at"])
    result["lead_hours"] = (target_times - issued_at).dt.total_seconds() / 3600.0
    return result[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
