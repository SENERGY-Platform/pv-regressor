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
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "cloud_cover",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
]

FEATURE_COLUMNS = [
    "solar_elevation",
    *WEATHER_VALUE_COLUMNS,
    "lead_hours",
]

HOUR = pd.Timedelta(hours=1)


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


def column(frame, name):
    """A column of the frame, or all missing where the frame lacks it."""
    if name in frame:
        return frame[name]
    return pd.Series(np.nan, index=frame.index, dtype=float)


def weather_values(frame):
    return pd.DataFrame(
        {name: pd.to_numeric(column(frame, name), errors="coerce") for name in WEATHER_VALUE_COLUMNS},
        index=frame.index,
    )


def has_weather_values(frame):
    """Rows with at least one weather variable; the model has nothing to go on otherwise."""
    return weather_values(frame).notna().any(axis=1)


def period_start(forecasted_for):
    """Start of the hour a forecast for valid time H describes.

    Open-Meteo radiation and precipitation at H are the mean and sum over [H - 1 h, H).
    """
    return as_utc(forecasted_for).dt.floor("h") - HOUR


def resolve_issued_at(frame):
    """Issue time from issued_at, else from lead_days, else from the row's time."""
    issued_at = as_utc(column(frame, "issued_at"))
    # Missing leads left out: pandas 1.x warns on converting NaN to a timedelta.
    lead_days = pd.to_numeric(column(frame, "lead_days"), errors="coerce").dropna()
    forecasted_for = as_utc(column(frame, "forecasted_for")).loc[lead_days.index]
    issued_at = issued_at.fillna(forecasted_for - pd.to_timedelta(lead_days * 24, unit="h"))
    if "time" in frame:
        issued_at = issued_at.fillna(as_utc(frame["time"]))
    return issued_at


def lead_hours(frame):
    return (as_utc(column(frame, "forecasted_for")) - resolve_issued_at(frame)).dt.total_seconds() / 3600.0


def build_training_frame(weather, power):
    """Join each forecast to the measured mean power of the hour it describes."""
    # A republished import history holds every forecast twice.
    # Unique labels, which the index-aligned steps below rely on.
    weather = weather.drop_duplicates().reset_index(drop=True)
    power = power.reset_index(drop=True)

    values = weather_values(weather)
    weather = pd.DataFrame({
        "forecasted_for": as_utc(column(weather, "forecasted_for")),
        "issued_at": resolve_issued_at(weather),
    }, index=weather.index).join(values)

    if "energy_time" in power:
        power["measured_at"] = as_utc(power["energy_time"])
    else:
        power["measured_at"] = pd.NaT
    if "time" in power:
        power["measured_at"] = power["measured_at"].fillna(as_utc(power["time"]))
    power["energy"] = pd.to_numeric(power["energy"], errors="coerce")

    weather = weather.dropna(subset=["forecasted_for", "issued_at"])
    weather = weather[has_weather_values(weather)]
    weather = weather[weather["issued_at"] <= weather["forecasted_for"]]
    power = power.dropna(subset=["measured_at", "energy"])
    weather["target_hour"] = period_start(weather["forecasted_for"])
    power["target_hour"] = power["measured_at"].dt.floor("h")

    # Multiple measurements within an hour represent power samples, hence mean.
    hourly_power = power.groupby("target_hour", as_index=False)["energy"].mean()
    weather = weather.groupby(
        ["issued_at", "forecasted_for", "target_hour"], as_index=False
    )[WEATHER_VALUE_COLUMNS].mean()
    joined = weather.merge(hourly_power, on="target_hour", how="inner", validate="many_to_one")
    joined["lead_hours"] = lead_hours(joined)
    return joined.sort_values(["target_hour", "issued_at"]).reset_index(drop=True)


def build_features(frame, latitude, longitude):
    """Build the exact feature contract shared by training and inference."""
    result = pd.DataFrame(index=frame.index)
    # The sun position in the middle of the hour the forecast describes.
    middles = period_start(column(frame, "forecasted_for")) + HOUR / 2
    observer = Observer(latitude=latitude, longitude=longitude)
    result["solar_elevation"] = [
        elevation(observer, value.to_pydatetime()) if not pd.isna(value) else np.nan
        for value in middles
    ]
    result = result.join(weather_values(frame))
    result["lead_hours"] = lead_hours(frame)
    return result[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
