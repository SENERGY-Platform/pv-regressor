import numpy as np
import pandas as pd
from astral import sun
import os
import pickle

def preprocess_power_data(new_power_data, timezone):
    time=pd.to_datetime(new_power_data['energy_time'])
    if time.tzinfo == None:
        time = time.tz_localize(timezone)
    power = new_power_data['energy']
    return time, power


def preprocess_weather_data(new_weather_data, weather_time, observer):

    today_sunrise = pd.to_datetime(sun.sunrise(observer, date=weather_time))
    today_sunset = pd.to_datetime(sun.sunset(observer, date=weather_time))

    dist_today_sunrise = np.abs((weather_time-today_sunrise).total_seconds())
    dist_today_sunset = np.abs((weather_time-today_sunset).total_seconds())

    if (weather_time <= today_sunrise) or (today_sunset <= weather_time):
        min_seconds_to_sunrise_or_sunset = 0
    else:
        min_seconds_to_sunrise_or_sunset = min(dist_today_sunrise, dist_today_sunset)

    # Normalization of relative humidity (min=0, max=100), uv-index (min=0, max~10), cloud area fraction
    # (min=0, max=100) is easy. For temperature we choose min=-10, max=30 and for precipitation amount: min=0, max=10.

    aux_list = [[min_seconds_to_sunrise_or_sunset, data_point['instant_air_temperature'], data_point['instant_relative_humidity'],
                 data_point['instant_ultraviolet_index_clear_sky'], data_point['1_hours_precipitation_amount'],
                 data_point['instant_cloud_area_fraction']] for data_point in new_weather_data]

    weather_array = np.array(aux_list)
    weather_array = np.nan_to_num(weather_array)

    return weather_array

def batch_standardize(array):
    if np.std(array, axis=0).all()==False: # If the array consisting of the respective std dev contains zeros the function returns Nan! 
        return array
    else:
        return (array-np.mean(array, axis=0))/np.std(array, axis=0)

def standardize_sample(array, mean, std):
    if np.array(std).all()==False:
        return array
    else:
        return (array-mean)/std

def re_standardize_sample(array, mean, std):
    return array*std + mean

def get_sunrise_sunset(observer, time):
    sunrise = pd.to_datetime(sun.sunrise(observer, date=time))
    sunset = pd.to_datetime(sun.sunset(observer, date=time))
    return sunrise, sunset

def load_data(data_path):
    agent_files = [filename for filename in os.listdir(data_path) if filename.startswith("agent")]
    old_agents = []
    for filename in agent_files:
        with open(f'{data_path}/{filename}','rb') as f:
            agent = pickle.load(f)
            if agent.power_list != []:
                old_agents.append(agent)
    weather_data_array = np.vstack([agent.initial_weather_data for agent in old_agents]) # This array contains all seen weather samples as rows. It's the input of the regression. 
    power_mean_array = np.vstack([np.mean([power_value for _, power_value in agent.power_list]) for agent in old_agents])# This array contains the power of all the agents. It's the target of the regression
    std_weather_data_array = batch_standardize(weather_data_array)
    std_power_mean_array = batch_standardize(power_mean_array.reshape((-1,)))
    return std_weather_data_array, std_power_mean_array, (np.mean(weather_data_array, axis=0), np.std(weather_data_array, axis=0)), (np.mean(power_mean_array, axis=0), np.std(power_mean_array, axis=0))

def fit_model(model, std_weather_data_array, std_power_mean_array):
    model.fit(std_weather_data_array, std_power_mean_array)
    return model


    