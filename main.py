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

__all__ = ("Operator", )

from operator_lib.util import OperatorBase, Selector, logger
from algo import aux_functions, Agent
import pickle
import pandas as pd
import numpy as np
import os
import astral
from timezonefinder import TimezoneFinder
#from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import NotFittedError

from operator_lib.util import Config
class CustomConfig(Config):
    data_path = "/opt/data"
    lat = 51.34
    long = 12.36
    logger_level = "info"
    

class Operator(OperatorBase):
    configType = CustomConfig

    selectors = [
        Selector({"name": "weather_func", "args": ["instant_air_temperature", "instant_relative_humidity", "instant_ultraviolet_index_clear_sky",
                                                   "1_hours_precipitation_amount", "instant_cloud_area_fraction", "weather_time", "forecasted_for"]}),
        Selector({"name": "power_func", "args": ["energy_time", "energy"]})
    ]

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        data_path = self.config.data_path
        
        if not os.path.exists(data_path):
            os.mkdir(data_path)

        self.data_path = data_path
        
        self.lat = float(self.config.lat)
        self.long = float(self.config.long)
        self.observer = astral.Observer(latitude=self.lat, longitude=self.long)

        tf = TimezoneFinder()
        self.timezone=tf.certain_timezone_at(lng=self.long, lat=self.lat)

        self.weather_same_timestamp = []

        self.agents = []
        

        self.weather_mean = 0
        self.weather_std = 1
        self.power_mean = 0
        self.power_std = 1

        params = {
                 "n_estimators": 5000,
                 "max_depth": 64,
                 "min_samples_split": 5,
                 "learning_rate": 0.01,
                 "loss": "squared_error",
                }
        
        self.model = GradientBoostingRegressor(**params)

        self.model_file = f'{data_path}/model.pickle'

        self.add_microsec = 0

        self.operator_start = pd.Timestamp.now(tz=self.timezone)

    def run_new_weather(self, new_weather_data):
        weather_time = pd.to_datetime(new_weather_data[0]['weather_time'])

        new_weather_array = aux_functions.preprocess_weather_data(new_weather_data, self.observer, case="new_agent")
        new_weather_input = np.mean(new_weather_array, axis=0)
        
        self.agents.append(Agent())
        newest_agent = self.agents[-1]
        newest_agent.save_weather_data(new_weather_input)
        newest_agent.initial_time = weather_time
        if new_weather_input[0] == 0: # This happens if all three weather data points lie in the night between sunset and sunrise.
            predicted_solar_power = 0
            return predicted_solar_power
        std_new_weather_input = aux_functions.standardize_sample(new_weather_input, self.weather_mean, self.weather_std)
        try:
            model_output = self.model.predict(std_new_weather_input.reshape(1,-1))
            predicted_solar_power = aux_functions.re_standardize_sample(model_output, self.power_mean, self.power_std)
            newest_agent.prediction = predicted_solar_power
            return predicted_solar_power
        except NotFittedError:
            logger.info("Model not fitted yet.")
            return 

    def run_new_power(self, new_power_data):
        time, new_power_value = aux_functions.preprocess_power_data(new_power_data, self.timezone)

        old_indices = []
        
        for i, agent in enumerate(self.agents):
            if agent.initial_time + pd.Timedelta(1,'hours') >= time:
                if new_power_value != None:
                    agent.update_power_list(time, new_power_value)
            elif agent.initial_time + pd.Timedelta(1,'hours') < time:
                agents_initial_time_string = agent.initial_time.strftime('%Y-%m-%d %X')
                with open(f'{self.data_path}/agent_{agents_initial_time_string}.pickle', 'wb') as f:
                    pickle.dump(agent, f)
                std_weather_data_array, std_power_mean_array, (self.weather_mean, self.weather_std), (self.power_mean, self.power_std) = aux_functions.load_data(self.data_path)
                self.model = aux_functions.fit_model(self.model, std_weather_data_array, std_power_mean_array)
                old_indices.append(i)

        old_indices = sorted(old_indices, reverse=True)
        for index in old_indices:
            del self.agents[index]

    def create_power_forecast(self, new_weather_data):
        power_forecast = []
        new_weather_array = aux_functions.preprocess_weather_data(new_weather_data, self.observer, case="create_power_forecast")
        new_weather_forecasted_for = [pd.to_datetime(datapoint['forecasted_for']) for datapoint in new_weather_data]
        for i in range(0,len(new_weather_array)):
            new_weather_input = np.mean(new_weather_array[i:i+2], axis=0)
            if new_weather_input[0] == 0: # If data time lies between sunrise and sunset there's definitely no solar power.
                    power_forecast.append((new_weather_forecasted_for[i],0))
            else:
                std_new_weather_input = aux_functions.standardize_sample(new_weather_input, self.weather_mean, self.weather_std)
                try:
                    model_output = self.model.predict(std_new_weather_input.reshape(1,-1))
                    power_forecast.append((new_weather_forecasted_for[i],aux_functions.re_standardize_sample(model_output, self.power_mean, self.power_std)))
                except NotFittedError:
                    logger.info("Model not fitted yet.")
        return power_forecast
        
    def run(self, data, selector, device_id=None):
        logger.debug(selector + ": " + str(data))
        if selector == 'weather_func':
            self.add_microsec += 1
            if len(self.weather_same_timestamp)<47: # Number of weather forecast messages is 48.
                self.weather_same_timestamp.append(data)
            elif len(self.weather_same_timestamp)==47:
                self.weather_same_timestamp.append(data)
                new_weather_data = self.weather_same_timestamp
                _ = self.run_new_weather(new_weather_data[0:2])
                power_forecast = self.create_power_forecast(new_weather_data)
                self.weather_same_timestamp = []
                if len(power_forecast)==48:  #TODO: Implement on conditions that ensures relatively good output. (How much data does one need for good training?)
                    logger.info("PV-Operator-Output:", [{'timestamp':timestamp.strftime('%Y-%m-%dT%H:%M:%S.%fZ'), 'value': float(forecast)} for timestamp, forecast in power_forecast])
                    if pd.to_datetime(data['weather_time']) - self.operator_start < pd.Timedelta(7, 'd'):
                        logger.debug("Still in initial phase!")
                        td_until_start = pd.Timedelta(7,'d') - (pd.to_datetime(data['weather_time']) - self.operator_start)
                        hours_until_start = int(td_until_start.total_seconds()/(60*60))
                        return [{"timestamp":(timestamp + pd.Timedelta(self.add_microsec, "microsecond")).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                                 "output": float(forecast),
                                 "initial_phase": f"Die Anwendung befindet sich noch für ca. {hours_until_start} Stunden in der Initialisierungsphase"}
                                for timestamp, forecast in power_forecast]
                    else:
                        return [{"timestamp":(timestamp + pd.Timedelta(self.add_microsec, "microsecond")).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                                 "output": float(forecast),
                                 "initial_phase": ""} for timestamp, forecast in power_forecast]
        elif selector == 'power_func':
            self.run_new_power(data)

from operator_lib.operator_lib import OperatorLib
if __name__ == "__main__":
    OperatorLib(Operator(), name="pv-regressor-operator", git_info_file='git_commit')
