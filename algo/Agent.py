class Agent:
    def __init__(self):
        self.initial_weather_data = None
        self.initial_time = None
        self.power_list = []
        self.prediction = None
        
    def save_weather_data(self,weather_data):
        self.initial_weather_data = weather_data
        
    def update_power_list(self,time, new_power_value):
        self.power_list.append((time, new_power_value))
        
