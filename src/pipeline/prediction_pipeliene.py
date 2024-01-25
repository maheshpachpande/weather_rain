import pandas as pd
import sys
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path='artifact\model.pkl'
            preprocessor_path='artifact\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                Location: str,
                MinTemp: float,
                MaxTemp: float,
                Rainfall: float,
                Evaporation: float,
                Sunshine: float,
                WindGustSpeed: float,
                WindGustDir: str,
                WindDir9am: str,
                Humidity9am: float,
                Humidity3pm: float,
                Pressure9am: float,
                Pressure3pm: float,
                Cloud9am: float,
                Cloud3pm: float,
                Temp9am: float,
                Temp3pm: int,
                WindSpeed9am: float,
                WindSpeed3pm: float,
                WindDir3pm: float
                ):
        
        self.Location =  Location
        self.MinTemp = MinTemp
        self.MaxTemp = MaxTemp
        self.Rainfall = Rainfall
        self.Evaporation = Evaporation
        self.Sunshine = Sunshine
        self.WindGustSpeed = WindGustSpeed
        self.WindGustDir = WindGustDir
        self.WindDir9am = WindDir9am
        self.WindDir3pm = WindDir3pm
        self.Humidity9am = Humidity9am
        self.Humidity3pm = Humidity3pm
        self.Pressure9am = Pressure9am
        self.Pressure3pm = Pressure3pm
        self.Cloud9am = Cloud9am
        self.Cloud3pm = Cloud3pm
        self.Temp9am = Temp9am
        self.Temp3pm = Temp3pm
        self.WindSpeed9am = WindSpeed9am
        self.WindSpeed3pm = WindSpeed3pm
        


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                    'Location':[self.Location],
                    'MinTemp':[self.MinTemp],
                    'MaxTemp':[self.MaxTemp],
                    'Rainfall':[self.Rainfall],
                    'Evaporation':[self.Evaporation],
                    'Sunshine' :[self.Sunshine],
                    'WindGustSpeed' :[self.WindGustSpeed],
                    'WindGustDir':[self.WindGustDir],
                    'WindDir9am':[self.WindDir9am],
                    'WindDir3pm':[self.WindDir3pm],
                    'Humidity9am':[self.Humidity9am],
                    'Humidity3pm':[self.Humidity3pm],
                    'Pressure9am':[self.Pressure9am],
                    'Pressure3pm':[self.Pressure3pm],
                    'Cloud9am':[self.Cloud9am],
                    'Cloud3pm':[self.Cloud3pm],
                    'Temp9am':[self.Temp9am],
                    'Temp3pm':[self.Temp3pm],
                    'WindSpeed9am':[self.WindSpeed9am],
                    'WindSpeed3pm':[self.WindSpeed3pm]
                    }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)