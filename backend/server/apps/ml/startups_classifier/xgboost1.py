
import joblib
import numpy as np
import statistics
from scipy import stats
from numpy import NaN
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

class XGBClassifier1:
    def __init__(self):
        path_to_artifacts = "../../research/"
        self.values_fill_missing1 =  joblib.load(path_to_artifacts + "train_mode.joblib")
        self.values_fill_missing2 =  joblib.load(path_to_artifacts + "train_median.joblib")
        self.replace1 =  joblib.load(path_to_artifacts + "dict_start.joblib")
        self.replace2 =  joblib.load(path_to_artifacts + "dict_start2.joblib")
        self.replace3 =  joblib.load(path_to_artifacts + "dict_start3.joblib")
        self.encoders = joblib.load(path_to_artifacts + "encoders.joblib")
        self.trainingset = joblib.load(path_to_artifacts + "X_train.joblib")
        self.model = joblib.load(path_to_artifacts + "xgboost.joblib")

    def preprocessing(self, input_data):
        # JSON to pandas DataFrame
        input_data = pd.DataFrame(input_data, index=[0])
        # fill missing values
        input_data.fillna(self.values_fill_missing1)
        input_data.fillna(self.values_fill_missing2)
        #Dictionaries
        input_data.replace(self.replace1)
        input_data.replace(self.replace2)
        
        return input_data

   
    def preprocessing(self, input_data):
        # JSON to pandas DataFrame
        input_data = pd.DataFrame(input_data, index=[0])
        
        #Dictionaries
        input_data.replace(self.replace3)
        # convert categoricals
        for column in [
            "Headquarters_Location", 
            "Funding_Stage", 
            "Industry_Groups",
        ]:
            categorical_convert = self.encoders[column]
            input_data[column] = categorical_convert.transform(input_data[column])

        
        
        return input_data

    
    def cap_data(input_data):
        for col in input_data.columns:
            print("capping the ",col)
            if (((input_data[col].dtype)=='float64') | ((input_data[col].dtype)=='int64')):
                outliers = stats.zscore(input_data[col])
                input_data[col][outliers < -3] = statistics.median(input_data[col])
                input_data[col][outliers >  3] = statistics.median(input_data[col])
            else:
                input_data[col]=input_data[col]
        return input_data

    def predict(self, input_data):
        return self.model.predict_proba(input_data)

    def postprocessing(self, input_data):
        label = 0
        if input_data[1] > 0.5:
            label = 1
        return {"probability": input_data[1], "label": label, "status": "ok"}

    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)[0]  # only one sample
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction