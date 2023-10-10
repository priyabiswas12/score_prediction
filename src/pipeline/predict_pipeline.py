import sys
import pandas as pd
import numpy as np
import os


from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path) #this function is in utils which is common functionality
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData: # responsible for mapping all the inputs that we're giving the html to the backend 
    def __init__(  self, #we are getting these values form the web app
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self): # return all the input in the form of a df
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
        




# if __name__=="__main__":  
#     data=CustomData(
#                 gender="Female",
#                 race_ethnicity="Group B",
#                 parental_level_of_education="some college",
#                 lunch="standard",
#                 test_preparation_course="completed",
#                 reading_score=float(15),
#                 writing_score=float(15))

#     pred_df=data.get_data_as_data_frame()
#     print(pred_df)
#     print("Before Prediction")

#     predict_pipeline=PredictPipeline()
#     print("Mid Prediction")
#     results=predict_pipeline.predict(pred_df)
#     print("after Prediction")