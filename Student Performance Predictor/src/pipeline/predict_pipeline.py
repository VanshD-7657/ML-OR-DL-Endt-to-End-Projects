import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
            try:
                model_path = 'artifacts/model.pkl'
                preprocessor_path = 'artifacts/preprocessor.pkl'
                model = load_object(file_path=model_path)
                preprocessor = load_object(file_path=preprocessor_path)
                data_scaled = preprocessor.transform(features)
                print("Shape after preprocessing:", data_scaled.shape)
                print("Model expects:", model.coef_.shape)
                preds = model.predict(data_scaled)
                return preds
            
            except Exception as e:
                raise CustomException(e, sys)



class CustomerData:
    def __init__(self, Hours_Studied, Attendance, Sleep_Hours, Previous_Scores, Physical_Activity,
                 Parental_Involvement, School_Type, Gender, Parental_Education_Level, Access_to_Resources, Tutoring_Sessions):
        self.Hours_Studied = Hours_Studied
        self.Attendance = Attendance
        self.Sleep_Hours = Sleep_Hours
        self.Previous_Scores = Previous_Scores
        self.Physical_Activity = Physical_Activity
        self.Parental_Involvement = Parental_Involvement
        self.School_Type = School_Type
        self.Gender = Gender
        self.Parental_Education_Level = Parental_Education_Level
        self.Access_to_Resources = Access_to_Resources
        self.Tutoring_Sessions = Tutoring_Sessions

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "Hours_Studied": [self.Hours_Studied],
                "Attendance": [self.Attendance],
                "Sleep_Hours": [self.Sleep_Hours],
                "Previous_Scores": [self.Previous_Scores],
                "Physical_Activity": [self.Physical_Activity],
                "Parental_Involvement": [self.Parental_Involvement],
                "School_Type": [self.School_Type],
                "Gender": [self.Gender],
                "Parental_Education_Level": [self.Parental_Education_Level],
                "Access_to_Resources": [self.Access_to_Resources],
                "Tutoring_Sessions": [self.Tutoring_Sessions],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
