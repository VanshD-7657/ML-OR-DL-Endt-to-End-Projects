from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomerData

application = Flask(__name__)
app = application

# Home Page
@app.route('/')
def index():
    return render_template('home.html')   # directly load form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Form Data:", request.form)
        data = CustomerData(
            Hours_Studied = request.form.get('Hours_Studied'),
            Attendance = request.form.get('Attendance'),
            Sleep_Hours = request.form.get('Sleep_Hours'),
            Previous_Scores = request.form.get('Previous_Scores'),
            Physical_Activity = request.form.get('Physical_Activity'),
            Parental_Involvement = request.form.get('Parental_Involvement'),
            School_Type = request.form.get('School_Type'),
            Gender = request.form.get('Gender'),
            Parental_Education_Level = request.form.get('Parental_Education_Level'),
            Access_to_Resources = request.form.get('Access_to_Resources'),
            Tutoring_Sessions = float(request.form.get('Tutoring_Sessions'))
        )

        pred_df = data.get_data_as_dataframe() 
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=round(results[0], 2))

    except Exception as e:
        return render_template('home.html', results=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0")
