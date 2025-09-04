# 🎓 Student Performance Predictor

This project is an End-to-End Machine Learning application designed to predict student performance based on various factors such as study habits, attendance, parental involvement, and exam scores.

The goal is to showcase how to build a production-ready ML project — from data collection, preprocessing, and model training to deployment.

## 📂 Project Structure
Student Performance Predictor/
│
├── artifacts/             # Stores artifacts like preprocessor, train & test data
│   ├── data.csv
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── train.csv
│   └── test.csv
│
├── catboost_info/         # CatBoost related model training info
├── logs/                  # Logging for debugging & monitoring
├── mlproject.egg-info/    # Project metadata
├── notebook/              # Jupyter notebooks (EDA, experiments)
├── src/                   # Source code (data pipeline, training, utils,logging or exception handling)
├── templates/             # HTML templates (for Flask web app) (FRONTEND)
├── venv/                  # Virtual environment
│
├── app.py                 # Main app runner
├── application.py         # Flask application entry point
└── Readme.md              # Documentation

## 🔑 Features

✔️ Data ingestion and preprocessing pipeline
✔️ Train-test split with saved artifacts
✔️ Feature engineering & transformation (saved as preprocessor.pkl)
✔️ Machine Learning model training (CatBoost / other models)
✔️ Logging for better monitoring & debugging
✔️ Flask web app for real-time prediction
✔️ Deployment-ready project structure

## 🛠️ Tech Stack
Languages: Python (3.9+)
ML Libraries: Pandas, NumPy, Scikit-learn, CatBoost
Visualization: Matplotlib, Seaborn
Backend: Flask
Other Tools: Logging, Pickle

## ⚙️ How It Works

Data Collection → Collects student data (train.csv, test.csv).

Preprocessing → Cleans, scales, and encodes data (preprocessor.pkl).

Model Training → Trains ML model to predict student performance.

Model Deployment → Flask app with HTML UI for user interaction.

Prediction → User inputs student details → Model predicts performance.

## 🚀 Getting Started

#### 1️⃣ Clone the Repository
git clone https://github.com/<your-username>/Student-Performance-Predictor.git
cd Student-Performance-Predictor

#### 2️⃣ Create Virtual Environment
conda create -n student_performance python=3.9 -y
conda activate student_performance

#### 3️⃣ Install Dependencies
pip install -r requirements.txt

#### 4️⃣ Run Application
python application.py

The app will run on http://127.0.0.1:5000/

## 📊 Example Use Case

A teacher enters student details (attendance, study time, past scores, etc.)
The model predicts whether the student’s performance will be High / Medium / Low
Helps in early intervention for weak students

## 📈 Future Improvements

Add Deep Learning model for comparison
Deploy on  AWS 
Integrate with MongoDB / SQL for storing input-output records
Add CI/CD pipeline

## 👨‍💻 Author

Vansh Dhall
📌 Data Scientist in progress | Building AI projects with real-world impact

✨ Don’t forget to star ⭐ this repo if you find it useful!
