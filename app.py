from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Load model and preprocessors
model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')
ordinal_encoder = joblib.load('ordinal_encoder.pkl')

app = Flask(__name__)

# Serve HTML form
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Mapping
            gender_map = {'Male': 0, 'Female': 1}
            parental_education_map = {
                'High School': 0,
                "Bachelor's": 1,
                "Master's": 2,
                'PhD': 3
            }
            yes_no_map = {'No': 0, 'Yes': 1}

            # Read form data
            data = {
                'Gender': gender_map[request.form['gender']],
                'Study_Hours_per_Week': float(request.form['study_hours']),
                'Attendance_Rate': float(request.form['attendance']),
                'Past_Exam_Scores': float(request.form['past_scores']),
                'Parental_Education_Level': parental_education_map[request.form['parent_education']],
                'Internet_Access_at_Home': yes_no_map[request.form['internet']],
                'Extracurricular_Activities': yes_no_map[request.form['activities']],
            }

            # Convert and scale
            input_df = pd.DataFrame([data])
            input_df_scaled = scaler.transform(input_df)

            # Predict
            prediction = model.predict(input_df_scaled)[0]
            return render_template('form.html', prediction=round(prediction, 2))

        except Exception as e:
            return render_template('form.html', prediction=f"Error: {str(e)}")

    return render_template('form.html')


# Predict from form
@app.route('/predict_form', methods=['POST'])
def predict_form():
    try:
        # Map human-readable values to numeric
        gender_map = {'Male': 0, 'Female': 1}
        parental_education_map = {
            'High School': 0,
            'PhD': 1,
            'Bachelors': 2,
            'Masters': 3
        }
        yes_no_map = {'No': 0, 'Yes': 1}

        # Read form data
        data = {
            'Gender': gender_map[request.form['Gender']],
            'Study_Hours_per_Week': float(request.form['Study_Hours_per_Week']),
            'Attendance_Rate': float(request.form['Attendance_Rate']),
            'Past_Exam_Scores': float(request.form['Past_Exam_Scores']),
            'Parental_Education_Level': parental_education_map[request.form['Parental_Education_Level']],
            'Internet_Access_at_Home': yes_no_map[request.form['Internet_Access_at_Home']],
            'Extracurricular_Activities': yes_no_map[request.form['Extracurricular_Activities']],
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Predict
        prediction = model.predict(input_df)[0]

        return render_template('form.html', prediction=round(prediction, 2))

    except Exception as e:
        return render_template('form.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
