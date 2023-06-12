from flask import Flask, request
from logging import FileHandler, WARNING
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

@app.errorhandler(500)
def internal_server_error(e):
    return repr(e), 500

@app.route('/')
def hello_world():
	return'Hello World!'

# Result page 
@app.route("/outputstroke",methods=["POST"])
# Define the tasks when recieving a POST request 
def outputstroke():
    with open('model/model.pkl', 'rb') as f:
        model_dict = pickle.load(f)

    model = model_dict['model']
    scaler = model_dict['scaler']

    #gender
    g = request.form['gender']
    
    #age
    a = request.form['age']
    a = int(a)

    #hyper-tension
    hyt = request.form['hypertension']
    hyt = hyt.lower()
    if hyt == "yes":
        hyt = 1
    else:
        hyt = 0
    #heart-disease
    ht = request.form['heart-disease']
    ht = ht.lower()
    if ht == "yes":
        ht = 1
    else:
        ht = 0
    #marriage
    m = request.form['marriage']
    m = m.lower()

    #worktype
    w = request.form['worktype']
    w = w.lower()

    #residency-type
    r = request.form['residency']
    r = r.lower()
    
    #glucose-levels
    gl = float(request.form['glucose'])

    #bmi
    b = float(request.form['bmi'])

    #smoking
    s = request.form['smoking'].lower()

    # Create a dataframe for the data
    df = pd.DataFrame(data = {'age': [a], 
                              'hypertension': [hyt], 
                              'heart_disease': [ht], 
                              'avg_glucose_level': [gl], 
                              'bmi': [b],
                              'gender_Female': [0 if g == 'male' else 1],
                              'gender_Male': [1 if g == 'male' else 0],
                              'ever_married_No': [0 if m == 'yes' else 1],
                              'ever_married_Yes': [1 if m == 'yes' else 0],
                              'work_type_Govt_job': [1 if w == 'government' else 0],
                              'work_type_Never_worked': [1 if w == 'others' else 0],
                              'work_type_Private': [1 if w == 'private' else 0],
                              'work_type_Self-employed': [1 if w =='self-employed' else 0],
                              'work_type_children': [1 if w == 'student' else 0],
                              'Residence_type_Rural': [1 if r == 'rural' else 0],
                              'Residence_type_Urban': [0 if r == 'rural' else 1],
                              'smoking_status_Unknown': [1 if s == 'unknown' else 0],
                              'smoking_status_formerly smoked': [1 if s == 'formerly smoked' else 0],
                              'smoking_status_never smoked': [1 if s == 'never smoked' else 0],
                              'smoking_status_smokes': [1 if s == 'smokes' else 0]})
                              

    scaled_data = scaler.transform(df)
    
    with open('errorlog.txt', 'a') as f:
        f.write(f'{df.head()}\n')
        f.write(f'{scaler.scale_}\n')
        f.write(f'{scaled_data}\n')

    # try to make prediction, otherwise notify user the entries are invalid
    try:
        # make prediction
        #prediction = stroke_pred(g,a,hyt,ht,m,w,r,gl,b,s)
        prediction = model.predict(scaled_data)

        with open('errorlog.txt', 'a') as f:
            f.write(f'Prediction: {prediction[0]}')
        # render index_2 for result page
        return {'prediction': int(prediction[0])}

    except ValueError:
        return "Please Enter Valid Values"

# Result page 
@app.route("/outputheart",methods=["POST"])
# Define the tasks when recieving a POST request 
def outputheart():
    with open('model/heart_disease.pkl', 'rb') as f:
        model_dict = pickle.load(f)

    model = model_dict['model']
    scaler = model_dict['scaler']
    
    with open('errorlog.txt', 'a') as f:
        f.write(f'Model loaded\n')


    #gender
    bmi = float(request.form['bmi'])
    smoking = request.form['smoking'].lower()
    drinking = request.form['drinking'].lower()
    stroke = request.form['stroke'].lower()
    ph_health = float(request.form['ph_health'])
    men_health = float(request.form['men_health'])
    stairs = request.form['stairs'].lower()
    gender = request.form['gender'].lower()
    age_group = request.form['age-group']
    race = request.form['race'].lower()
    diabetes = request.form['diabetes'].lower()
    exercise = request.form['exercise'].lower()
    gen_health = request.form['gen-health'].lower()
    sleep = float(request.form['sleep'])
    kid_stone = request.form['kid-stone'].lower()
    asthma = request.form['asthma'].lower()
    skin_cancer = request.form['skin-cancer'].lower()
    
    with open('errorlog.txt', 'a') as f:
        f.write(f'Data read\n')

    smoking = 1 if smoking == 'yes' else 0
    drinking = 1 if drinking == 'yes' else 0
    stroke = 1 if stroke == 'yes' else 0
    stairs = 1 if stairs == 'yes' else 0
    diabetes = 1 if diabetes == 'yes' else 0
    exercise = 1 if exercise == 'yes' else 0
    asthma = 1 if asthma == 'yes' else 0
    kid_stone = 1 if kid_stone == 'yes' else 0
    skin_cancer = 1 if skin_cancer == 'yes' else 0

    with open('errorlog.txt', 'a') as f:
        f.write(f'Data gathered\n')

    # Create a dataframe for the data
    df = pd.DataFrame(data = {'BMI': [bmi], 
                              'Smoking': [smoking], 
                              'AlcoholDrinking': [drinking], 
                              'Stroke': [stroke],
                              'PhysicalHealth': [ph_health],
                              'MentalHealth': [men_health],
                              'DiffWalking': [stairs],
                              'Diabetic': [diabetes],
                              'PhysicalActivity': [exercise],
                              'SleepTime': [sleep],
                              'Asthma': [asthma],
                              'KidneyDisease': [kid_stone],
                              'SkinCancer': [skin_cancer],
                              'Sex_Female': [0 if gender == 'male' else 1],
                              'Sex_Male': [1 if gender == 'male' else 0],
                              'AgeCategory_18-24': [1 if age_group == '18-24' else 0],
                              'AgeCategory_25-29': [1 if age_group == '25-29' else 0],
                              'AgeCategory_30-34': [1 if age_group == '30-34' else 0],
                              'AgeCategory_35-39': [1 if age_group == '35-39' else 0],
                              'AgeCategory_40-44': [1 if age_group == '40-44' else 0],
                              'AgeCategory_45-49': [1 if age_group == '45-49' else 0],
                              'AgeCategory_50-54': [1 if age_group == '50-54' else 0],
                              'AgeCategory_55-59': [1 if age_group == '55-59' else 0],
                              'AgeCategory_60-64': [1 if age_group == '60-64' else 0],
                              'AgeCategory_65-69': [1 if age_group == '65-69' else 0],
                              'AgeCategory_70-74': [1 if age_group == '70-74' else 0],
                              'AgeCategory_75-79': [1 if age_group == '75-79' else 0],
                              'AgeCategory_80 or older': [1 if age_group == '80 or older' else 0],
                              'Race_American Indian/Alaskan Native': [1 if race == 'american indian/alaskan native' else 0],
                              'Race_Asian': [1 if race == 'asian' else 0],
                              'Race_Black': [1 if race == 'black' else 0],
                              'Race_Hispanic': [1 if race == 'hispanic' else 0],
                              'Race_Other': [1 if race == 'other' else 0],
                              'Race_White': [1 if race == 'white' else 0],
                              'GenHealth_Excellent': [1 if gen_health == 'excellent' else 0],
                              'GenHealth_Fair': [1 if gen_health == 'fair' else 0],
                              'GenHealth_Good': [1 if gen_health == 'good' else 0],
                              'GenHealth_Poor': [1 if gen_health == 'poor' else 0],
                              'GenHealth_Very good': [1 if gen_health == 'very good' else 0],
                              })
                              

    with open('errorlog.txt', 'a') as f:
        f.write(f'Dataframe Created\n')

    scaled_data = scaler.transform(df)
    
    with open('errorlog.txt', 'a') as f:
        f.write(f'{df.head()}\n')
        f.write(f'{scaler.scale_}\n')
        f.write(f'{scaled_data}\n')

    # try to make prediction, otherwise notify user the entries are invalid
    try:
        # make prediction
        #prediction = stroke_pred(g,a,hyt,ht,m,w,r,gl,b,s)
        prediction = model.predict(scaled_data)

        with open('errorlog.txt', 'a') as f:
            f.write(f'Prediction: {prediction[0]}')
        # render index_2 for result page
        return {'prediction': int(prediction[0])}

    except ValueError:
        return "Please Enter Valid Values"


if __name__=="__main__":
	app.run()
