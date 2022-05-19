
# importing the necessary dependencies
import numpy as np #used for numerical analysis
import pandas as pd # used for data manipulation
from flask import Flask, render_template, request
# Flask-It is our framework which we are going to use to run/serve our application.
#request-for accessing file which was uploaded by the user on our application.
import pickle

 
import json
import requests
import os
# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "3QXKp5BxGwIzwS0TySElux6PSEZvCqbmnPkwZrkjz2c6"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

# NOTE: manually define and pass the array(s) of values to be scored in the next line

app = Flask(__name__) # initializing a flask app
model = pickle.load(open('fitness.pkl','rb')) #loading the model

@app.route('/')# route to display the home page
def home():
    return render_template('home.html') #rendering the home page
@app.route('/Prediction',methods=['POST','GET'])
def prediction():
    return render_template('indexnew.html')
@app.route('/Home',methods=['POST','GET'])
def my_home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])# route to show the predictions in a web UI

def predict():
    
    #reading the inputs given by the user
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['sad','neutral','happy','step_count',
                    'calories_burned','hours_of_sleep','weight_kg']
    
    
    df = pd.DataFrame(features_value, columns=features_name)
    
    # predictions using the loaded model file
    output = model.predict(df)
    
    # showing the prediction results in a UI# showing the prediction results in a UI
    return render_template('result.html', prediction_text=output)
    payload_scoring = {"input_data": [{"fields": [['sad','neutral','happy','step_count',
                    'calories_burned','hours_of_sleep','weight_kg']], "values": input_features}]}

    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/51541a6f-cd48-45e7-86bb-6eb9ff91e676/predictions?version=2022-03-26', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    print(response_scoring.json())
if __name__ == '__main__':
    # running the app
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=False,use_reloader=False)

 

