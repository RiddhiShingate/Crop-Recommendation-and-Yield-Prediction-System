from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import sklearn

import pickle
import joblib


#importing model
model = pickle.load(open('model.pkl','rb'));

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file, protocol=4)

#loading models
dtr = pickle.load(open('dtr.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))

# creating flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')




@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            N = int(request.form['Nitrogen'])
            P = int(request.form['Phosporus'])
            K = int(request.form['Potassium'])
            temp = float(request.form['Temperature'])
            humidity = float(request.form['Humidity'])
            ph = float(request.form['Ph'])
            rainfall = float(request.form['Rainfall'])

            data = np.array([[N, P, K, temp, humidity, ph, rainfall]])
            my_prediction = model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('index.html', result=final_prediction)
        except KeyError:
            # Handle missing form field error
            return "Error: Some form fields are missing. Please fill out all fields and try again."
    else:

        return render_template('index.html')

@app.route("/yeildPredict", methods=['GET', 'POST'])
def yeildPredict():
    if request.method == 'POST':
        Year = request.form['Year']
        Production = request.form['production']
        Area = request.form['Area']
        Crop = request.form['Crop']
        Season = request.form['Season']
        Area=Area;
        features = np.array([[Year, Area, Production, Season, Crop]], dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1, -1)

        return render_template('predict.html', prediction=prediction)
    else:

        return render_template('predict.html')
# python main
if __name__ == "__main__":
    app.run(debug=True)