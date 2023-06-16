from flask import Flask, request, jsonify, make_response, render_template
import joblib
import json
import pandas as pd
import numpy as np
from airbnb.airbnb import Airbnb

model = joblib.load('./model/extratrees.joblib')
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predict', methods = ['POST'])
def predict_country():

    try:
        payload = request.get_json()

        if payload:
            df_raw = pd.DataFrame(json.loads(payload))

            pipeline = Airbnb()

            df_sessions = pipeline.load_data()
            df, df_sessions = pipeline.transform_data(df_raw, df_sessions)
            df = pipeline.feature_engineering(df_raw, df_sessions)
            X = pipeline.data_preprocessing(df)

            predicted_country_destinations_proba = pipeline.predict(model, X)

            classes = model.classes_
            predicted_classes = np.argmax(predicted_country_destinations_proba, axis=1)
            class_proba = np.max(predicted_country_destinations_proba, axis=1)
            predicted_class_names = [classes[i] for i in predicted_classes]

            df['country_destination'] = predicted_class_names
            df['proba'] = class_proba

            response = df[['id', 'country_destination', 'proba']].to_json(orient='records')
            response = make_response(
                jsonify(
                    response),
                200,
                )
            response.headers["Content-Type"] = "application/json"
            return response

        else:
            response = make_response(
                jsonify(
                    {"message": str(e), "Error": "No payload to predict"}),
                400,
                )
            response.headers["Content-Type"] = "application/json"
            return response

    except Exception as e:
        response = make_response(
                jsonify(
                    {"message": str(e), "Error": "danger"}),
                500,
            )
        response.headers["Content-Type"] = "application/json"
        return response
    

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port = 5000 )
