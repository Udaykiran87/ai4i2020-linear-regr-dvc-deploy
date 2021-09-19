import joblib
from flask import Flask, render_template, request, jsonify
import os
import yaml
import numpy as np
from prediction_service import prediction
import logging

webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)

@app.route('/')
def form():
    return render_template('form.html')


@app.route('/data/', methods=['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"

    if request.method == "POST":
        try:
            if request.form.get("Predict_Air_Temp_using_Elasticnet_Linear_Regr"):
                data_req = dict(request.form)
                data_req.pop("Predict_Air_Temp_using_Elasticnet_Linear_Regr")
                response = prediction.form_response(data)
                return render_template('results.html', response=response)
            elif request.json:
                response = prediction.api_response(request.json)
                return jsonify(response)
        except Exception as e:
            print(e)
            error = {"error": "Something went wrong!! Try again later!"}
            error = {"error": e}
            return render_template("404.html", error=error)

@app.route('/profie_report/', methods=['POST', 'GET'])
def profie_report():
    return render_template('ori_df_profiling.html')

if __name__ == '__main__':
    app.run(host='localhost', port=5000)