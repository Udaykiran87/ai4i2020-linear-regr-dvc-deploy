import joblib
from flask import Flask, render_template, request, jsonify
import os
import yaml
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from joblib import Parallel, delayed
import time
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso,RidgeCV,LassoCV, ElasticNet , ElasticNetCV,LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


import logging

params_path = "params.yaml"
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
        # rotational_speed = float(request.form['Rotational_speed'])
        # torque = float(request.form['Torque'])
        # tool_wear = float(request.form['Tool_wear'])
        # machine_failure = float(request.form['Machine_failure'])
        # twf = float(request.form['TWF'])
        # hdf = float(request.form['HDF'])
        # pwf = float(request.form['PWF'])
        # osf = float(request.form['OSF'])
        # rnf = float(request.form['RNF'])
        # if request.form.get("Predict_Air_Temp_using_Linear_Regr"):
        #     file = 'linear_reg.sav'
        # elif request.form.get("Predict_Air_Temp_using_Lasso_Linear_Regr"):
        #     file = 'lasso_linear_reg.sav'
        # elif request.form.get("Predict_Air_Temp_using_Ridge_Linear_Regr"):
        #     file = 'ridge_linear_reg.sav'
        # elif request.form.get("Predict_Air_Temp_using_Elasticnet_Linear_Regr"):
        #     file = 'elastic_linear_reg.sav'
        # saved_model = pickle.load(open(file, 'rb'))
        # prediction = saved_model.predict([[rotational_speed,torque,tool_wear,machine_failure,twf,hdf,pwf,osf,rnf]])
        if request.form.get("Predict_Air_Temp_using_Elasticnet_Linear_Regr"):
            data = dict(request.form)
            data.pop("Predict_Air_Temp_using_Elasticnet_Linear_Regr")
            data = [list(map(float, data.values()))]
            response = predict(data)
        return render_template('results.html', response=response)

    if request.json:
        response = api_response(request)
        return jsonify(response)

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def predict(data):
    config = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    model = joblib.load(model_dir_path)
    prediction = model.predict(data)
    print(prediction)
    return prediction[0]

def api_response(request):
    try:
        data = np.array([list(request.json.values())])
        response = predict(data)
        response = {"resonse":response}
        return response
    except Exception as e:
        print(e)
        error = {"error": "Something went wrong!! Try again"}
        return error

@app.route('/profie_report/', methods=['POST', 'GET'])
def profie_report():
    return render_template('ori_df_profiling.html')


if __name__ == '__main__':
    # linear_regr = Linearregression('challange_dataset.csv', 'Air temperature [K]')
    #
    # # load_data()
    # linear_regr.load_data()
    #
    # # profiling_data()
    # linear_regr.pandas_profiling('ori_df_profiling.html')
    #
    # # fillna()
    # nan_count = linear_regr.check_NaN()
    # print(nan_count)
    #
    # # handle_multicolinearity()
    # linear_regr.view_multicolinearity_by_vif()
    # linear_regr.drop_multicolinearity_by_vif(vif_thresh=10)
    #
    # # create independent feature and dependent feature
    # linear_regr.create_X_Y()
    #
    # # Standardization
    # linear_regr.standardize_train()
    #
    # # Split dataset
    # linear_regr.train_test_split(test_size=0.15, random_state=100)
    #
    # # build_model()
    # linear_regr.build_model()
    #
    # # save_model()
    # linear_regr.save_model('linear_reg.sav')
    #
    # # model_accuracy()
    # accuracy = linear_regr.calc_accuracy()
    # print(accuracy)
    #
    # # build_lasso_model()
    # linear_regr.build_lasso_model(cv=10, max_iter=20000)
    #
    # # save_lasso_model()
    # linear_regr.save_lasso_model('lasso_linear_reg.sav')
    #
    # # lasso_model_accuracy()
    # lasso_accuracy = linear_regr.calc_lasso_accuracy()
    # print(accuracy)
    #
    # # build_ridge_model()
    # linear_regr.build_ridge_model(cv=10)
    #
    # # save_ridge_model()
    # linear_regr.save_ridge_model('ridge_linear_reg.sav')
    #
    # # ridge_model_accuracy()
    # ridge_accuracy = linear_regr.calc_ridge_accuracy()
    # print(ridge_accuracy)
    #
    # # build_elasticnet_model()
    # linear_regr.build_elasticnet_model(cv=10)
    #
    # # save_elasticnet_model()
    # linear_regr.save_elasticnet_model('elastic_linear_reg.sav')
    #
    # # elasticnet_model_accuracy()
    # elasticnet_accuracy = linear_regr.calc_elasticnet_accuracy()
    # print(elasticnet_accuracy)

    app.run(host='localhost', port=5000)