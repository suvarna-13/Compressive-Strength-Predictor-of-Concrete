import numpy as np
import pandas as pd
from keras.models import load_model
from flask import Flask, render_template, request
import pickle
import sklearn
# import logging

"""Application logging"""

# logging.basicConfig(filename='deployment_logs.log', level=logging.INFO,
#                     format='%(levelname)s:%(asctime)s:%(message)s')  # configuring logging operations

app = Flask(__name__)

# model = pickle.load(open(r'models\XGBoost_Regressor_model.pkl','rb'))  # loading the saved XGBoost_regressor model
# model = pickle.load(open('models\XGBoost_Regressor_model.pkl','rb'))  # loading the saved XGBoost_regressor model
#model = load_model('compressive.h5') # loading the saved XGBoost_regressor model
model = pickle.load(open('concrete_strength.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    For rendering results on HTML GUI
    """
    if request.method == "POST":
        # ['age', 'cement', 'water', 'fly_ash', 'superplasticizer', 'blast_furnace_slag']
        Age = request.form.get('age')
        Cement = request.form.get('cement')
        Water = request.form.get('water')
        FlyAsh = request.form.get('fa')
        SuperPlasticizer = request.form.get('sp')
        Slag = request.form.get('bfs')
        CoarseAggr = request.form.get('cag')
        FineAggr = request.form.get('fag')



        # logging operation
#         logging.info(f"Age (in days): {f_list[0]}, Cement (in kg): {f_list[1]},"
#                      f"Water (in kg): {f_list[2]}, Fly ash (in kg): {f_list[3]},"
#                      f"Superplasticizer (in kg): {f_list[4]}, Blast furnace slag (in kg): {f_list[5]}")


        result = model.predict(np.array([[Cement, Slag, FlyAsh, Water, SuperPlasticizer, CoarseAggr, FineAggr, Age]]))
        res = "%.2f" % round(result[0], 2)


        # logging operation
#         logging.info(f"The Predicted Concrete Compressive strength is {result} MPa")

#         logging.info("Prediction getting posted to the web page.")
        return render_template('index.html',
                               prediction_text=f"The Concrete compressive strength is {res} MPa")


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)