import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import tensorflow as tf


app=Flask(__name__)
## Load the model
model=pickle.load(open('NN_Model.pkl','rb'))
scalar=pickle.load(open('scalling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)

    predictions_probs = tf.math.sigmoid(output)
    predictions = np.where(predictions_probs >= 0.5, 1, 0)

    print(predictions[0])

    # Convert the NumPy array to a list before returning the JSON response
    output_list = predictions[0].tolist() if isinstance(output[0], np.ndarray) else predictions[0]
    return jsonify(output_list)
    

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]

    predictions_probs = tf.math.sigmoid(output)
    predictions = np.where(predictions_probs >= 0.5, 1, 0)


    # Convert the NumPy array to a list if it's an ndarray
    output_list = predictions.tolist() if isinstance(output, np.ndarray) else predictions
    return render_template("home.html", prediction_text="The output is: {}".format(output_list))
    



if __name__=="__main__":
    app.run(debug=True)
