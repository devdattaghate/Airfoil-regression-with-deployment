import pickle
from flask import Flask, request, jsonify

import numpy as np
import pandas as pd

app= Flask(__name__)
model= pickle.load(open('model.pkl', 'rb'))

@app.route('/predict_api',methods=['POST'])
def predict_api():

    data= request.json["data"]
    print(data)
    new_data = np.reshape([list(data.values())],(1,-1))
    output= model.predict(new_data)[0]
    return jsonify(output)

if __name__=="__main__":
    app.run(debug=True)
