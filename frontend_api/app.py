from flask import Flask, request, jsonify
from flask_cors import CORS
from ml.regression_pipeline import load_data, train_and_log_model
import mlflow
import mlflow.sklearn
import pandas as pd
import os

app = Flask(__name__)
CORS(app) 

@app.route('/train', methods=['POST'])
def train():
    df = load_data()
    mse = train_and_log_model(df)
    return jsonify({"status": "ok", "mse": mse})

@app.route('/predict', methods=['POST'])
def predict():
    sepal_width = request.json.get("sepal_width")
    model = mlflow.sklearn.load_model("runs:/b20fd94bdc5a44e989f8c5770a575d98/model")
    df = pd.DataFrame([[sepal_width]], columns=["sepal_width"])
    pred = model.predict(df)[0]
    return jsonify({"prediction": round(pred, 2)})

@app.route('/', methods=['GET'])
def hello():
    return "LA CONNEXION EST BONNE OUAAAAIS!"


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8000))  # Render injecte la variable PORT
    app.run(host="0.0.0.0", port=port)         # écoute bien sur 0.0.0.0
