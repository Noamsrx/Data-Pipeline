import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import os
 
DB_URI = "postgresql://iris_calcul_user:ZvWXZGrCqxZUtaCfpcSz5TGHATbbIgKB@dpg-d04v37be5dus738nt110-a.frankfurt-postgres.render.com:5432/iris_calcul"
engine = create_engine(DB_URI)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
 
# 1. Charger le fichier CSV (à remplacer plus tard par une connexion PostgreSQL)
import requests
response = requests.get("https://data-pipeline-lobo.onrender.com/iris")

 
 # appel vers l’API dans Docker
df = pd.DataFrame(response.json())
 
 
def load_data():
    try:
        query = "SELECT sepal_width, sepal_length FROM iris_measurements"
        df = pd.read_sql(query, engine)
        print("✅ Données chargées depuis PostgreSQL :")
        print(df.head())
        return df[["sepal_width", "sepal_length"]]
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données depuis PostgreSQL : {str(e)}")
        return None
 
 
def train_and_log_model(df):
    if df is None:
        return None
 
    X = df[["sepal_width"]]
    y = df["sepal_length"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)
 
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        signature = infer_signature(X_test, predictions)
 
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "model", signature=signature)
 
        print(f"✅ MSE: {mse}")
 
        example = pd.DataFrame([[3.5]], columns=["sepal_width"])
        pred = model.predict(example)
        print(f"🔍 Pour une largeur de 3.5 cm, la longueur prédite est : {pred[0]:.2f} cm")
 
        plt.figure(figsize=(8, 5))
        plt.scatter(X, y, label="Données réelles", color="blue")
        plt.plot(X, model.predict(X), color="red", label="Régression (modèle)")
        plt.xlabel("Largeur des sépales (sepal_width)")
        plt.ylabel("Longueur des sépales (sepal_length)")
        plt.title("Régression linéaire : sepal_width → sepal_length")
        plt.legend()
        plt.grid(True)
 
        graph_path = "sepal_regression_plot.png"
        plt.savefig(graph_path)
        plt.close()
 
        mlflow.log_artifact(graph_path)
        os.remove(graph_path)
 
        return mse
 
 
if __name__ == "__main__":
    data = load_data()
    train_and_log_model(data)