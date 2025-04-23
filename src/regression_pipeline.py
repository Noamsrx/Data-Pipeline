# import pandas as pd
# import mlflow
# import mlflow.sklearn
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from mlflow.models.signature import infer_signature  #

# # 1. Charger le fichier CSV
# df = pd.read_csv("Iris_Data.csv")  # Mets bien ton fichier au bon endroit + remplacer par l'api de postgree

# # 2. Définir X et y
# X = df[["sepal_width"]]
# y = df["sepal_length"]

# # 3. Split train/test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 4. Suivi MLflow
# with mlflow.start_run():
#     # Modèle
#     model = LinearRegression()
#     model.fit(X_train, y_train)

#     # Prédictions et évaluation
#     predictions = model.predict(X_test)
#     mse = mean_squared_error(y_test, predictions)

#     # ✅ Ajouter la signature (automatiquement à partir des données)
#     signature = infer_signature(X_test, predictions)

#     # Logging avec signature
#     mlflow.log_param("model", "LinearRegression")
#     mlflow.log_metric("mse", mse)
#     mlflow.sklearn.log_model(model, "model", signature=signature)

#     print(f"✅ MSE: {mse}")

#     # 🔍 Test de prédiction
#     example = pd.DataFrame([[3.5]], columns=["sepal_width"])  # ✅ pour éviter le warning
#     pred = model.predict(example)
#     print(f"🔍 Pour une largeur de 3.5 cm, la longueur prédite est : {pred[0]:.2f} cm")
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import os

# 1. Charger le fichier CSV (à remplacer plus tard par une connexion PostgreSQL)
df = pd.read_csv("Iris_Data.csv")

# 2. Définir X (entrée) et y (sortie)
X = df[["sepal_width"]]
y = df["sepal_length"]

# 3. Split en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Suivi avec MLflow
with mlflow.start_run():
    # Entraîner le modèle
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prédictions + métrique
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    # Signature
    signature = infer_signature(X_test, predictions)

    # Logging MLflow
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model", signature=signature)

    print(f"✅ MSE: {mse}")

    # 🔍 Test de prédiction manuelle
    example = pd.DataFrame([[3.5]], columns=["sepal_width"])
    pred = model.predict(example)
    print(f"🔍 Pour une largeur de 3.5 cm, la longueur prédite est : {pred[0]:.2f} cm")

    # 📈 Générer un graphe de l'entraînement
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, label="Données réelles", color="blue")
    plt.plot(X, model.predict(X), color="red", label="Régression (modèle)")
    plt.xlabel("Largeur des sépales (sepal_width)")
    plt.ylabel("Longueur des sépales (sepal_length)")
    plt.title("Régression linéaire : sepal_width → sepal_length")
    plt.legend()
    plt.grid(True)

    # Sauvegarder le graphe
    graph_path = "sepal_regression_plot.png"
    plt.savefig(graph_path)
    plt.close()

    # Logger le graphe dans MLflow
    mlflow.log_artifact(graph_path)

    # Supprimer le fichier local (optionnel)
    os.remove(graph_path)
