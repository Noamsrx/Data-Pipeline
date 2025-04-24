from flask import Flask, jsonify
from sqlalchemy import create_engine
import pandas as pd
import os

app = Flask(__name__)

# Connexion PostgreSQL via nom du service Docker
db_url = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@db:5432/iris_calcul")
engine = create_engine(db_url)

@app.route("/iris", methods=["GET"])
def get_iris_data():
    try:
        df = pd.read_sql("SELECT * FROM iris_measurements", engine)
        print(f"✅ Données récupérées ({len(df)} lignes)")
        return df.to_json(orient="records")
    except Exception as e:
        print(f"❌ Erreur dans /iris : {e}")
        return jsonify({"error": str(e)}), 500

# 🔥 Ce bloc permet de lancer le serveur sur toutes les interfaces réseau (Docker friendly)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
