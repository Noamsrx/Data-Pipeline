import pandas as pd
from sqlalchemy import create_engine
import os

csv_path = "iris.csv"

user = os.getenv("POSTGRES_USER", "postgres")
password = os.getenv("POSTGRES_PASSWORD", "postgres")
host = "db"  # service Docker
port = "5432"
database = os.getenv("POSTGRES_DB", "iris_calcul")

engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")

df = pd.read_csv(csv_path)
print("Aperçu des données :")
print(df.head())

df.to_sql("iris_measurements", engine, if_exists="replace", index=False)
print("✅ Données insérées dans la table 'iris_measurements'")
