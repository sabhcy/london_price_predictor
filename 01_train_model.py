# Databricks notebook source
# MAGIC %md
# MAGIC # London Housing – Load data, train & serve model
# MAGIC 1. Loads the CSV into a Unity Catalog Delta table
# MAGIC 2. Trains a gradient-boosting model on log(price) with MLflow tracking
# MAGIC 3. Registers the model in Unity Catalog and (optionally) creates a serving endpoint
# MAGIC
# MAGIC Swap the CSV for real data (e.g. HM Land Registry Price Paid Data joined with EPC data) –
# MAGIC just keep the same column names.

# COMMAND ----------
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "housing")
dbutils.widgets.text("csv_path", "/Workspace/Users/<you>/london-housing-app/data/london_housing.csv")
dbutils.widgets.text("endpoint", "london-house-price")
CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
TABLE = f"{CATALOG}.{SCHEMA}.london_sales"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.london_house_price"

# COMMAND ----------
import pandas as pd
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
pdf = pd.read_csv(dbutils.widgets.get("csv_path"), parse_dates=["sale_date"])
spark.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TABLE)
display(spark.table(TABLE).limit(10))

# COMMAND ----------
import mlflow, numpy as np
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

CAT = ["borough", "property_type", "tenure", "epc_rating"]
NUM = ["bedrooms", "bathrooms", "floor_area_sqft", "year_built", "dist_to_tube_km",
       "new_build", "has_garden", "has_parking", "sale_year", "sale_month"]

df = spark.table(TABLE).toPandas()
df["sale_date"] = pd.to_datetime(df["sale_date"])
df["sale_year"], df["sale_month"] = df.sale_date.dt.year, df.sale_date.dt.month
for c in ["new_build", "has_garden", "has_parking"]:
    df[c] = df[c].astype(int)
X, y = df[CAT + NUM], np.log(df["price_gbp"])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_registry_uri("databricks-uc")
with mlflow.start_run(run_name="london_hgb") as run:
    params = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31)
    pipe = Pipeline([
        ("pre", ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1), CAT)], remainder="passthrough")),
        ("gbm", HistGradientBoostingRegressor(categorical_features=list(range(len(CAT))),
                                              random_state=42, **params))])
    pipe.fit(X_tr, y_tr)
    pred = np.exp(pipe.predict(X_te)); actual = np.exp(y_te)
    mlflow.log_params(params)
    mlflow.log_metrics({"r2": r2_score(actual, pred), "mae": mean_absolute_error(actual, pred),
                        "mape": mean_absolute_percentage_error(actual, pred)})
    mlflow.sklearn.log_model(pipe, "model", registered_model_name=MODEL_NAME,
                             signature=infer_signature(X_tr, pipe.predict(X_tr)),
                             input_example=X_tr.head(3))
print("Registered", MODEL_NAME)

# COMMAND ----------
# MAGIC %md ### Optional: create a serving endpoint (model returns log(price); the app applies exp)

# COMMAND ----------
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from mlflow import MlflowClient

w = WorkspaceClient()
version = max(int(v.version) for v in MlflowClient().search_model_versions(f"name='{MODEL_NAME}'"))
w.serving_endpoints.create_and_wait(
    name=dbutils.widgets.get("endpoint"),
    config=EndpointCoreConfigInput(served_entities=[ServedEntityInput(
        entity_name=MODEL_NAME, entity_version=str(version),
        workload_size="Small", scale_to_zero_enabled=True)]))
