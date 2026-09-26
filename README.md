# London Housing Price Predictor – Databricks App

A Streamlit app on Databricks Apps to explore London house prices and estimate property values.

| Tab | What it does |
|---|---|
| Market overview | KPIs, quarterly price trend by property type, £/sqft by borough, sales map |
| Price predictor | Enter property details and get an estimate, an 80% range, a comparables histogram and the closest sales |
| Model insights | Hold-out R², MAE, MAPE, actual vs predicted, feature importance |
| Data | Filtered records with CSV download |

## Project structure
```
app.py                     Streamlit app
app.yaml                   Databricks Apps config (command + env vars)
requirements.txt
data/london_housing.csv    Sample data: 15,000 sales in 33 local authorities, 2019–2026
notebooks/01_train_model.py  Optional: Delta table + MLflow model + serving endpoint
generate_data.py           Regenerates the sample CSV
```

## Deploy (about 5 minutes)
1. **Upload**: import this folder into your workspace, e.g. `/Workspace/Users/<you>/london-housing-app`.
2. **Create the app**: go to **Compute → Apps → Create app → Custom**, name it `london-housing`.
3. **Deploy**: point the app at the folder, or use the CLI:
   ```bash
   databricks sync . /Workspace/Users/<you>/london-housing-app
   databricks apps deploy london-housing --source-code-path /Workspace/Users/<you>/london-housing-app
   ```
Out of the box, the app runs on the bundled CSV and trains its model at startup, so you don't need any other resources.

## Optional: production mode
1. Run `notebooks/01_train_model.py`. It creates the Delta table `main.housing.london_sales`, registers `main.housing.london_house_price` in Unity Catalog, and creates the serving endpoint `london-house-price`.
2. In the app's settings, add the resources:
   - A **SQL warehouse** (key `sql-warehouse`, *Can use*)
   - The **serving endpoint** (*Can query*)
3. Grant the app's service principal `SELECT` on the table.
4. Uncomment the env vars in `app.yaml` and redeploy.

## Using real data
Replace the CSV or table with real London data, for example HM Land Registry Price Paid Data joined to EPC certificates for floor area and rating, plus TfL station locations for distance to the tube. Keep the same column names:
`sale_date, borough, property_type, tenure, new_build, bedrooms, bathrooms, floor_area_sqft, year_built, epc_rating, has_garden, has_parking, dist_to_tube_km, latitude, longitude, price_gbp`

> The bundled data is synthetic. It is realistic in structure and borough pricing, but it is **not** real transaction data.
