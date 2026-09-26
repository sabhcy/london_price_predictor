"""
London Housing Price Predictor - Databricks App (Streamlit)

Data source (auto-selected):
  1. Unity Catalog table via SQL Warehouse   -> set HOUSING_TABLE + DATABRICKS_WAREHOUSE_ID
  2. Bundled sample CSV (data/london_housing.csv) -> fallback, works out of the box

Model (auto-selected):
  1. Model Serving endpoint (trained by notebooks/01_train_model.py) -> set SERVING_ENDPOINT
  2. In-app Gradient Boosting model trained on load -> fallback
"""
import os
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

st.set_page_config(page_title="London Housing Price Predictor", page_icon="🏠", layout="wide")

HOUSING_TABLE = os.getenv("HOUSING_TABLE", "")          # e.g. main.housing.london_sales
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
SERVING_ENDPOINT = os.getenv("SERVING_ENDPOINT", "")    # e.g. london-house-price
LOCAL_CSV = os.path.join(os.path.dirname(__file__), "data", "london_housing.csv")

CAT_COLS = ["borough", "property_type", "tenure", "epc_rating"]
NUM_COLS = ["bedrooms", "bathrooms", "floor_area_sqft", "year_built", "dist_to_tube_km",
            "new_build", "has_garden", "has_parking", "sale_year", "sale_month"]
FEATURES = CAT_COLS + NUM_COLS
TARGET = "price_gbp"


# --------------------------------------------------------------------------- data
@st.cache_data(ttl=3600, show_spinner="Loading London housing data...")
def load_data() -> tuple[pd.DataFrame, str]:
    if HOUSING_TABLE and WAREHOUSE_ID:
        try:
            from databricks import sql
            from databricks.sdk.core import Config

            cfg = Config()  # picks up the app's service principal automatically
            with sql.connect(server_hostname=cfg.host,
                             http_path=f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
                             credentials_provider=lambda: cfg.authenticate) as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT * FROM {HOUSING_TABLE}")
                    df = cur.fetchall_arrow().to_pandas()
            return prep(df), f"Unity Catalog: `{HOUSING_TABLE}`"
        except Exception as e:  # fall back gracefully
            st.warning(f"Could not read {HOUSING_TABLE} ({e}). Using sample data.")
    return prep(pd.read_csv(LOCAL_CSV)), "Bundled sample dataset"


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["sale_year"] = df["sale_date"].dt.year
    df["sale_month"] = df["sale_date"].dt.month
    for c in ["new_build", "has_garden", "has_parking"]:
        df[c] = df[c].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
    df["price_per_sqft"] = df[TARGET] / df["floor_area_sqft"]
    return df


# --------------------------------------------------------------------------- model
@st.cache_resource(show_spinner="Training price model...")
def train_model(df: pd.DataFrame):
    X, y = df[FEATURES], np.log(df[TARGET])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pre = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                    unknown_value=-1), CAT_COLS)],
                            remainder="passthrough")
    model = Pipeline([("pre", pre),
                      ("gbm", HistGradientBoostingRegressor(
                          max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                          categorical_features=list(range(len(CAT_COLS))), random_state=42))])
    model.fit(X_tr, y_tr)

    pred_log = model.predict(X_te)
    actual, pred = np.exp(y_te), np.exp(pred_log)
    resid_sd = float(np.std(y_te - pred_log))
    metrics = {"R²": r2_score(actual, pred),
               "MAE": mean_absolute_error(actual, pred),
               "MAPE": mean_absolute_percentage_error(actual, pred),
               "resid_sd": resid_sd}
    sample = X_te.sample(min(1500, len(X_te)), random_state=0)
    imp = permutation_importance(model, sample, y_te.loc[sample.index],
                                 n_repeats=3, random_state=0)
    importance = (pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean})
                  .sort_values("importance"))
    scatter = pd.DataFrame({"Actual": actual.values, "Predicted": pred,
                            "borough": X_te["borough"].values})
    return model, metrics, importance, scatter


def predict(model, row: pd.DataFrame) -> float:
    if SERVING_ENDPOINT:
        try:
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()
            resp = w.serving_endpoints.query(name=SERVING_ENDPOINT,
                                             dataframe_records=row[FEATURES].to_dict("records"))
            return float(np.exp(resp.predictions[0]))
        except Exception as e:
            st.info(f"Serving endpoint unavailable ({e}); using in-app model.")
    return float(np.exp(model.predict(row[FEATURES])[0]))


gbp = lambda v: f"£{v:,.0f}"

# --------------------------------------------------------------------------- UI
df, source = load_data()
model, metrics, importance, scatter = train_model(df)

st.title("🏠 London Housing Price Predictor")
st.caption(f"Data source: {source} · {len(df):,} sales · "
           f"{df.sale_date.min():%b %Y} – {df.sale_date.max():%b %Y}")

with st.sidebar:
    st.header("Filters")
    boroughs = st.multiselect("Borough", sorted(df.borough.unique()))
    ptypes = st.multiselect("Property type", sorted(df.property_type.unique()))
    yr = st.slider("Sale year", int(df.sale_year.min()), int(df.sale_year.max()),
                   (int(df.sale_year.min()), int(df.sale_year.max())))
    f = df[df.sale_year.between(*yr)]
    if boroughs: f = f[f.borough.isin(boroughs)]
    if ptypes:   f = f[f.property_type.isin(ptypes)]
    st.metric("Sales in selection", f"{len(f):,}")

tab_mkt, tab_pred, tab_model, tab_data = st.tabs(
    ["📊 Market overview", "🔮 Price predictor", "🧠 Model insights", "📄 Data"])

# ---- Market overview
with tab_mkt:
    if f.empty:
        st.warning("No sales match the filters.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Median price", gbp(f[TARGET].median()))
        c2.metric("Median £/sqft", gbp(f.price_per_sqft.median()))
        c3.metric("Median size", f"{f.floor_area_sqft.median():,.0f} sqft")
        yoy = f.groupby("sale_year")[TARGET].median()
        c4.metric("Latest YoY change",
                  f"{(yoy.iloc[-1]/yoy.iloc[-2]-1):+.1%}" if len(yoy) > 1 else "n/a")

        trend = (f.set_index("sale_date").groupby([pd.Grouper(freq="QE"), "property_type"])
                 [TARGET].median().reset_index())
        st.plotly_chart(px.line(trend, x="sale_date", y=TARGET, color="property_type",
                                title="Median price by quarter",
                                labels={"sale_date": "", TARGET: "Median price (£)"}),
                        width="stretch")

        l, r = st.columns(2)
        by_b = (f.groupby("borough").agg(median_ppsf=("price_per_sqft", "median"),
                                          sales=(TARGET, "size")).reset_index()
                .sort_values("median_ppsf"))
        l.plotly_chart(px.bar(by_b, x="median_ppsf", y="borough", orientation="h",
                              height=750, title="Median £/sqft by borough",
                              labels={"median_ppsf": "£/sqft", "borough": ""}),
                       width="stretch")
        r.plotly_chart(px.scatter_map(f.sample(min(3000, len(f)), random_state=1),
                                      lat="latitude", lon="longitude", color="price_per_sqft",
                                      color_continuous_scale="Viridis", zoom=9, height=750,
                                      hover_data=["borough", "property_type", TARGET],
                                      map_style="carto-positron", title="Sales map (£/sqft)"),
                       width="stretch")

# ---- Predictor
with tab_pred:
    st.subheader("Estimate the value of a London property")
    with st.form("predict"):
        a, b, c = st.columns(3)
        borough = a.selectbox("Borough", sorted(df.borough.unique()))
        ptype = a.selectbox("Property type", ["Flat", "Terraced", "Semi-Detached", "Detached"])
        tenure = a.selectbox("Tenure", ["Leasehold", "Freehold"])
        beds = b.number_input("Bedrooms", 0, 8, 2)
        baths = b.number_input("Bathrooms", 1, 6, 1)
        sqft = b.number_input("Floor area (sqft)", 250, 6000, 800, step=25)
        built = c.number_input("Year built", 1800, date.today().year, 1930)
        epc = c.selectbox("EPC rating", list("ABCDEFG"), index=2)
        tube = c.slider("Distance to nearest tube/rail (km)", 0.0, 5.0, 0.6, 0.1)
        d1, d2, d3 = st.columns(3)
        new_build = d1.checkbox("New build")
        garden = d2.checkbox("Garden", value=ptype != "Flat")
        parking = d3.checkbox("Parking")
        go = st.form_submit_button("Predict price", type="primary")

    if go:
        today = pd.Timestamp.today()
        row = pd.DataFrame([dict(borough=borough, property_type=ptype, tenure=tenure,
                                 epc_rating=epc, bedrooms=beds, bathrooms=baths,
                                 floor_area_sqft=sqft, year_built=built, dist_to_tube_km=tube,
                                 new_build=int(new_build), has_garden=int(garden),
                                 has_parking=int(parking),
                                 sale_year=min(today.year, int(df.sale_year.max())),
                                 sale_month=today.month)])
        price = predict(model, row)
        z = 1.2816 * metrics["resid_sd"]  # 80% interval on log scale
        lo, hi = price * np.exp(-z), price * np.exp(z)
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated price", gbp(price))
        m2.metric("80% range", f"{gbp(lo)} – {gbp(hi)}")
        m3.metric("Implied £/sqft", gbp(price / sqft))

        comps = df[(df.borough == borough) & (df.property_type == ptype)]
        if len(comps):
            fig = px.histogram(comps, x=TARGET, nbins=40,
                               title=f"{ptype} sales in {borough} (n={len(comps):,})",
                               labels={TARGET: "Sale price (£)"})
            fig.add_vline(x=price, line_dash="dash", line_color="red",
                          annotation_text="Your estimate")
            st.plotly_chart(fig, width="stretch")
            st.caption("Comparable recent sales")
            st.dataframe(comps.assign(gap=(comps.floor_area_sqft - sqft).abs())
                         .nsmallest(10, "gap")
                         [["sale_date", "bedrooms", "floor_area_sqft", "epc_rating",
                           "dist_to_tube_km", TARGET]], hide_index=True, width="stretch")

# ---- Model insights
with tab_model:
    st.subheader("Model performance (20% hold-out)")
    k1, k2, k3 = st.columns(3)
    k1.metric("R²", f"{metrics['R²']:.3f}")
    k2.metric("Mean absolute error", gbp(metrics["MAE"]))
    k3.metric("MAPE", f"{metrics['MAPE']:.1%}")
    l, r = st.columns(2)
    fig = px.scatter(scatter, x="Actual", y="Predicted", opacity=0.35, log_x=True, log_y=True,
                     title="Actual vs predicted", hover_data=["borough"])
    mn, mx = scatter.Actual.min(), scatter.Actual.max()
    fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx, line=dict(color="red", dash="dash"))
    l.plotly_chart(fig, width="stretch")
    r.plotly_chart(px.bar(importance, x="importance", y="feature", orientation="h",
                          title="Feature importance (permutation)"), width="stretch")
    st.caption("Model: gradient-boosted trees on log(price). "
               + ("Predictions served from endpoint `" + SERVING_ENDPOINT + "`."
                  if SERVING_ENDPOINT else "Trained in-app at startup."))

# ---- Data
with tab_data:
    st.dataframe(f.drop(columns=["sale_year", "sale_month"]), hide_index=True,
                 width="stretch", height=500)
    st.download_button("Download filtered data (CSV)", f.to_csv(index=False),
                       "london_housing_filtered.csv", "text/csv")
