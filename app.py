# BUILD: green-ivory v3
import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import streamlit as st


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="House Price AI",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# 2. BRAND PALETTE  (Green / Tangerine Yellow / Ivory)
# ==================================================

GREEN = "#009B4D"       # primary: buttons, active tabs, borders, main chart colour
YELLOW = "#FFCC00"      # accent: highlights, shadows, secondary chart colour
IVORY = "#FAF5E9"       # background / surfaces
INK = "#00381C"         # very dark shade of the green, used ONLY for body text (readability)
MUTED = "rgba(0,56,28,0.65)"   # INK at reduced opacity for secondary text

PALETTE = [GREEN, YELLOW, "rgba(0,155,77,0.55)", "rgba(255,204,0,0.60)", "rgba(0,155,77,0.30)", INK]
SCALE = [[0.0, YELLOW], [1.0, GREEN]]


# ==================================================
# 3. PATHS & CONSTANTS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

DT_MODEL_PATH = MODEL_DIR / "decision_tree_pipeline.pkl"
LR_MODEL_PATH = MODEL_DIR / "linear_regression_pipeline.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.csv"
DATA_PATHS = [BASE_DIR / "data" / "housing.csv", BASE_DIR / "housing.csv"]

TARGET = "median_house_value"
PRICE_CAP = 500001  # the California dataset caps prices at this value
OCEAN_OPTIONS = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
MODEL_CHOICES = ["Decision Tree", "Linear Regression", "Compare Both"]

# Streamlit >= 1.50 replaced use_container_width with width="stretch"
_v = re.match(r"(\d+)\.(\d+)", st.__version__)
NEW_API = (int(_v[1]), int(_v[2])) >= (1, 50) if _v else False
W = {"width": "stretch"} if NEW_API else {"use_container_width": True}


# ==================================================
# 4. CUSTOM CSS
# ==================================================

CSS = """
<style>
.stApp {
    background:
        radial-gradient(1100px 560px at 90% -10%, rgba(255,204,0,.18), transparent 60%),
        radial-gradient(900px 500px at -10% 110%, rgba(0,155,77,.10), transparent 60%),
        __IVORY__;
    color: __INK__;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: __IVORY__;
    border-right: 2px solid __GREEN__;
}
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }

h1, h2, h3, h4 { color: __GREEN__ !important; letter-spacing: .2px; }
p, label, li { color: __INK__; }
hr { border-color: rgba(0,155,77,.35) !important; }

/* ---------- hero ---------- */
.hero {
    background: linear-gradient(120deg, rgba(0,155,77,.22) 0%, __IVORY__ 70%);
    border: 2px solid __GREEN__;
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 1.2rem;
    box-shadow: 6px 6px 0 __YELLOW__;
}
.hero-title { font-size: 2.1rem; font-weight: 800; color: __GREEN__; line-height: 1.15; }
.hero-sub   { font-size: 1.02rem; color: __INK__; opacity: .85; margin-top: 6px; }

/* ---------- metrics ---------- */
[data-testid="stMetric"] {
    background: __IVORY__;
    border: 1.5px solid __GREEN__;
    border-left: 6px solid __YELLOW__;
    padding: 16px 18px;
    border-radius: 14px;
    box-shadow: 3px 3px 0 rgba(0,155,77,.55);
}
[data-testid="stMetricLabel"] p {
    color: __MUTED__; font-weight: 700; font-size: .74rem;
    text-transform: uppercase; letter-spacing: .9px;
}
[data-testid="stMetricValue"] { color: __GREEN__; }

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button,
[data-testid="stFormSubmitButton"] > button {
    background: __GREEN__; color: __IVORY__; border: 1.5px solid __GREEN__;
    border-radius: 10px; font-weight: 600; min-height: 44px;
    transition: all .2s ease;
}
.stButton > button p, .stDownloadButton > button p,
[data-testid="stFormSubmitButton"] > button p { color: inherit !important; }
.stButton > button:hover, .stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    background: __YELLOW__; color: __INK__; border-color: __GREEN__;
    transform: translateY(-1px); box-shadow: 3px 3px 0 __GREEN__;
}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: __IVORY__; border: 1.5px solid __GREEN__; border-bottom: none;
    border-radius: 10px 10px 0 0; padding: 8px 18px;
}
.stTabs [aria-selected="true"] { background: __GREEN__; }
.stTabs [aria-selected="true"] p { color: __IVORY__ !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: __YELLOW__; height: 4px; }

/* ---------- forms, expanders, dataframes ---------- */
[data-testid="stForm"] {
    border: 1.5px solid __GREEN__;
    border-radius: 16px;
    background: __IVORY__;
    padding: 1.2rem 1.3rem;
}
[data-testid="stExpander"] {
    border: 1.5px solid __GREEN__; border-radius: 12px; background: __IVORY__;
}
[data-testid="stDataFrame"] { border: 1.5px solid __GREEN__; border-radius: 10px; }

/* ---------- price card ---------- */
.price-card {
    background: __IVORY__;
    border: 2px solid __GREEN__; border-radius: 18px;
    padding: 26px; text-align: center; margin: .4rem 0 1rem 0;
    box-shadow: 6px 6px 0 __YELLOW__;
}
.price-label { color: __MUTED__; font-size: .85rem; letter-spacing: 1.2px; text-transform: uppercase; }
.price-value { color: __GREEN__; font-size: 3rem; font-weight: 800; line-height: 1.15; }
.price-sub   { color: __INK__; opacity: .8; font-size: .92rem; }

/* ---------- chips ---------- */
.chip {
    display: inline-block; padding: 4px 12px; margin: 3px 4px 3px 0;
    border-radius: 999px; font-size: .78rem; font-weight: 600;
    border: 1.5px solid __GREEN__; color: __INK__; background: __IVORY__;
}
.chip-gold { background: __YELLOW__; border-color: __YELLOW__; color: __INK__; }
.chip-ok   { background: __GREEN__; color: __IVORY__; }
.chip-bad  { border-style: dashed; color: __INK__; background: __IVORY__; }

.feature-card {
    background: __IVORY__;
    border: 1.5px solid __GREEN__; border-top: 6px solid __YELLOW__;
    border-radius: 14px; padding: 18px 20px; height: 100%;
}
.feature-card h4 { margin: 0 0 6px 0; }
.feature-card p  { margin: 0; color: __INK__; opacity: .88; }
</style>
"""

CSS = (CSS.replace("__IVORY__", IVORY).replace("__GREEN__", GREEN).replace("__YELLOW__", YELLOW)
          .replace("__INK__", INK).replace("__MUTED__", MUTED))

st.markdown(CSS, unsafe_allow_html=True)


# ==================================================
# 5. UI HELPERS
# ==================================================

def hero(title, subtitle, icon=""):
    st.markdown(
        f'<div class="hero"><div class="hero-title">{icon} {title}</div>'
        f'<div class="hero-sub">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def chip(text, kind=""):
    return f'<span class="chip {kind}">{text}</span>'


def fmt_usd(x):
    return f"${x:,.0f}"


def style_fig(fig, height=380):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=PALETTE,
        font=dict(color=INK),
        height=height,
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        title_font=dict(color=GREEN, size=16),
    )
    fig.update_xaxes(gridcolor="rgba(0,155,77,.15)", zerolinecolor="rgba(0,155,77,.35)")
    fig.update_yaxes(gridcolor="rgba(0,155,77,.15)", zerolinecolor="rgba(0,155,77,.35)")
    return fig


def show_plot(fig, key):
    st.plotly_chart(fig, key=key, theme=None, **W)


def price_banner(price, label, sub):
    st.markdown(
        f'<div class="price-card"><div class="price-label">{label}</div>'
        f'<div class="price-value">{fmt_usd(price)}</div>'
        f'<div class="price-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def gauge(value, title):
    top = max(PRICE_CAP, value * 1.1)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"prefix": "$", "valueformat": ",.0f", "font": {"color": GREEN}},
            title={"text": title, "font": {"color": MUTED, "size": 14}},
            gauge={
                "axis": {"range": [0, top], "tickcolor": MUTED, "tickformat": "$,.0s"},
                "bar": {"color": GREEN},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, top * 0.30], "color": "rgba(255,204,0,0.20)"},
                    {"range": [top * 0.30, top * 0.60], "color": "rgba(255,204,0,0.45)"},
                    {"range": [top * 0.60, top], "color": "rgba(255,204,0,0.80)"},
                ],
            },
        )
    )
    return style_fig(fig, height=300)


def geo_figure(df, zoom=5.2, center=None, color=None, size=None, title=None, height=520):
    """Map scatter that works on both old and new Plotly versions (no token needed)."""
    new = hasattr(px, "scatter_map")
    fn = px.scatter_map if new else px.scatter_mapbox
    style = {"map_style": "carto-positron"} if new else {"mapbox_style": "carto-positron"}
    kwargs = dict(lat="latitude", lon="longitude", zoom=zoom, height=height, title=title, **style)
    if center:
        kwargs["center"] = center
    if color:
        kwargs["color"] = color
    if size:
        kwargs["size"] = size
    return fn(df, **kwargs)


# ==================================================
# 6. DATA, MODELS & METRICS
# ==================================================

def create_features(data):
    """Same feature engineering used when the models were trained in the notebook."""
    data = data.copy()
    data["rooms_per_household"] = data["total_rooms"] / data["households"].replace(0, np.nan)
    data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"].replace(0, np.nan)
    data["population_per_household"] = data["population"] / data["households"].replace(0, np.nan)
    return data


@st.cache_resource(show_spinner="Loading trained models…")
def load_models():
    if not DT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Decision Tree model file is missing: {DT_MODEL_PATH}")
    if not LR_MODEL_PATH.exists():
        raise FileNotFoundError(f"Linear Regression model file is missing: {LR_MODEL_PATH}")
    return joblib.load(DT_MODEL_PATH), joblib.load(LR_MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_data():
    for path in DATA_PATHS:
        if path.exists():
            return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_metrics():
    if not METRICS_PATH.exists():
        return pd.DataFrame()
    m = pd.read_csv(METRICS_PATH)
    m = m.rename(columns={"R2 Score": "R2", "R² Score": "R2", "R²": "R2"})
    if "Model" in m.columns:
        m["Model"] = m["Model"].map(
            lambda s: "Decision Tree" if "decision" in str(s).lower() else "Linear Regression"
        )
    return m


def metric_for(model_name, column):
    m = load_metrics()
    if m.empty or column not in m.columns:
        return None
    row = m[m["Model"] == model_name]
    return float(row[column].iloc[0]) if not row.empty else None


@st.cache_data(show_spinner="Scoring both models on the hold-out test set…")
def test_set_predictions(_dt, _lr, n_rows):
    """Re-creates the notebook's 80/20 split (random_state=42) and scores both models."""
    from sklearn.model_selection import train_test_split

    df = load_data()
    X = create_features(df.drop(columns=TARGET))
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    return pd.DataFrame(
        {
            "Actual": y_test.values,
            "Decision Tree": _dt.predict(X_test),
            "Linear Regression": _lr.predict(X_test),
        }
    )


def feature_table(pipe, kind):
    names = [re.sub(r"^(num|cat|remainder)__", "", n) for n in pipe[:-1].get_feature_names_out()]
    est = pipe.named_steps["model"]
    values = est.feature_importances_ if kind == "dt" else est.coef_
    out = pd.DataFrame({"Feature": names, "Value": values})
    out["Abs"] = out["Value"].abs()
    return out.sort_values("Abs", ascending=True)


def nearest_homes(df, lat, lon, k=10):
    dist = (df["latitude"] - lat) ** 2 + ((df["longitude"] - lon) * np.cos(np.radians(lat))) ** 2
    return df.loc[dist.nsmallest(k).index]


# ==================================================
# 7. SESSION STATE
# ==================================================

DEFAULTS = dict(
    longitude=-118.0, latitude=34.0, housing_median_age=25,
    total_rooms=1500, total_bedrooms=300, population=800, households=250,
    median_income=4.0, ocean_proximity="<1H OCEAN", selected_model="Decision Tree",
)

# Illustrative sample profiles (not real listings)
PRESETS = {
    "🌊 Coastal home": dict(longitude=-118.49, latitude=34.02, housing_median_age=30, total_rooms=2200,
                            total_bedrooms=420, population=1100, households=400,
                            median_income=7.5, ocean_proximity="NEAR OCEAN"),
    "🌉 Bay Area": dict(longitude=-122.25, latitude=37.85, housing_median_age=40, total_rooms=1400,
                        total_bedrooms=300, population=700, households=280,
                        median_income=5.5, ocean_proximity="NEAR BAY"),
    "🏙️ City family": dict(longitude=-118.30, latitude=34.05, housing_median_age=35, total_rooms=1800,
                           total_bedrooms=380, population=1500, households=420,
                           median_income=3.8, ocean_proximity="<1H OCEAN"),
    "🌵 Inland": dict(longitude=-119.80, latitude=36.75, housing_median_age=20, total_rooms=1800,
                      total_bedrooms=380, population=1300, households=360,
                      median_income=2.8, ocean_proximity="INLAND"),
}

for _key, _val in DEFAULTS.items():
    st.session_state.setdefault(_key, _val)
st.session_state.setdefault("prediction_history", [])
st.session_state.setdefault("last_result", None)


def apply_preset(name):
    st.session_state.update(PRESETS[name])
    st.session_state.last_result = None
    st.toast(f"Loaded sample profile: {name}", icon="✨")


def reset_form():
    st.session_state.update(DEFAULTS)
    st.session_state.last_result = None
    st.toast("Inputs reset to defaults", icon="↩️")


def clear_history():
    st.session_state.prediction_history = []
    st.session_state.last_result = None
    st.toast("Prediction history cleared", icon="🗑️")


# ==================================================
# 8. LOAD MODELS SAFELY
# ==================================================

models_loaded, load_error = True, ""
try:
    dt_model, lr_model = load_models()
except Exception as error:
    models_loaded, load_error = False, str(error)

df_data = load_data()


# ==================================================
# 9. SIDEBAR
# ==================================================

PAGES = [
    "🏠 Dashboard",
    "🔮 Predict House Price",
    "📊 Model Comparison",
    "🗺️ Data Explorer",
    "📜 Prediction History",
    "ℹ️ About Project",
]

st.sidebar.markdown(f"## 🏡 House Price AI")
st.sidebar.caption("Machine Learning Prediction Platform")
page = st.sidebar.radio("Navigation", PAGES, key="nav")
st.sidebar.divider()

st.sidebar.markdown(
    chip("Models ready", "chip-ok") if models_loaded else chip("Models not loaded", "chip-bad"),
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    chip(f"{len(st.session_state.prediction_history)} predictions", "chip-gold"),
    unsafe_allow_html=True,
)
st.sidebar.info("Predicts California house prices with trained Decision Tree and Linear Regression models.")

with st.sidebar.expander("⚙️ System info"):
    st.write(f"**scikit-learn:** {sklearn.__version__}")
    st.write(f"**Streamlit:** {st.__version__}")
    st.write(f"**pandas:** {pd.__version__}")
    st.caption("The saved models were trained with scikit-learn 1.6.1.")
    st.write("**Build:** green-ivory v3")

if not models_loaded:
    st.error(f"Unable to load models: {load_error}")
    if "_RemainderColsList" in load_error or "sklearn" in load_error.lower():
        st.info("This is a scikit-learn version mismatch. Run: `python -m pip install scikit-learn==1.6.1`")
    else:
        st.info("Check that both trained .pkl files are inside the `models` folder.")


# ==================================================
# 10. PAGE: DASHBOARD
# ==================================================

if page == PAGES[0]:

    hero("House Price Prediction Dashboard",
         "Predict California house prices with Machine Learning — compare models, explore data, test what-if scenarios.",
         "🏡")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Models Loaded", "2 / 2" if models_loaded else "0 / 2")
    c2.metric("Dataset Records", f"{len(df_data):,}" if df_data is not None else "20,640")
    c3.metric("Average Price", fmt_usd(df_data[TARGET].mean()) if df_data is not None else "—")
    c4.metric("Session Predictions", len(st.session_state.prediction_history))

    st.write("")

    if df_data is not None:
        a, b = st.columns(2)
        with a:
            avg = df_data.groupby("ocean_proximity")[TARGET].mean().sort_values().reset_index()
            fig = px.bar(avg, x=TARGET, y="ocean_proximity", orientation="h",
                         title="Average price by ocean proximity",
                         color=TARGET, color_continuous_scale=SCALE)
            fig.update_layout(coloraxis_showscale=False, xaxis_title="Average price (USD)", yaxis_title="")
            show_plot(style_fig(fig), "dash_ocean")
        with b:
            fig = px.histogram(df_data, x=TARGET, nbins=50, title="House price distribution")
            fig.update_traces(marker_color=GREEN, marker_line_color=IVORY, marker_line_width=0.5)
            fig.update_layout(xaxis_title="Median house value (USD)", yaxis_title="Count")
            show_plot(style_fig(fig), "dash_hist")
    else:
        st.info("Place `housing.csv` in the `data` folder to see dataset charts here.")

    st.subheader("Project Overview")
    a, b = st.columns(2)
    for col, name, icon, text in [
        (a, "Decision Tree", "🌳", "Learns a series of decision rules from the house features. Captures non-linear patterns."),
        (b, "Linear Regression", "📈", "Learns a weighted combination of the features. Simple, fast and easy to interpret."),
    ]:
        r2, rmse = metric_for(name, "R2"), metric_for(name, "RMSE")
        chips = ""
        if r2 is not None:
            chips += chip(f"R² {r2:.3f}", "chip-gold")
        if rmse is not None:
            chips += chip(f"RMSE {fmt_usd(rmse)}")
        col.markdown(
            f'<div class="feature-card"><h4>{icon} {name}</h4><p>{text}</p><div style="margin-top:10px">{chips}</div></div>',
            unsafe_allow_html=True,
        )

    st.subheader("How to Use the Application")
    s1, s2, s3 = st.columns(3)
    s1.markdown("**1 · Enter details**\n\nOpen **Predict House Price**, pick a sample profile or type your own values.")
    s2.markdown("**2 · Choose a model**\n\nDecision Tree, Linear Regression — or **Compare Both** side by side.")
    s3.markdown("**3 · Explore results**\n\nSee the estimate, a what-if analysis, nearby homes and download your history.")


# ==================================================
# 11. PAGE: PREDICT
# ==================================================

elif page == PAGES[1]:

    hero("Predict House Price",
         "Enter the property details to estimate its median house value (USD).", "🔮")
    st.caption("Predictions are statistical estimates, not property valuations.")

    # ---- quick-fill + reset (outside the form so callbacks can update the form widgets)
    st.markdown("**Quick-fill with a sample profile**")
    cols = st.columns(len(PRESETS) + 1)
    for col, name in zip(cols, PRESETS):
        col.button(name, on_click=apply_preset, args=(name,), key=f"preset_{name}", **W)
    cols[-1].button("↩️ Reset", on_click=reset_form, key="reset_btn", **W)

    with st.form("prediction_form"):
        st.subheader("📍 Location")
        c1, c2 = st.columns(2)
        longitude = c1.number_input("Longitude", min_value=-125.0, max_value=-113.0, step=0.01, key="longitude")
        latitude = c2.number_input("Latitude", min_value=32.0, max_value=42.0, step=0.01, key="latitude")

        st.subheader("🏘️ Property Details")
        c1, c2, c3 = st.columns(3)
        housing_median_age = c1.number_input("Housing Median Age", min_value=1, max_value=100, key="housing_median_age")
        total_rooms = c2.number_input("Total Rooms", min_value=1, key="total_rooms")
        total_bedrooms = c3.number_input("Total Bedrooms", min_value=1, key="total_bedrooms")
        c1, c2 = st.columns(2)
        population = c1.number_input("Population", min_value=1, key="population")
        households = c2.number_input("Households", min_value=1, key="households")

        st.subheader("💰 Income & Location Type")
        c1, c2 = st.columns(2)
        median_income = c1.number_input("Median Income (tens of thousands USD)", min_value=0.1,
                                        max_value=20.0, step=0.1, key="median_income")
        ocean_proximity = c2.selectbox("Ocean Proximity", OCEAN_OPTIONS, key="ocean_proximity")

        st.subheader("🤖 Machine Learning Model")
        selected_model = st.radio("Choose an algorithm", MODEL_CHOICES, horizontal=True, key="selected_model")
        submitted = st.form_submit_button("🚀 Predict House Price", **W)

    # ---- run prediction
    if submitted:
        if not models_loaded:
            st.error("Models could not be loaded.")
        elif total_bedrooms > total_rooms:
            st.error("Total bedrooms cannot exceed total rooms.")
        elif households > population:
            st.error("Households cannot exceed population.")
        else:
            raw = {
                "longitude": longitude, "latitude": latitude,
                "housing_median_age": housing_median_age, "total_rooms": total_rooms,
                "total_bedrooms": total_bedrooms, "population": population,
                "households": households, "median_income": median_income,
                "ocean_proximity": ocean_proximity,
            }
            input_df = create_features(pd.DataFrame([raw]))
            try:
                with st.spinner("Running model(s)…"):
                    preds = {}
                    if selected_model in ("Decision Tree", "Compare Both"):
                        preds["Decision Tree"] = max(0.0, float(dt_model.predict(input_df)[0]))
                    if selected_model in ("Linear Regression", "Compare Both"):
                        preds["Linear Regression"] = max(0.0, float(lr_model.predict(input_df)[0]))

                stamp = datetime.now().strftime("%H:%M:%S")
                for name, value in preds.items():
                    st.session_state.prediction_history.append({
                        "Time": stamp, "Model": name, "Predicted Price (USD)": round(value, 2),
                        "Median Income": median_income, "Total Rooms": total_rooms,
                        "Ocean Proximity": ocean_proximity, "Latitude": latitude, "Longitude": longitude,
                    })
                st.session_state.last_result = {"raw": raw, "preds": preds, "choice": selected_model}
                st.toast("Prediction complete!", icon="✅")
            except Exception as error:
                st.error(f"Prediction failed: {error}")

    # ---- show last result (persists while you browse its tabs)
    res = st.session_state.last_result
    if res and models_loaded:
        raw, preds = res["raw"], res["preds"]
        both = len(preds) == 2
        main = float(np.mean(list(preds.values())))

        st.divider()
        t1, t2, t3 = st.tabs(["💰 Estimate", "🔍 What-if analysis", "🗺️ Location & nearby homes"])

        # ----- tab 1: estimate
        with t1:
            if both:
                price_banner(main, "Average of both models", "Decision Tree + Linear Regression")
            else:
                name = next(iter(preds))
                price_banner(main, f"{name} estimate", "Estimated median house value (USD)")

            mcols = st.columns(len(preds) + (1 if both else 0))
            for col, (name, value) in zip(mcols, preds.items()):
                col.metric(name, fmt_usd(value))
                mae = metric_for(name, "MAE")
                if mae:
                    col.caption(f"Typical error ≈ ± {fmt_usd(mae)} (test-set MAE)")
            if both:
                gap = abs(list(preds.values())[0] - list(preds.values())[1])
                mcols[-1].metric("Model gap", fmt_usd(gap))
                mcols[-1].caption(f"{gap / main * 100:.1f}% of the average estimate")

            g1, g2 = st.columns([1.15, 1])
            with g1:
                show_plot(gauge(main, "Estimate vs. dataset price range"), "gauge")
            with g2:
                if df_data is not None:
                    pct = (df_data[TARGET] < main).mean() * 100
                    cat = raw["ocean_proximity"]
                    cat_avg = df_data.loc[df_data["ocean_proximity"] == cat, TARGET].mean()
                    st.metric("Price percentile", f"{pct:.0f}th",
                              help="Share of houses in the dataset with a lower median value.")
                    st.metric(f"Average for {cat}", fmt_usd(cat_avg),
                              delta=f"{(main / cat_avg - 1) * 100:+.1f}% estimate vs. avg", delta_color="off")
                else:
                    st.info("Add `housing.csv` to see how this estimate compares with the dataset.")

            if both:
                cmp_df = pd.DataFrame({"Model": list(preds), "Price": list(preds.values())})
                fig = px.bar(cmp_df, x="Model", y="Price", color="Model", text_auto="$,.0f",
                             title="Model estimates side by side")
                fig.update_layout(showlegend=False, yaxis_title="USD", xaxis_title="")
                show_plot(style_fig(fig, 340), "cmp_models")

        # ----- tab 2: what-if
        with t2:
            st.markdown("See how the estimate changes when **one** input varies and everything else stays the same.")
            sweep_vars = ["median_income", "housing_median_age", "total_rooms", "total_bedrooms",
                          "population", "households", "latitude", "longitude"]
            var = st.selectbox("Feature to vary", sweep_vars, key="whatif_var")
            if df_data is not None:
                lo, hi = float(df_data[var].quantile(0.01)), float(df_data[var].quantile(0.99))
            else:
                lo, hi = float(raw[var]) * 0.3, float(raw[var]) * 2.0
            lo, hi = st.slider("Range", min_value=float(min(lo, raw[var])), max_value=float(max(hi, raw[var])),
                               value=(float(min(lo, raw[var])), float(max(hi, raw[var]))), key=f"whatif_range_{var}")
            xs = np.linspace(lo, hi, 40)
            grid = pd.DataFrame([raw] * len(xs))
            grid[var] = xs
            grid = create_features(grid)
            curves = pd.DataFrame({
                var: xs,
                "Decision Tree": np.clip(dt_model.predict(grid), 0, None),
                "Linear Regression": np.clip(lr_model.predict(grid), 0, None),
            }).melt(id_vars=var, var_name="Model", value_name="Predicted price")
            fig = px.line(curves, x=var, y="Predicted price", color="Model", title=f"Estimated price vs. {var}")
            fig.add_vline(x=raw[var], line_dash="dash", line_color=INK,
                          annotation_text="your input", annotation_font_color=INK)
            fig.update_layout(yaxis_title="Predicted price (USD)")
            show_plot(style_fig(fig, 420), "whatif")
            st.caption("The Decision Tree gives step-like curves; Linear Regression gives straight lines.")

        # ----- tab 3: location + comps
        with t3:
            if df_data is not None:
                sample = df_data.sample(min(4000, len(df_data)), random_state=1)[["latitude", "longitude"]].copy()
                sample["Group"], sample["size"] = "Dataset homes", 3
                me = pd.DataFrame([{"latitude": raw["latitude"], "longitude": raw["longitude"],
                                    "Group": "Your property", "size": 14}])
                fig = geo_figure(pd.concat([sample, me]), zoom=5.4,
                                 center={"lat": raw["latitude"], "lon": raw["longitude"]},
                                 color="Group", size="size", title="Your property on the map", height=480)
                for trace in fig.data:
                    trace.marker.color = YELLOW if trace.name == "Your property" else GREEN
                    trace.marker.opacity = 1.0 if trace.name == "Your property" else 0.35
                show_plot(style_fig(fig, 480), "map_me")

                near = nearest_homes(df_data, raw["latitude"], raw["longitude"], k=10)
                n1, n2 = st.columns(2)
                n1.metric("Avg. of 10 nearest homes", fmt_usd(near[TARGET].mean()),
                          help="Actual median values of the 10 closest block groups in the dataset.")
                n2.metric("Your estimate vs. neighbours", f"{(main / near[TARGET].mean() - 1) * 100:+.1f}%")
                st.dataframe(near[["latitude", "longitude", "median_income", "housing_median_age", TARGET]],
                             hide_index=True, **W,
                             column_config={TARGET: st.column_config.NumberColumn("Actual value", format="$%d")})
            else:
                st.map(pd.DataFrame({"latitude": [raw["latitude"]], "longitude": [raw["longitude"]]}))

            st.markdown("**Engineered features used by the model**")
            eng = create_features(pd.DataFrame([raw]))[["rooms_per_household", "bedrooms_per_room", "population_per_household"]]
            st.dataframe(eng.round(3), hide_index=True, **W)


# ==================================================
# 12. PAGE: MODEL COMPARISON
# ==================================================

elif page == PAGES[2]:

    hero("Model Performance Comparison",
         "Decision Tree vs Linear Regression — test-set metrics, predictions and what drives them.", "📊")

    metrics_df = load_metrics()
    required = ["Model", "MAE", "MSE", "RMSE", "R2"]

    if metrics_df.empty:
        st.info("Metrics file not found. Export your comparison table as `model_metrics.csv` into the project folder.")
    elif not all(c in metrics_df.columns for c in required):
        st.error(f"model_metrics.csv must contain: {', '.join(required)}")
    else:
        best_rmse = metrics_df.loc[metrics_df["RMSE"].idxmin(), "Model"]
        best_r2 = metrics_df.loc[metrics_df["R2"].idxmax(), "Model"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Lowest RMSE", best_rmse, help="Lower is better")
        c2.metric("Highest R²", best_r2, help="Higher is better")
        c3.metric("Best R² score", f"{metrics_df['R2'].max():.3f}")

        tab_a, tab_b, tab_c = st.tabs(["📋 Metrics", "🎯 Actual vs. predicted", "🧠 Feature importance"])

        with tab_a:
            st.dataframe(
                metrics_df, hide_index=True, **W,
                column_config={
                    "MAE": st.column_config.NumberColumn(format="$%.0f"),
                    "RMSE": st.column_config.NumberColumn(format="$%.0f"),
                    "MSE": st.column_config.NumberColumn(format="%.3e"),
                    "R2": st.column_config.ProgressColumn("R²", min_value=0.0, max_value=1.0, format="%.3f"),
                },
            )
            a, b = st.columns(2)
            with a:
                melted = metrics_df.melt(id_vars="Model", value_vars=["MAE", "RMSE"],
                                         var_name="Metric", value_name="USD")
                fig = px.bar(melted, x="Metric", y="USD", color="Model", barmode="group",
                             title="Error metrics (lower is better)", text_auto="$,.0f")
                show_plot(style_fig(fig), "cmp_err")
            with b:
                fig = px.bar(metrics_df, x="Model", y="R2", color="Model", text_auto=".3f",
                             title="R² score (higher is better)")
                fig.update_layout(showlegend=False, yaxis_range=[0, 1], xaxis_title="")
                show_plot(style_fig(fig), "cmp_r2")
            st.caption("MAE/RMSE are in USD. MSE is in squared USD. R² is the share of price variation the model explains.")

        with tab_b:
            if not models_loaded or df_data is None:
                st.info("Needs both the trained models and `housing.csv`.")
            else:
                try:
                    scored = test_set_predictions(dt_model, lr_model, len(df_data))
                    which = st.radio("Model", ["Decision Tree", "Linear Regression"], horizontal=True, key="avp_model")
                    sub = scored.sample(min(2000, len(scored)), random_state=0)
                    a, b = st.columns(2)
                    with a:
                        fig = px.scatter(sub, x="Actual", y=which, opacity=0.5, title=f"{which}: actual vs. predicted")
                        fig.update_traces(marker=dict(color=GREEN, size=5))
                        top = float(max(sub["Actual"].max(), sub[which].max()))
                        fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="Perfect prediction",
                                                 line=dict(color=YELLOW, width=3, dash="dash")))
                        show_plot(style_fig(fig), "avp")
                    with b:
                        resid = pd.DataFrame({"Residual": scored[which] - scored["Actual"]})
                        fig = px.histogram(resid, x="Residual", nbins=60, title="Prediction errors (predicted − actual)")
                        fig.update_traces(marker_color=YELLOW, marker_line_color=GREEN, marker_line_width=0.6)
                        show_plot(style_fig(fig), "resid")
                except Exception as error:
                    st.warning(f"Could not score the test set: {error}")

        with tab_c:
            if not models_loaded:
                st.info("Models are not loaded.")
            else:
                try:
                    a, b = st.columns(2)
                    with a:
                        t = feature_table(dt_model, "dt")
                        fig = px.bar(t, x="Value", y="Feature", orientation="h",
                                     title="Decision Tree · feature importance")
                        fig.update_traces(marker_color=GREEN)
                        show_plot(style_fig(fig, 460), "fi_dt")
                    with b:
                        t = feature_table(lr_model, "lr")
                        t["Direction"] = np.where(t["Value"] >= 0, "Raises price", "Lowers price")
                        fig = px.bar(t, x="Value", y="Feature", orientation="h", color="Direction",
                                     color_discrete_map={"Raises price": GREEN, "Lowers price": YELLOW},
                                     title="Linear Regression · coefficients")
                        show_plot(style_fig(fig, 460), "fi_lr")
                    st.caption("Linear coefficients apply to scaled numeric features, so numeric bars are comparable "
                               "with each other. One-hot (ocean proximity) bars are on a different scale.")
                except Exception as error:
                    st.warning(f"Could not read feature names from the pipelines: {error}")


# ==================================================
# 13. PAGE: DATA EXPLORER
# ==================================================

elif page == PAGES[3]:

    hero("Data Explorer", "Filter and explore the California housing dataset used to train the models.", "🗺️")

    if df_data is None:
        st.info("Place `housing.csv` in the `data` folder (or the project root) to use the explorer.")
    else:
        with st.expander("🎚️ Filters", expanded=True):
            f1, f2, f3 = st.columns(3)
            price_rng = f1.slider("House value (USD)", int(df_data[TARGET].min()), int(df_data[TARGET].max()),
                                  (int(df_data[TARGET].min()), int(df_data[TARGET].max())), step=5000, key="f_price")
            inc_rng = f2.slider("Median income", float(df_data["median_income"].min()),
                                float(df_data["median_income"].max()),
                                (float(df_data["median_income"].min()), float(df_data["median_income"].max())),
                                key="f_income")
            cats = f3.multiselect("Ocean proximity", sorted(df_data["ocean_proximity"].unique()),
                                  default=sorted(df_data["ocean_proximity"].unique()), key="f_cats")

        mask = (
            df_data[TARGET].between(*price_rng)
            & df_data["median_income"].between(*inc_rng)
            & df_data["ocean_proximity"].isin(cats)
        )
        view = df_data[mask]

        m1, m2, m3 = st.columns(3)
        m1.metric("Matching records", f"{len(view):,}", help=f"of {len(df_data):,}")
        m2.metric("Average price", fmt_usd(view[TARGET].mean()) if len(view) else "—")
        m3.metric("Average income", f"{view['median_income'].mean():.2f}" if len(view) else "—")

        if view.empty:
            st.warning("No records match these filters.")
        else:
            t1, t2, t3, t4 = st.tabs(["🗺️ Map", "📈 Distributions", "🔗 Correlations", "📄 Data"])

            with t1:
                sample = view.sample(min(6000, len(view)), random_state=2)
                fig = geo_figure(sample, zoom=4.8, center={"lat": 37.2, "lon": -119.5},
                                 color=TARGET, title="Houses coloured by median value", height=560)
                fig.update_layout(coloraxis=dict(colorscale=SCALE))
                fig.update_traces(marker=dict(size=5, opacity=0.7))
                show_plot(style_fig(fig, 560), "ex_map")
                st.caption("Showing a random sample of up to 6,000 points for speed.")

            with t2:
                num_cols = [c for c in df_data.select_dtypes(include=np.number).columns]
                a, b = st.columns(2)
                col = a.selectbox("Histogram column", num_cols, index=num_cols.index(TARGET), key="ex_hist_col")
                fig = px.histogram(view, x=col, nbins=50, color="ocean_proximity", title=f"Distribution of {col}")
                show_plot(style_fig(fig), "ex_hist")
                xcol = b.selectbox("Scatter X axis", num_cols, index=num_cols.index("median_income"), key="ex_x")
                fig = px.scatter(view.sample(min(4000, len(view)), random_state=3), x=xcol, y=TARGET,
                                 color="ocean_proximity", opacity=0.6, title=f"{xcol} vs. house value")
                show_plot(style_fig(fig), "ex_scatter")

            with t3:
                corr = view.select_dtypes(include=np.number).corr()
                fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale=SCALE,
                                title="Correlation matrix")
                show_plot(style_fig(fig, 560), "ex_corr")

            with t4:
                st.dataframe(
                    view, hide_index=True, **W,
                    column_config={TARGET: st.column_config.NumberColumn("median_house_value", format="$%d")},
                )
                st.download_button("⬇️ Download filtered data (CSV)", view.to_csv(index=False).encode("utf-8"),
                                   file_name="housing_filtered.csv", mime="text/csv", key="dl_filtered", **W)


# ==================================================
# 14. PAGE: PREDICTION HISTORY
# ==================================================

elif page == PAGES[4]:

    hero("Prediction History", "Every prediction made during this session.", "📜")

    history = st.session_state.prediction_history

    if not history:
        st.info("No predictions yet. Go to **Predict House Price** to make your first one.")
    else:
        hist_df = pd.DataFrame(history)
        hist_df["#"] = range(1, len(hist_df) + 1)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predictions", len(hist_df))
        c2.metric("Average", fmt_usd(hist_df["Predicted Price (USD)"].mean()))
        c3.metric("Highest", fmt_usd(hist_df["Predicted Price (USD)"].max()))
        c4.metric("Lowest", fmt_usd(hist_df["Predicted Price (USD)"].min()))

        fig = px.line(hist_df, x="#", y="Predicted Price (USD)", color="Model", markers=True,
                      title="Predictions over the session")
        fig.update_layout(xaxis_title="Prediction number")
        show_plot(style_fig(fig, 360), "hist_line")

        st.dataframe(
            hist_df.drop(columns="#"), hide_index=True, **W,
            column_config={"Predicted Price (USD)": st.column_config.NumberColumn(format="$%.0f")},
        )

        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download history (CSV)", hist_df.drop(columns="#").to_csv(index=False).encode("utf-8"),
                           file_name="prediction_history.csv", mime="text/csv", key="dl_hist", **W)
        d2.button("🗑️ Clear history", on_click=clear_history, key="clear_hist", **W)


# ==================================================
# 15. PAGE: ABOUT
# ==================================================

elif page == PAGES[5]:

    hero("About the Project", "House price prediction using supervised machine-learning regression.", "ℹ️")

    a, b = st.columns([1.2, 1])
    with a:
        st.markdown("""
### Workflow
1. Data collection  
2. Exploratory data analysis  
3. Data cleaning & preprocessing  
4. Feature engineering (rooms / bedrooms / people per household)  
5. Feature selection  
6. Model training & hyper-parameter tuning (GridSearchCV)  
7. Model evaluation  
8. Deployment with Streamlit  

### Dataset
California Housing dataset — 20,640 block groups with location, housing and income features.
The target is `median_house_value` (capped at $500,001).
""")
    with b:
        st.markdown("### Technologies")
        st.markdown(
            "".join(chip(t) for t in ["Python", "Pandas", "NumPy", "Scikit-learn", "Plotly", "Streamlit", "Joblib"])
            + "".join(chip(t, "chip-gold") for t in ["Decision Tree", "Linear Regression", "Pipelines", "GridSearchCV"]),
            unsafe_allow_html=True,
        )
        st.markdown("### App features")
        st.markdown(
            "- Predict with one model or **compare both**\n"
            "- Sample profiles and one-click reset\n"
            "- What-if analysis and nearby-home comparison\n"
            "- Feature importance and actual-vs-predicted plots\n"
            "- Interactive data explorer with map\n"
            "- Downloadable prediction history"
        )


# ==================================================
# 16. FOOTER
# ==================================================

st.divider()
st.caption("House Price AI · Built with Python, Scikit-learn and Streamlit")