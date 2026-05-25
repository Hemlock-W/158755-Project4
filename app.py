import streamlit as st
import pandas as pd
import numpy as np
import requests
import glob
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import scipy.stats as stats
from io import StringIO
 
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
 
# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NZ Energy Demand Predictor",
    layout="wide",
)
 
# ── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0e1117; color: #969696; }
 
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #161b22; }
 
    /* Cards / metric containers */
    div[data-testid="metric-container"] {
        background-color: #1c2230;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px 18px;
    }
 
    /* Section headers */
    h1, h2, h3 { color: #58a6ff; }
 
    /* Dividers */
    hr { border-color: #30363d; }
 
    /* Select box */
    div[data-baseweb="select"] > div {
        background-color: #1c2230;
        border-color: #30363d;
        color: #FFFFFF;
    }
 
    /* DataFrames */
    .stDataFrame { background-color: #161b22; }
 
    /* Buttons */
    .stButton > button {
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
    }
    .stButton > button:hover { background-color: #2ea043; }
    
    /* Markdown https://nz.pinterest.com/pin/color-palette-for-black-backgrounds--289497082311303940/ */
    .stMarkdown {color: #969696;}
    
    /* Info / warning boxes */
    .stAlert { background-color: #1c2230; border-color: #30363d; }
</style>
""", unsafe_allow_html=True)
 
# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching grid data…")
def get_grid_data():
    # Load all CSV files
    files = glob.glob("archive/*Grid_import*.csv")

    df_list = []
    for file in files:
        temp = pd.read_csv(file)
        df_list.append(temp)

    # Combine all files
    df_elec = pd.concat(df_list, ignore_index=True)
    return df_elec


@st.cache_data(show_spinner="Fetching weather data…")
def get_weather_data(url, city):
    data = requests.get(url, timeout=30).json()
    df = pd.DataFrame(data["daily"])
    df["date"] = pd.to_datetime(df["time"])
    df = df.rename(columns={
        "temperature_2m_max": f"temp_max_{city}",
        "temperature_2m_min": f"temp_min_{city}",
        "precipitation_sum": f"rain_{city}",
    })
    return df[["date", f"temp_max_{city}", f"temp_min_{city}", f"rain_{city}"]]
 
 
@st.cache_data(show_spinner="Fetching holiday data…")
def get_holidays(year):
    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/NZ"
    data = requests.get(url, timeout=10).json()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "name"]]
 
 
@st.cache_data(show_spinner="Building dataset…")
def build_dataset():
    # Weather for 3 cities (2023-01-01 → 2024-12-31)
    base = "https://archive-api.open-meteo.com/v1/archive"
    urls = {
        "akl": f"{base}?latitude=-36.8485&longitude=174.7633&start_date=2023-01-01&end_date=2024-12-31&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Pacific/Auckland",
        "wlg": f"{base}?latitude=-41.2865&longitude=174.7762&start_date=2023-01-01&end_date=2024-12-31&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Pacific/Auckland",
        "chc": f"{base}?latitude=-43.5321&longitude=172.6362&start_date=2023-01-01&end_date=2024-12-31&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Pacific/Auckland",
    }
    dfs = [get_weather_data(u, c) for c, u in urls.items()]
    df_weather = dfs[0].merge(dfs[1], on="date").merge(dfs[2], on="date")
 
    # Average weather across cities
    df_weather["temp_max_avg"] = df_weather[["temp_max_akl","temp_max_wlg","temp_max_chc"]].mean(axis=1)
    df_weather["temp_min_avg"] = df_weather[["temp_min_akl","temp_min_wlg","temp_min_chc"]].mean(axis=1)
    df_weather["rain_avg"]     = df_weather[["rain_akl","rain_wlg","rain_chc"]].mean(axis=1)
 
    # Cleaning of dataset
    df_elec = get_grid_data()
    df_elec.columns = df_elec.columns.str.strip()
    df_elec['Trading_Date'] = pd.to_datetime(df_elec['Trading_Date'])
    df_elec = df_elec.rename(columns={
            "Trading_Date": "date"
        })
    # Set trading date as index in descending order
    df_elec.drop(['TP49', 'TP50'], axis=1, inplace=True)
    # NAN value will be the value of previous date
    df_elec = df_elec.bfill()
    tp_cols = [col for col in df_elec.columns if col.startswith("TP")]
    df_elec["daily_total"] = df_elec[tp_cols].sum(axis=1, skipna=True)
    df_daily = df_elec.groupby("date")["daily_total"].sum().reset_index()
    df_daily.rename(columns={"daily_total": "demand"}, inplace=True)
 
    # Holidays
    df_holidays = pd.concat([get_holidays(y) for y in [2023, 2024]], ignore_index=True)
    df_final = pd.merge(df_daily, df_weather, on="date", how="inner")
    df_final["is_holiday"] = df_final["date"].dt.date.isin( df_holidays["date"].dt.date ).astype(int)
    df_final["month"]      = df_final["date"].dt.month
    df_final["dayofweek"]  = df_final["date"].dt.dayofweek
 
    df_final.set_index("date", inplace=True)
    df_final.sort_index(ascending=True, inplace=True)
    return df_final


def make_lag_features(series, lags=[1, 2, 3, 7]):
    df_lags = pd.DataFrame(index=series.index)
    for lag in lags:
        df_lags[f"lag_{lag}"] = series.shift(lag)
    return df_lags

 
def train_model(df, model_name):
    features = ["temp_max_avg", "temp_min_avg", "rain_avg", "is_holiday", "month", "dayofweek"]
    X = df[features]
    y = df["demand"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X = X.sort_index()
    y = y.sort_index()
 
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    if model_name == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
 
    elif model_name == "Time Lagged Prediction":
        split = int(len(X) * 0.8) # No future data leaks
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        train_model_predict = pd.Series(model.predict(X_train), index=y_train.index)
        residuals = y_train - train_model_predict
        
        lag_df = make_lag_features(residuals, lags=[7])
        lag_df["residual"] = residuals
        lag_df.dropna(inplace=True) # Drop NA caused by shifting
        X_lag = lag_df.drop(columns=["residual"])
        y_lag = lag_df["residual"]

        lag_model = LinearRegression()
        test_model_predict = pd.Series(model.predict(X_test), index=y_test.index)
        combined_residuals = pd.concat([residuals, y_test-test_model_predict])
        lag_df = make_lag_features(combined_residuals, lags=[7])
        X_lag_test = lag_df.loc[test_model_predict.index]
        X_lag_test.dropna(inplace=True)
        lag_model.fit(X_lag, y_lag)
        residual_pred = pd.Series(lag_model.predict(X_lag_test), index=X_lag_test.index)
        baseline_aligned = test_model_predict.loc[residual_pred.index]
        y_pred = baseline_aligned + residual_pred
        y_test = y_test.loc[residual_pred.index]
 
    elif model_name == "Polynomial Regression (deg=2)":
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_p = poly.fit_transform(X_train)
        X_test_p  = poly.transform(X_test)
        model = LinearRegression()
        model.fit(X_train_p, y_train)
        y_pred = model.predict(X_test_p)
 
    elif model_name == "KNN (k=10, unscaled)":
        model = KNeighborsRegressor(n_neighbors=10)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
 
    elif model_name == "KNN (k=10, scaled)":
        model = KNeighborsRegressor(n_neighbors=10)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
 
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    return y_test, y_pred, rmse, r2
 
 
# ── App layout ────────────────────────────────────────────────────────────────
st.title("NZ Energy Demand Predictor")
st.caption("Weather-driven electricity demand prediction for New Zealand - Auckland, Wellington, Christchurch")
st.divider()
 
# Load data
with st.spinner("Loading dataset…"):
    try:
        df = build_dataset()
        data_ok = True
    except Exception as e:
        st.error(f"Could not fetch live data: {e}")
        data_ok = False
 
if data_ok:
    # - PART 1: Model Selection -
    st.header("Model Selection")
 
    col_sel, col_desc = st.columns([1, 2])
 
    MODEL_INFO = {
        "Linear Regression": {
            "desc": "Baseline model using temperature & rainfall only.",
            "features": "temp_max_avg, temp_min_avg, rain_avg, month, dayofweek",
        },
        "Time Lagged Prediction": {
            "desc": "Prediction of residual using Linear Regression.",
            "features": "temp_max_avg, temp_min_avg, rain_avg, is_holiday, month",
        },
        "Polynomial Regression (deg=2)": {
            "desc": "Captures non-linear interactions between weather variables (degree 2).",
            "features": "Polynomial expansion of weather + time features",
        },
        "KNN (k=10, unscaled)": {
            "desc": "K-Nearest Neighbours regression without feature scaling.",
            "features": "temp_max_avg, temp_min_avg, rain_avg, is_holiday, month, dayofweek",
        },
        "KNN (k=10, scaled)": {
            "desc": "KNN with StandardScaler — usually outperforms the unscaled version.",
            "features": "Scaled: temp_max_avg, temp_min_avg, rain_avg, is_holiday, month, dayofweek",
        },
    }
 
    with col_sel:
        chosen = st.selectbox(
            "Choose a prediction model",
            list(MODEL_INFO.keys()),
            index=0,
        )
 
    with col_desc:
        st.markdown(f"**Description:** {MODEL_INFO[chosen]['desc']}")
        st.markdown(f"**Features:** `{MODEL_INFO[chosen]['features']}`")
 
    st.divider()
 
    # Run model
    y_test, y_pred, rmse, r2 = train_model(df, chosen)
 
    # Metrics row
    m1, m2, m3 = st.columns(3)
    m1.metric("Model", chosen)
    m2.metric("RMSE", f"{rmse:,.0f} MWh")
    m3.metric("R² Score", f"{r2:.4f}")
 
    st.divider()
 
    # - PART 2: Prediction Graph -
    st.header("Prediction Output")
 
    tab1, tab2 = st.tabs(["Actual vs Predicted", "Demand Over Time"])
 
    with tab1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=y_test.values, y=y_pred,
            mode="markers",
            marker=dict(color="#58a6ff", opacity=0.65, size=6),
            name="Predictions",
        ))
        # Perfect prediction line
        lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        fig1.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi],
            mode="lines",
            line=dict(color="#f78166", dash="dash", width=2),
            name="Perfect Prediction",
        ))
        fig1.update_layout(
            title=f"Actual vs Predicted Demand — {chosen}",
            xaxis_title="Actual Demand (MWh)",
            yaxis_title="Predicted Demand (MWh)",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="#e0e0e0",
            legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        )
        st.plotly_chart(fig1, use_container_width=True)
 
    with tab2:
        # Show full time-series with rolling average
        df_plot = df.copy().sort_index()
        df_plot["rolling_7d"] = df_plot["demand"].rolling(7).mean()
 
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["demand"],
            mode="lines",
            line=dict(color="#58a6ff", width=1),
            opacity=0.4,
            name="Daily Demand",
        ))
        fig2.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["rolling_7d"],
            mode="lines",
            line=dict(color="#3fb950", width=2),
            name="7-Day Average",
        ))
        fig2.update_layout(
            title="Daily Electricity Demand with 7-Day Rolling Average",
            xaxis_title="Date",
            yaxis_title="Demand (MWh)",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="#e0e0e0",
            legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        )
        st.plotly_chart(fig2, use_container_width=True)
 
    st.divider()
 
    # - PART 3: EDA -
    st.header("Exploratory Data Analysis")
 
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Electricity Demand Over Time", 
                            "Scatter Plot Matrix", "Correlation Matrix", 
                            "Distribution of Electricity Demand",
                            "Temperature vs Electricity Demand",
                            "Holiday vs Non-Holiday"])
 
    with tab1:
        # Show full time-series with rolling average
        df_plot = df.copy().sort_index()
        df_plot["rolling_7d"] = df_plot["demand"].rolling(7).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["demand"],
            mode="lines",
            line=dict(color="#58a6ff", width=1),
            opacity=0.4,
            name="Daily Demand",
        ))
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["rolling_7d"],
            mode="lines",
            line=dict(color="#3fb950", width=2),
            name="7-Day Average",
        ))
        fig.update_layout(
            title="Daily Electricity Demand with 7-Day Rolling Average",
            xaxis_title="Date",
            yaxis_title="Demand (kWh)",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="#e0e0e0",
            legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        )
        st.plotly_chart(fig, use_container_width=True)
 
    with tab2:
        temp_precip = df[[
            "temp_max_avg", "temp_min_avg", "rain_avg", "demand"
        ]]
        fig = px.scatter_matrix(
            df,
            dimensions=temp_precip,
            title="Pair Plot"
        )
        fig.update_layout( 
            title="Scatter Plot Matrix", 
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="#e0e0e0",
            legend=dict(bgcolor="#161b22", bordercolor="#30363d")
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        corr_df = df[[
            "demand",
            "temp_max_avg", "temp_min_avg", "rain_avg"
        ]]
        corr_m = corr_df.corr()
        fig = ff.create_annotated_heatmap(
            z=corr_m.values,
            x=list(corr_df),
            y=list(corr_df),
            annotation_text=np.around(corr_m.values, decimals=4),
            colorscale='Spectral'
        )
        fig.update_layout( 
            title="Correlation Matrix", 
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="#e0e0e0",
            legend=dict(bgcolor="#161b22", bordercolor="#30363d")
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=df["demand"],
                histnorm='probability density', 
                name='Histogram',
                marker_color='purple',
                opacity=0.6
            )
        )
        x_range = np.linspace(min(df["demand"]), max(df["demand"]), 200)
        kde = stats.gaussian_kde(df["demand"])
        y_kde = kde(x_range)

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_kde,
                mode='lines',
                name='KDE',
                line=dict(color='crimson', width=2)
            )
        )
        fig.update_layout(
            title="Distribution of Electricity Demand",
            xaxis_title="Electricity Demand",
            yaxis_title="Frequency",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="#e0e0e0",
            legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
            barmode='overlay',
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        fig = px.scatter(
            df, 
            x='temp_min_avg', 
            y='demand', 
            trendline='ols', # Adds the regression line
            title="Temperature vs Electricity Demand with Trend Line"
        )
        fig.update_layout(  
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="#e0e0e0",
            legend=dict(bgcolor="#161b22", bordercolor="#30363d")
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab6:
        holiday_avg = df.groupby("is_holiday")["demand"].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=["Non-Holiday", "Holiday"],
                y=holiday_avg['demand']
            )
        )
        fig.update_layout(
            title="Holiday vs Non-Holiday",
            xaxis_title="Electricity Demand",
            yaxis_title="Electricity Demand (kWh)",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="#e0e0e0",
            legend=dict(bgcolor="#161b22", bordercolor="#30363d")
        )
        st.plotly_chart(fig, use_container_width=True)
    st.divider()


    # - PART 4: Dataset -
    st.header("Dataset")
 
    display_cols = [
        "demand", "temp_max_avg", "temp_min_avg", "rain_avg",
        "is_holiday", "month", "dayofweek",
        "temp_max_akl", "temp_min_akl", "rain_akl",
        "temp_max_wlg", "temp_min_wlg", "rain_wlg",
        "temp_max_chc", "temp_min_chc", "rain_chc",
    ]
    df_show = df[[c for c in display_cols if c in df.columns]].copy()
    df_show.index = df_show.index.date   # clean date display
 
    col_filter, col_rows = st.columns([2, 1])
    with col_filter:
        search = st.text_input("Filter columns (comma-separated, leave blank to show all)")
    with col_rows:
        n_rows = st.slider("Rows to display", 10, len(df_show), 50)
 
    if search.strip():
        wanted = [c.strip() for c in search.split(",")]
        df_show = df_show[[c for c in wanted if c in df_show.columns]]
 
    st.dataframe(
        df_show.head(n_rows).style.format(precision=2),
        use_container_width=True,
        height=420,
    )
 
    st.caption(f"Showing {n_rows} of {len(df_show)} rows · {df_show.shape[1]} columns")
 
    # Summary statistics toggle
    with st.expander("Summary Statistics"):
        st.dataframe(df_show.describe().T.style.format(precision=2), use_container_width=True)
    