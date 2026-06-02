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
from functools import reduce
 
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor


# - Page config -
st.set_page_config(
    page_title="NZ Energy Demand Predictor",
    layout="wide",
)
 
# - Dark theme CSS -
st.markdown("""
<style>
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
</style>
""", unsafe_allow_html=True)
 
# - Helpers -
@st.cache_data(show_spinner="Fetching pricing data…")
def get_pricing_data():
       pricing = pd.read_csv("archive/Wholesale_price_trends_20260524152740.csv", skiprows=11)
       pricing['date'] = pd.to_datetime(pricing['Period start'], format='%d/%m/%Y')
       pricing = pricing[['date', 'Price ($/MWh)']].rename(columns={'Price ($/MWh)': 'price'})
       pricing['price'] = pricing['price'] / 1000  # $/MWh to $/kWh
       pricing = pricing[pricing['date'] >= '2023-01-01']
       return pricing

@st.cache_data(show_spinner="Fetching ECT data…")
def get_ect_data():
    ect = pd.read_csv("archive/electronic-card-transactions-january-2026-csv-tables.csv")
    # Filter to actual (not seasonally adjusted / trend) total-industry spend.
    # Series_title_3/4/5 may still split by card type, so groupby().sum()
    # collapses everything to one genuine total per month.
    ect = ect[
        (ect["Series_title_2"] == "RTS total industries") &
        (ect["Series_title_1"] == "Actual")
    ]
    ect["Period"] = ect["Period"].astype(str)
    ect["Year"]   = ect["Period"].str.split(".").str[0].astype(int)
    ect["Month"]  = ect["Period"].str.split(".").str[1].astype(int)
    ect = ect[ect["Year"] >= 2023]                                # ← drop pre-2023
    ect["month"]  = pd.to_datetime(
        dict(year=ect["Year"], month=ect["Month"], day=1)
    ).dt.to_period("M")
    ect = ect[["month", "Data_value"]].rename(columns={"Data_value": "ECT"})
    return ect.groupby("month", as_index=False)["ECT"].sum()  # 1 row per month



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
    # Weather for cities (2023-01-01 → 2025-12-31)
    base = "https://archive-api.open-meteo.com/v1/archive"
    urls = {
        "akl": f"{base}?latitude=-36.8485&longitude=174.7633&start_date=2023-01-01&end_date=2025-12-31&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Pacific/Auckland",
        "wlg": f"{base}?latitude=-41.2865&longitude=174.7762&start_date=2023-01-01&end_date=2025-12-31&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Pacific/Auckland",
        "chc": f"{base}?latitude=-43.5321&longitude=172.6362&start_date=2023-01-01&end_date=2025-12-31&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Pacific/Auckland",
        "ham": f"{base}?latitude=-37.7870&longitude=175.2793&start_date=2023-01-01&end_date=2025-12-31&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Pacific/Auckland",
        "tau": f"{base}?latitude=-37.6878&longitude=176.1651&start_date=2023-01-01&end_date=2025-12-31&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Pacific/Auckland",
        "dun": f"{base}?latitude=-45.8788&longitude=170.5028&start_date=2023-01-01&end_date=2025-12-31&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Pacific/Auckland",
        "qtn": f"{base}?latitude=-45.0312&longitude=168.6626&start_date=2023-01-01&end_date=2025-12-31&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Pacific/Auckland",
        "rot": f"{base}?latitude=-38.1368&longitude=176.2497&start_date=2023-01-01&end_date=2025-12-31&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Pacific/Auckland",
    }


    dfs = [get_weather_data(u, c) for c, u in urls.items()]
    df_weather = reduce(lambda left, right: pd.merge(left, right, on="date"), dfs)
 
    # Average weather across cities
    city_codes = list(urls.keys())

    df_weather["temp_max_avg"] = df_weather[[f"temp_max_{c}" for c in city_codes]].mean(axis=1)
    df_weather["temp_min_avg"] = df_weather[[f"temp_min_{c}" for c in city_codes]].mean(axis=1)
    df_weather["rain_avg"] = df_weather[[f"rain_{c}" for c in city_codes]].mean(axis=1)
 
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
    df_elec = df_elec.ffill()
    tp_cols = [col for col in df_elec.columns if col.startswith("TP")]
    df_elec["daily_total"] = df_elec[tp_cols].sum(axis=1, skipna=True)
    df_daily = df_elec.groupby("date")["daily_total"].sum().reset_index()
    df_daily.rename(columns={"daily_total": "demand"}, inplace=True)
 
    # Holidays
    df_holidays = pd.concat([get_holidays(y) for y in [2023, 2024, 2025]], ignore_index=True)
    df_final = pd.merge(df_daily, df_weather, on="date", how="inner")
    df_final["is_holiday"] = df_final["date"].dt.date.isin( df_holidays["date"].dt.date ).astype(int)
    df_final["month"]      = df_final["date"].dt.month
    df_final["dayofweek"]  = df_final["date"].dt.dayofweek
 
    # ECT: one monthly value repeated for every day in that month
    df_ect = get_ect_data()
    df_final["month_period"] = pd.to_datetime(df_final["date"]).dt.to_period("M")
    df_ect = df_ect.rename(columns={"month": "month_period"})
    df_final = df_final.merge(df_ect, on="month_period", how="left")
    df_final.drop(columns=["month_period"], inplace=True)
    
    df_final["ECT"] = df_final["ECT"].ffill().bfill()

    # ── PRICING: daily price for each day (BEFORE setting index!) ──────────────
    df_pricing = get_pricing_data()
    df_final = df_final.merge(df_pricing, on="date", how="left")
    df_final["price"] = df_final["price"].ffill().bfill()


    df_final.set_index("date", inplace=True)
    df_final.sort_index(ascending=True, inplace=True)
    return df_final


def make_lag_features(series, lags=[1, 2, 3, 7]):
    df_lags = pd.DataFrame(index=series.index)
    for lag in lags:
        df_lags[f"lag_{lag}"] = series.shift(lag)
    
    # Rolling statistics
    df_lags["rolling_mean_7"]  = series.shift(1).rolling(7).mean()
    df_lags["rolling_std_7"]   = series.shift(1).rolling(7).std() 
    df_lags["rolling_min_7"]   = series.shift(1).rolling(7).min()
    df_lags["rolling_max_7"]   = series.shift(1).rolling(7).max()
    
    # Lag difference
    df_lags["lag_diff_1"] = series.shift(1) - series.shift(2)
    df_lags["lag_diff_7"] = series.shift(1) - series.shift(8)

    return df_lags

 
def train_model(df, model_name):
    features = ["temp_max_avg", "temp_min_avg", "rain_avg", "is_holiday", "month", "dayofweek", "ECT", "price"]
    X = df[features]
    y = df["demand"]

    split = int(len(X) * 0.8) # No future data leaks
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    model = LinearRegression()

    if model_name == "Linear Regression":
        lag_model = LinearRegression()
    elif model_name == "SVR":
        lag_model = Pipeline([
            ("scalar", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=100, epsilon=0.01))
        ])
    elif model_name == "KNN (k=10, scaled)":
        model = lag_model = Pipeline([
            ("scalar", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=10))
        ])
    elif model_name == "Random Forest Regressor":
        model = lag_model = Pipeline([
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
        ])
    elif model_name == "Neural Network Regressor":
        lag_model = Pipeline([
            ("scalar", StandardScaler()),
            ("mlp", MLPRegressor(hidden_layer_sizes=(150, 50, 10), activation='relu', 
                                solver='adam', max_iter=300))
        ])
    elif model_name == "Histogram Gradient Boosting Regressor":
        model = lag_model = Pipeline([
            ("scalar", StandardScaler()),
            ("hgbr", HistGradientBoostingRegressor(learning_rate=0.01, max_iter=200, max_depth=50))
        ])
    
    
    model.fit(X_train, y_train)
    train_model_predict = pd.Series(model.predict(X_train), index=y_train.index)
    residuals = y_train - train_model_predict
    
    lag_df = make_lag_features(residuals, lags=[1, 2, 7])
    lag_df["residual"] = residuals
    lag_df.dropna(inplace=True) # Drop NA caused by shifting
    X_lag = lag_df.drop(columns=["residual"])
    y_lag = lag_df["residual"]

 
    test_model_predict = pd.Series(model.predict(X_test), index=y_test.index)
    combined_residuals = pd.concat([residuals, y_test-test_model_predict])
    lag_df = make_lag_features(combined_residuals, lags=[1, 2, 7])
    X_lag_test = lag_df.loc[test_model_predict.index]
    X_lag_test.dropna(inplace=True)
    lag_model.fit(X_lag, y_lag)
    residual_pred = pd.Series(lag_model.predict(X_lag_test), index=X_lag_test.index)
    baseline_aligned = test_model_predict.loc[residual_pred.index]
    y_pred = baseline_aligned + residual_pred
    y_test = y_test.loc[residual_pred.index]

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return y_test, y_pred, rmse, r2, mae, mape
 
 
# - App layout -
st.title("NZ Energy Demand Predictor")
st.caption("Weather-driven electricity demand prediction for New Zealand")
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
            "desc": "Prediction of residual using Linear Regression.",
            "features": "temp_max_avg, temp_min_avg, rain_avg, is_holiday, month, dayofweek, ECT, price",
        },
        "SVR": {
            "desc": "Prediction of residual using SVR.",
            "features": "temp_max_avg, temp_min_avg, rain_avg, is_holiday, month, dayofweek, ECT, price",
        },
        "KNN (k=10, scaled)": {
            "desc": "Prediction of residual using K-Nearest Neighbours Regression.",
            "features": "temp_max_avg, temp_min_avg, rain_avg, is_holiday, month, dayofweek, ECT, price",
        },
        "Random Forest Regressor": {
            "desc": "Prediction of residual using Random Forest Regression.",
            "features": "temp_max_avg, temp_min_avg, rain_avg, is_holiday, month, dayofweek, ECT, price",
        },
        "Neural Network Regressor": {
            "desc": "Prediction of residual using Neural Network Regression Feed Foward.",
            "features": "temp_max_avg, temp_min_avg, rain_avg, is_holiday, month, dayofweek, ECT, price",
        },
        "Histogram Gradient Boosting Regressor": {
            "desc": "Prediction of residual using Neural Network Regression Feed Foward.",
            "features": "temp_max_avg, temp_min_avg, rain_avg, is_holiday, month, dayofweek, ECT, price",
        }
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
    y_test, y_pred, rmse, r2, mae, mape = train_model(df, chosen)
 
    # Metrics row
    m1, m2, m3 = st.columns(3)
    m1.metric("Model", chosen)
    m2.metric("RMSE", f"{rmse:,.2f} kWh")
    m3.metric("R² Score", f"{r2:.4f}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean Absolute Error", f"{mae:.2f}")
    m3.metric("Mean Absolute Percentage Error", f"{mape:,.2f}")

    st.divider()
 
    # - PART 2: Prediction Graph -
    st.header("Prediction Output")
 
    tab1, tab2 = st.tabs(["Actual vs Predicted", "Predicted and Actual Over Time"])
 
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
        y_actual_aligned = df.loc[y_test.index].sort_index()
        y_pred_series = pd.Series(y_pred, index=y_test.index).sort_index()

        fig2 = go.Figure()
        # Actual demand
        fig2.add_trace(go.Scatter(
            x=y_actual_aligned.index, y=y_actual_aligned.demand,
            mode="lines",
            line=dict(color="#acf10c", width=1),
            opacity=0.4,
            name="Actual Demand",
        ))

        # Predicted demand
        fig2.add_trace(go.Scatter(
            x=y_pred_series.index, y=y_pred_series,
            mode="lines",
            line=dict(color="#F83003", width=2),
            opacity=0.4,
            name="Predicted Demand",
        ))

        fig2.update_layout(
            title=f"Predicted vs Actual Demand Over Time — {chosen}",
            xaxis_title="Date",
            yaxis_title="Demand (MWh)",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="#e0e0e0",
            legend=dict(bgcolor="#161b22", bordercolor="#30363d")
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
 
    # - PART 3: EDA -
    st.header("Exploratory Data Analysis")
 
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["Electricity Demand Over Time", 
                            "Scatter Plot Matrix", "Correlation Matrix", 
                            "Distribution of Electricity Demand",
                            "Temperature vs Electricity Demand",
                            "Holiday vs Non-Holiday",
                            "Demand and Price Trends",
                            "Demand-Price Relationship",
                            "Monthly Average Analysis"])
 
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
            colorscale='Viridis'
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
    
    with tab7:
        st.subheader("Demand and Price Trends Over Time")
        
        # Prepare data
        df_viz = df.reset_index()
        df_viz = df_viz[['date', 'demand', 'price']].dropna()
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Price", f"${df_viz['price'].mean():.4f}/kWh")
        with col2:
            st.metric("Min Price", f"${df_viz['price'].min():.4f}/kWh")
        with col3:
            st.metric("Max Price", f"${df_viz['price'].max():.4f}/kWh")
        with col4:
            correlation = df_viz['demand'].corr(df_viz['price'])
            st.metric("Correlation", f"{correlation:.3f}")
        
        # Time Series Chart
        fig1 = go.Figure()
        
        fig1.add_trace(go.Scatter(
            x=df_viz['date'], y=df_viz['demand'],
            name='Demand (kWh)', line=dict(color='#58a6ff', width=2), yaxis='y1'
        ))
        fig1.add_trace(go.Scatter(
            x=df_viz['date'], y=df_viz['price'],
            name='Price ($/kWh)', line=dict(color='#f85149', width=2), yaxis='y2'
        ))
        
        fig1.update_layout(
            title="Demand and Price Trends Over Time",
            xaxis=dict(title='Date'),
            yaxis=dict(title=dict(text='Demand (kWh)', font=dict(color='#58a6ff')), tickfont=dict(color='#58a6ff')),
            yaxis2=dict(title=dict(text='Price ($/kWh)', font=dict(color='#f85149')), tickfont=dict(color='#f85149'), overlaying='y', side='right'),
            hovermode='x unified', height=500, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#e0e0e0"
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with tab8:
        st.subheader("Demand-Price Relationship")
        
        df_viz = df.reset_index()
        df_viz = df_viz[['date', 'demand', 'price']].dropna()
        
        fig2 = px.scatter(
            df_viz, x='demand', y='price', trendline='ols',
            labels={'demand': 'Demand (kWh)', 'price': 'Price ($/kWh)'},
            title="Demand vs Price with Trend Line"
        )
        fig2.update_traces(marker=dict(size=5, opacity=0.6, color='#58a6ff'))
        fig2.update_layout(height=500, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#e0e0e0")
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab9:
        st.subheader("Monthly Average Analysis")
        
        df_viz = df.reset_index()
        df_viz = df_viz[['date', 'demand', 'price']].dropna()
        
        df_viz['month_name'] = pd.to_datetime(df_viz['date']).dt.strftime('%b %Y')
        monthly_avg = df_viz.groupby('month_name').agg({'demand': 'mean', 'price': 'mean'}).reset_index()
        monthly_avg = monthly_avg.tail(12)
        
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=monthly_avg['month_name'], y=monthly_avg['demand'], name='Avg Demand (kWh)', marker_color='#58a6ff', yaxis='y1'))
        fig3.add_trace(go.Bar(x=monthly_avg['month_name'], y=monthly_avg['price'], name='Avg Price ($/kWh)', marker_color='#f85149', yaxis='y2'))
        
        fig3.update_layout(
            title="Monthly Average Demand vs Price (Last 12 Months)",
            xaxis=dict(title='Month'),
            yaxis=dict(title=dict(text='Avg Demand (kWh)', font=dict(color='#58a6ff')), tickfont=dict(color='#58a6ff')),
            yaxis2=dict(title=dict(text='Avg Price ($/kWh)', font=dict(color='#f85149')), tickfont=dict(color='#f85149'), overlaying='y', side='right'),
            barmode='group', height=400, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#e0e0e0"
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    st.divider()


    # - PART 4: Dataset -
    st.header("Dataset")
 
    display_cols = [
        "demand","price" ,"ECT", "temp_max_avg", "temp_min_avg", "rain_avg",
        "is_holiday", "month", "dayofweek",
        "temp_max_akl", "temp_min_akl", "rain_akl",
        "temp_max_wlg", "temp_min_wlg", "rain_wlg",
        "temp_max_chc", "temp_min_chc", "rain_chc",
        "temp_max_ham", "temp_min_ham", "rain_ham",
        "temp_max_tau", "temp_min_tau", "rain_tau",
        "temp_max_dun", "temp_min_dun", "rain_dun",
        "temp_max_qtn", "temp_min_qtn", "rain_qtn",
        "temp_max_rot", "temp_min_rot", "rain_rot",
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
    