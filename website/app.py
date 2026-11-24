# ============================================================
# SMART ENERGY CONSUMPTION FORECASTING
# ADVANCED TASK 4: INTERACTIVE STREAMLIT DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib
import plotly.express as px


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Energy Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. THEME AND DATE FILTER STYLING
# ============================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F7FAF8;
    color: #172B26;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #153C30;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* Main headings */
h1, h2, h3 {
    color: #153C30 !important;
    font-weight: 750 !important;
}

p {
    color: #52665D;
}

/* KPI cards */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    padding: 20px;
    border: 1px solid #E1EAE4;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(25,65,47,0.05);
}

[data-testid="stMetricValue"] {
    color: #16845B !important;
    font-weight: 750;
}

[data-testid="stMetricLabel"] {
    color: #64796D !important;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background-color: #16845B !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #106B49 !important;
    color: #FFFFFF !important;
}

/* Keep the prediction form button text white */
div[data-testid="stFormSubmitButton"] button,
div[data-testid="stFormSubmitButton"] button p,
div[data-testid="stFormSubmitButton"] button span {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* ============================================================
   DATE FILTER PANEL
   ============================================================ */

.st-key-date_filter_panel {
    background-color: #153C30 !important;
    padding: 18px 18px 14px 18px !important;
    border-radius: 18px !important;
    border: none !important;
    margin: 18px 0 26px 0 !important;
}

/* Date filter heading and label */
.st-key-date_filter_panel h2,
.st-key-date_filter_panel h3,
.st-key-date_filter_panel p,
.st-key-date_filter_panel label {
    color: #FFFFFF !important;
}

/* White date box */
.st-key-date_filter_panel [data-testid="stDateInput"] {
    color-scheme: light !important;
}

.st-key-date_filter_panel [data-testid="stDateInput"] input {
    background-color: #FFFFFF !important;
    color: #153C30 !important;
    -webkit-text-fill-color: #153C30 !important;
    opacity: 1 !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    caret-color: #153C30 !important;
    color-scheme: light !important;
    border: none !important;
}

.st-key-date_filter_panel [data-testid="stDateInput"]
[data-baseweb="input"] {
    background-color: #FFFFFF !important;
    border: 2px solid #FFFFFF !important;
    border-radius: 12px !important;
    min-height: 58px !important;
    box-shadow: none !important;
}

/* Placeholder */
.st-key-date_filter_panel [data-testid="stDateInput"]
input::placeholder {
    color: #52665D !important;
    -webkit-text-fill-color: #52665D !important;
    opacity: 1 !important;
}

/* Calendar button and icon */
.st-key-date_filter_panel [data-testid="stDateInput"] button {
    background-color: #FFFFFF !important;
    color: #16845B !important;
}

.st-key-date_filter_panel [data-testid="stDateInput"] svg {
    color: #16845B !important;
}

/* Focus border */
.st-key-date_filter_panel [data-testid="stDateInput"]
[data-baseweb="input"]:focus-within {
    border: 2px solid #16845B !important;
    box-shadow: 0 0 0 2px rgba(22,132,91,0.15) !important;
}

/* Date filter note */
.date-filter-note {
    color: #D8F3E4 !important;
    font-size: 13px !important;
    margin-top: 8px !important;
    margin-bottom: 0 !important;
}

/* Other input fields */
[data-testid="stSelectbox"] input,
[data-testid="stNumberInput"] input {
    background-color: #FFFFFF !important;
    color: #153C30 !important;
}

/* Tables */
[data-testid="stDataFrame"] {
    border: 1px solid #E1EAE4;
    border-radius: 12px;
}

/* Dashboard header */
.dashboard-header {
    background: linear-gradient(110deg, #153C30, #25845A);
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 25px;
}

.dashboard-header h1 {
    color: #FFFFFF !important;
    margin-bottom: 7px;
}

.dashboard-header p {
    color: #D8F3E4 !important;
    margin-bottom: 0;
}

.section-label {
    color: #16845B;
    font-size: 12px;
    font-weight: 750;
    letter-spacing: 1.3px;
}

.footer {
    text-align: center;
    color: #829187;
    padding: 20px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Import the Task 5 recommendation system
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from recommendation_system import (
        analyze_energy_consumption,
        generate_recommendations,
        generate_high_consumption_report
    )
except ImportError as e:
    st.error(
        "Could not import the recommendation system. "
        "Check that src/recommendation_system.py exists. "
        f"Details: {e}"
    )
    st.stop()

DATA_PATH = os.path.join(
    BASE_DIR, "dataset", "energy_weather_merged.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "weather_random_forest.pkl"
)

METRICS_PATH = os.path.join(
    BASE_DIR,
    "evaluation",
    "weather_forecasting",
    "final_weather_forecasting_report.csv"
)

IMPORTANCE_PATH = os.path.join(
    BASE_DIR,
    "evaluation",
    "weather_forecasting",
    "weather_feature_importance.csv"
)


# ============================================================
# 4. LOAD DATA AND TRAINED MODEL
# ============================================================

@st.cache_data
def load_energy_data():
    data = pd.read_csv(
        DATA_PATH,
        parse_dates=["DateTime"]
    )
    return data.set_index("DateTime").sort_index()


@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    required_files = [
        DATA_PATH,
        MODEL_PATH,
        METRICS_PATH
    ]

    missing_files = [
        path for path in required_files
        if not os.path.exists(path)
    ]

    if missing_files:
        st.error("Required project files are missing:")
        for path in missing_files:
            st.write(path)
        st.stop()

    energy_data = load_energy_data()
    model_metrics = load_metrics()
    weather_model = load_model()

    if energy_data.empty:
        st.error("Energy dataset is empty.")
        st.stop()

    required_columns = [
        "Energy_kWh",
        "Temperature_C",
        "Humidity_Percent"
    ]

    if not all(col in energy_data.columns for col in required_columns):
        st.error("Required energy dataset columns are missing.")
        st.stop()

except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()


# ============================================================
# 5. MODEL FEATURE ORDER
# ============================================================

FEATURES = [
    "Hour_sin",
    "Hour_cos",
    "Day_sin",
    "Day_cos",
    "Month",
    "Energy_lag_1",
    "Energy_lag_24",
    "Energy_lag_168",
    "Temperature_C",
    "Humidity_Percent"
]

if hasattr(weather_model, "feature_names_in_"):
    saved_features = list(weather_model.feature_names_in_)

    if saved_features != FEATURES:
        st.error("Model feature order does not match dashboard.")
        st.write("Model features:", saved_features)
        st.write("Dashboard features:", FEATURES)
        st.stop()


# ============================================================
# 6. SIDEBAR NAVIGATION
# ============================================================

if "show_predict_page" not in st.session_state:
    st.session_state["show_predict_page"] = False


def return_to_main_menu():
    st.session_state["show_predict_page"] = False


with st.sidebar:

    st.markdown("# ⚡ SMART ENERGY")
    st.markdown("### FORECASTING SYSTEM")

    st.markdown("---")
    st.markdown("**MAIN MENU**")

    main_page = st.radio(
        "Navigation",
        [
            "Dashboard Overview",
            "Historical Consumption",
            "Peak Usage Analysis",
            "Energy Saving Recommendations",
            "Model Evaluation"
        ],
        label_visibility="collapsed",
        key="main_menu",
        on_change=return_to_main_menu
    )

    st.markdown("---")

    # Quick Prediction - corrected small spacing
    st.markdown(
        "<div style='height:5px'></div>",
        unsafe_allow_html=True
    )

    st.markdown("**QUICK PREDICTION**")

    if st.button(
        "⚡ Predict Energy",
        use_container_width=True,
        key="open_prediction_page"
    ):
        st.session_state["show_predict_page"] = True

    st.markdown("---")
    st.markdown("**PROJECT DETAILS**")
    st.caption("Smart Energy Forecasting")
    st.caption("Dataset: Sceaux, France")
    st.caption("Period: 2007–2010")


page = (
    "Predict Energy"
    if st.session_state["show_predict_page"]
    else main_page
)


# ============================================================
# 7. COMMON HEADER
# ============================================================

st.markdown("""
<div class="dashboard-header">
    <h1>⚡ Smart Energy Analytics</h1>
    <p>
        Interactive energy monitoring, demand forecasting
        and peak usage analysis
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 8. DATE FILTER - MAIN PAGE PANEL
# ============================================================

minimum_date = energy_data.index.min().date()
maximum_date = energy_data.index.max().date()

with st.container(key="date_filter_panel"):

    st.markdown("### DATE FILTER")

    selected_dates = st.date_input(
        "Select analysis period",
        value=(minimum_date, maximum_date),
        min_value=minimum_date,
        max_value=maximum_date,
        key="analysis_date_range"
    )


# ============================================================
# 9. PROCESS DATE FILTER
# ============================================================

if isinstance(selected_dates, (tuple, list)):

    if len(selected_dates) == 2:
        start_date, end_date = selected_dates

    elif len(selected_dates) == 1:
        start_date = selected_dates[0]
        end_date = selected_dates[0]

    else:
        start_date = minimum_date
        end_date = maximum_date

else:
    start_date = selected_dates
    end_date = selected_dates


if start_date > end_date:
    st.error("Start date cannot be after end date.")
    st.stop()


start_datetime = pd.Timestamp(start_date)
end_datetime = pd.Timestamp(end_date) + pd.Timedelta(days=1)

filtered_data = energy_data.loc[
    (energy_data.index >= start_datetime) &
    (energy_data.index < end_datetime)
].copy()


# ============================================================
# 10. DASHBOARD OVERVIEW
# ============================================================

if page == "Dashboard Overview":

    if filtered_data.empty:
        st.warning("No data available for this date range.")
        st.stop()

    st.markdown(
        '<p class="section-label">MAIN DASHBOARD</p>',
        unsafe_allow_html=True
    )

    st.header("Energy Consumption Overview")

    total_energy = filtered_data["Energy_kWh"].sum()
    average_energy = filtered_data["Energy_kWh"].mean()
    maximum_energy = filtered_data["Energy_kWh"].max()
    peak_timestamp = filtered_data["Energy_kWh"].idxmax()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Energy", f"{total_energy:,.2f} kWh")
    c2.metric("Average Consumption", f"{average_energy:.3f} kWh")
    c3.metric("Maximum Hourly Usage", f"{maximum_energy:.3f} kWh")
    c4.metric(
        "Peak Usage Time",
        peak_timestamp.strftime("%d %b, %H:%M")
    )

    st.markdown("---")
    st.subheader("Historical Energy Consumption")

    daily_consumption = (
        filtered_data["Energy_kWh"].resample("D").sum()
    )

    fig = px.area(
        daily_consumption,
        title="Daily Historical Energy Consumption",
        labels={
            "value": "Energy (kWh)",
            "DateTime": "Date"
        }
    )

    fig.update_traces(
        line_color="#16845B",
        fillcolor="rgba(22,132,91,0.15)"
    )

    fig.update_layout(
        template="plotly_white",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Peak Usage Summary")

    hourly_average = (
        filtered_data.groupby(filtered_data.index.hour)
        ["Energy_kWh"].mean()
    )

    peak_hour = hourly_average.idxmax()

    st.success(
        f"Highest average consumption occurs around "
        f"{peak_hour:02d}:00 hours."
    )


# ============================================================
# 11. HISTORICAL CONSUMPTION
# ============================================================

elif page == "Historical Consumption":

    if filtered_data.empty:
        st.warning("No historical records for this date range.")
        st.stop()

    st.markdown(
        '<p class="section-label">HISTORICAL ANALYTICS</p>',
        unsafe_allow_html=True
    )

    st.header("Historical Energy Consumption")

    aggregation = st.selectbox(
        "Select time interval",
        ["Hourly", "Daily", "Weekly", "Monthly"]
    )

    if aggregation == "Hourly":
        historical_series = filtered_data["Energy_kWh"]
        chart_title = "Hourly Energy Consumption"

    elif aggregation == "Daily":
        historical_series = (
            filtered_data["Energy_kWh"].resample("D").sum()
        )
        chart_title = "Daily Energy Consumption"

    elif aggregation == "Weekly":
        historical_series = (
            filtered_data["Energy_kWh"].resample("W").sum()
        )
        chart_title = "Weekly Energy Consumption"

    else:
        historical_series = (
            filtered_data["Energy_kWh"].resample("ME").sum()
        )
        chart_title = "Monthly Energy Consumption"

    fig = px.line(
        historical_series,
        title=chart_title,
        labels={
            "value": "Energy (kWh)",
            "DateTime": "Date"
        }
    )

    fig.update_traces(
        line_color="#16845B",
        line_width=2
    )

    fig.update_layout(
        template="plotly_white",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Consumption Statistics")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Average",
        f"{filtered_data['Energy_kWh'].mean():.3f} kWh"
    )

    c2.metric(
        "Minimum",
        f"{filtered_data['Energy_kWh'].min():.3f} kWh"
    )

    c3.metric(
        "Maximum",
        f"{filtered_data['Energy_kWh'].max():.3f} kWh"
    )

    st.subheader("Historical Data Records")

    st.dataframe(
        filtered_data.head(500),
        use_container_width=True
    )

    st.download_button(
        "Download Historical Data",
        data=filtered_data.to_csv().encode("utf-8"),
        file_name="historical_energy_consumption.csv",
        mime="text/csv"
    )


# ============================================================
# 12. PEAK USAGE ANALYSIS
# ============================================================

elif page == "Peak Usage Analysis":

    if filtered_data.empty:
        st.warning("No data available for peak usage analysis.")
        st.stop()

    st.markdown(
        '<p class="section-label">PEAK DEMAND INTELLIGENCE</p>',
        unsafe_allow_html=True
    )

    st.header("Peak Energy Usage Analysis")

    hourly_profile = (
        filtered_data.groupby(filtered_data.index.hour)
        ["Energy_kWh"].mean()
        .reset_index()
    )

    hourly_profile.columns = [
        "Hour",
        "Average_Energy_kWh"
    ]

    peak_hour_row = hourly_profile.loc[
        hourly_profile["Average_Energy_kWh"].idxmax()
    ]

    peak_hour = int(peak_hour_row["Hour"])
    peak_average = peak_hour_row["Average_Energy_kWh"]

    maximum_reading = filtered_data["Energy_kWh"].max()
    maximum_timestamp = filtered_data["Energy_kWh"].idxmax()

    daily_totals = (
        filtered_data["Energy_kWh"].resample("D").sum()
    )

    peak_day = daily_totals.idxmax()
    peak_day_energy = daily_totals.max()

    a, b, c = st.columns(3)

    a.metric("Peak Average Usage Hour", f"{peak_hour:02d}:00")
    b.metric("Highest Hourly Consumption", f"{maximum_reading:.3f} kWh")
    c.metric("Highest Daily Consumption", f"{peak_day_energy:.2f} kWh")

    st.markdown("---")
    st.subheader("Average Energy Usage by Hour")

    fig = px.bar(
        hourly_profile,
        x="Hour",
        y="Average_Energy_kWh",
        title="Hourly Peak Usage Pattern",
        color="Average_Energy_kWh",
        color_continuous_scale=["#C7E9D5", "#16845B"]
    )

    fig.add_vline(
        x=peak_hour,
        line_dash="dash",
        line_color="#153C30",
        annotation_text="Peak Hour"
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Hour of Day",
        yaxis_title="Average Energy (kWh)",
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 Highest Consumption Days")

    top_days = (
        daily_totals.sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    top_days.columns = ["Date", "Daily_Energy_kWh"]

    fig = px.bar(
        top_days,
        x="Date",
        y="Daily_Energy_kWh",
        title="Top 10 Peak Consumption Days",
        color="Daily_Energy_kWh",
        color_continuous_scale=["#C7E9D5", "#16845B"]
    )

    fig.update_layout(
        template="plotly_white",
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Peak Usage Details")

    st.write(
        f"**Peak average usage period:** "
        f"{peak_hour:02d}:00–{(peak_hour + 1) % 24:02d}:00"
    )

    st.write(
        f"**Peak average hourly energy:** {peak_average:.4f} kWh"
    )

    st.write(
        f"**Highest individual hourly reading:** "
        f"{maximum_reading:.4f} kWh at {maximum_timestamp}"
    )

    st.write(
        f"**Highest daily energy consumption:** "
        f"{peak_day_energy:.4f} kWh on {peak_day.date()}"
    )

    st.dataframe(top_days, use_container_width=True)


# ============================================================
# 13. ENERGY SAVING RECOMMENDATIONS
# ============================================================

elif page == "Energy Saving Recommendations":

    st.markdown(
        '<p class="section-label">SMART ENERGY MANAGEMENT</p>',
        unsafe_allow_html=True
    )

    st.header("Energy Saving Recommendations")

    st.write(
        "Analyze the selected consumption period and identify "
        "possible energy-saving opportunities."
    )

    if filtered_data.empty:
        st.warning("No data available for the selected date range.")
        st.stop()

    # Generate recommendations using the selected date range
    try:
        recommendation_df = generate_recommendations(filtered_data)
        analysis = analyze_energy_consumption(filtered_data)
        high_report = generate_high_consumption_report(filtered_data)

        if recommendation_df.empty:
            st.warning(
                "No recommendations could be generated for this period."
            )
            st.stop()

    except Exception as e:
        st.error(f"Unable to generate recommendations: {e}")
        st.stop()

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Consumption",
        f"{filtered_data['Energy_kWh'].mean():.3f} kWh"
    )

    col2.metric(
        "High-Consumption Records",
        f"{len(high_report):,}"
    )

    col3.metric(
        "High-Consumption Threshold",
        f"{analysis['high_threshold']:.3f} kWh"
    )

    st.markdown("---")

    # Recommendation cards
    st.subheader("Recommended Energy-Saving Actions")

    for _, row in recommendation_df.iterrows():

        with st.container(border=True):

            st.markdown(f"### {row['Category']}")

            if row["Priority"] == "High":
                st.error(f"Priority: {row['Priority']}")
            else:
                st.info(f"Priority: {row['Priority']}")

            st.write("**Finding:**", row["Finding"])
            st.write("**Suggested Action:**", row["Recommendation"])

    st.markdown("---")

    # Hourly usage chart
    st.subheader("Average Energy Consumption by Hour")

    hourly_usage = (
        filtered_data.groupby(filtered_data.index.hour)["Energy_kWh"]
        .mean()
        .reindex(range(24))
        .dropna()
        .rename_axis("Hour")
        .reset_index(name="Average_Energy_kWh")
    )

    fig = px.bar(
        hourly_usage,
        x="Hour",
        y="Average_Energy_kWh",
        title="Hourly Energy Consumption Pattern",
        color="Average_Energy_kWh",
        color_continuous_scale=["#C7E9D5", "#16845B"]
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Hour of Day",
        yaxis_title="Average Energy (kWh)",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # High-consumption records
    st.subheader("Top 10 High-Consumption Records")

    st.dataframe(
        high_report.head(10),
        use_container_width=True
    )

    # Download outputs
    st.download_button(
        "Download Energy-Saving Recommendations",
        data=recommendation_df.to_csv(index=False).encode("utf-8"),
        file_name="energy_saving_recommendations.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download High-Consumption Report",
        data=high_report.to_csv(index=True).encode("utf-8"),
        file_name="high_consumption_report.csv",
        mime="text/csv"
    )

    st.caption(
        "These are rule-based suggestions derived from historical "
        "whole-house consumption patterns. They do not identify "
        "individual appliances or guarantee a specific amount of savings."
    )


# ============================================================
# 14. QUICK PREDICTION
# ============================================================

elif page == "Predict Energy":

    st.markdown(
        '<p class="section-label">INTERACTIVE MACHINE LEARNING</p>',
        unsafe_allow_html=True
    )

    st.header("Predict Your Energy Consumption")

    st.write(
        "Enter weather conditions, time information and "
        "previous energy consumption to estimate hourly demand."
    )

    st.info(
        "The trained Weather-Based Random Forest model "
        "uses the same 10 features used during training."
    )

    with st.form("energy_prediction_form"):

        st.subheader("1. Weather Information")

        c1, c2 = st.columns(2)

        with c1:
            temperature = st.number_input(
                "Temperature (°C)",
                min_value=-30.0,
                max_value=50.0,
                value=18.0,
                step=0.5
            )

        with c2:
            humidity = st.number_input(
                "Relative Humidity (%)",
                min_value=0,
                max_value=100,
                value=65,
                step=1
            )

        st.markdown("---")
        st.subheader("2. Time Information")

        c3, c4, c5 = st.columns(3)

        with c3:
            hour = st.slider(
                "Hour of Day",
                min_value=0,
                max_value=23,
                value=14
            )

        with c4:
            day_of_week = st.selectbox(
                "Day of Week",
                options=list(range(7)),
                format_func=lambda x: [
                    "Monday", "Tuesday", "Wednesday",
                    "Thursday", "Friday", "Saturday", "Sunday"
                ][x]
            )

        with c5:
            month = st.selectbox(
                "Month",
                options=list(range(1, 13)),
                format_func=lambda x: pd.Timestamp(
                    2024, x, 1
                ).strftime("%B")
            )

        st.markdown("---")
        st.subheader("3. Previous Energy Consumption")

        c6, c7, c8 = st.columns(3)

        with c6:
            lag_1 = st.number_input(
                "Previous 1-hour Energy (kWh)",
                min_value=0.0,
                value=2.5,
                step=0.1
            )

        with c7:
            lag_24 = st.number_input(
                "Previous 24-hour Energy (kWh)",
                min_value=0.0,
                value=2.8,
                step=0.1
            )

        with c8:
            lag_168 = st.number_input(
                "Previous 168-hour Energy (kWh)",
                min_value=0.0,
                value=2.3,
                step=0.1
            )

        st.markdown("---")

        submitted = st.form_submit_button(
            "⚡ Predict Energy Consumption",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{
            "Hour_sin": np.sin(2 * np.pi * hour / 24),
            "Hour_cos": np.cos(2 * np.pi * hour / 24),
            "Day_sin": np.sin(2 * np.pi * day_of_week / 7),
            "Day_cos": np.cos(2 * np.pi * day_of_week / 7),
            "Month": month,
            "Energy_lag_1": lag_1,
            "Energy_lag_24": lag_24,
            "Energy_lag_168": lag_168,
            "Temperature_C": temperature,
            "Humidity_Percent": humidity
        }], columns=FEATURES)

        try:
            prediction = float(
                weather_model.predict(input_data)[0]
            )

            st.markdown("---")
            st.subheader("4. Prediction Result")

            st.success(
                "Energy consumption prediction generated successfully!"
            )

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    "Predicted Energy Consumption",
                    f"{prediction:.4f} kWh"
                )

            with result_col2:
                st.metric(
                    "Selected Hour",
                    f"{hour:02d}:00"
                )

            q75 = energy_data["Energy_kWh"].quantile(0.75)
            q90 = energy_data["Energy_kWh"].quantile(0.90)

            if prediction >= q90:
                demand_category = "High"
            elif prediction >= q75:
                demand_category = "Moderate"
            else:
                demand_category = "Lower"

            st.subheader("Predicted Demand Category")

            st.info(
                f"Estimated demand category: **{demand_category}**"
            )

            st.subheader("Input Summary")

            st.dataframe(
                input_data,
                use_container_width=True
            )

            result_chart = pd.DataFrame({
                "Category": ["Predicted Energy"],
                "Energy_kWh": [prediction]
            })

            fig = px.bar(
                result_chart,
                x="Category",
                y="Energy_kWh",
                title="Predicted Hourly Energy Consumption",
                color_discrete_sequence=["#16845B"]
            )

            fig.update_layout(
                template="plotly_white",
                yaxis_title="Energy (kWh)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            result_df = input_data.copy()
            result_df["Predicted_Energy_kWh"] = prediction
            result_df["Demand_Category"] = demand_category

            st.download_button(
                "Download Prediction Result",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="energy_prediction_result.csv",
                mime="text/csv"
            )

            st.caption(
                "This is a model-based estimate for one hour. "
                "It is not a guaranteed future consumption value. "
                "Previous energy values must represent the "
                "corresponding historical lags."
            )

            # =====================================================
            # PREDICTION-BASED ENERGY SAVING RECOMMENDATIONS
            # =====================================================

            st.markdown("---")
            st.header("💡 Energy Saving Recommendations")

            st.write(
                "These suggestions are selected using your predicted "
                "energy consumption and the historical consumption "
                "thresholds of this household."
            )

            historical_q75 = energy_data["Energy_kWh"].quantile(0.75)
            historical_q90 = energy_data["Energy_kWh"].quantile(0.90)

            if prediction >= historical_q90:

                st.error(
                    "High predicted energy consumption compared "
                    "with historical household usage."
                )

                st.markdown("**Recommended actions:**")

                st.markdown(
                    "- Avoid operating several high-power appliances "
                    "at the same time.\n"
                    "- Switch off lights and devices that are not needed.\n"
                    "- Check for unnecessary standby power consumption.\n"
                    "- Monitor consumption during this predicted peak period."
                )

            elif prediction >= historical_q75:

                st.warning(
                    "Moderately high predicted energy consumption "
                    "compared with historical household usage."
                )

                st.markdown("**Recommended actions:**")

                st.markdown(
                    "- Reduce unnecessary appliance usage.\n"
                    "- Use energy-efficient operating settings where available.\n"
                    "- Avoid leaving electrical devices switched on unnecessarily."
                )

            else:

                st.success(
                    "Predicted energy consumption is relatively low "
                    "compared with historical household usage."
                )

                st.markdown("**Recommended actions:**")

                st.markdown(
                    "- Continue monitoring your energy consumption.\n"
                    "- Maintain energy-efficient daily habits.\n"
                    "- Switch off unused electrical appliances."
                )

            st.caption(
                "Recommendations are general rule-based guidance. "
                "They compare the predicted value with the historical "
                "75th and 90th percentile thresholds; they do not identify "
                "individual appliances or guarantee energy savings."
            )

        except Exception as e:
            st.error(f"Prediction could not be generated: {e}")


# ============================================================
# 15. MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.markdown(
        '<p class="section-label">MODEL PERFORMANCE</p>',
        unsafe_allow_html=True
    )

    st.header("Machine Learning Model Evaluation")
    st.subheader("Model Comparison")

    st.dataframe(
        model_metrics.round(4),
        use_container_width=True
    )

    if os.path.exists(IMPORTANCE_PATH):

        importance = pd.read_csv(IMPORTANCE_PATH)

        st.subheader("Weather Model Feature Importance")

        fig = px.bar(
            importance.sort_values(
                "Importance",
                ascending=True
            ),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance Analysis",
            color_discrete_sequence=["#16845B"]
        )

        fig.update_layout(
            template="plotly_white",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

    if {"Model", "RMSE"}.issubset(model_metrics.columns):

        fig = px.bar(
            model_metrics,
            x="Model",
            y="RMSE",
            title="RMSE Comparison Between Models",
            color_discrete_sequence=["#16845B"]
        )

        fig.update_layout(template="plotly_white")

        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 16. FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<div class="footer">
    ⚡ Smart Energy Consumption Forecasting
    <br>
    Interactive Streamlit Dashboard | Machine Learning Project
</div>
""", unsafe_allow_html=True)