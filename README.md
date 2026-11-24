# ⚡ Smart Energy Consumption Forecasting Using Machine Learning

<p align="center">
  <strong>An End-to-End Machine Learning System for Energy Consumption Analysis, Forecasting and Intelligent Recommendations</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange" />
  <img src="https://img.shields.io/badge/Deep%20Learning-TensorFlow-red?logo=tensorflow" />
  <img src="https://img.shields.io/badge/Forecasting-Time%20Series-purple" />
  <img src="https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
</p>

---

## 📌 1. Project Overview

Electricity consumption forecasting is an important application of data science and machine learning. Understanding historical energy usage and predicting future demand can support efficient energy management, demand planning, and informed consumption decisions.

**Smart Energy Consumption Forecasting Using Machine Learning** is an end-to-end data science project that analyzes historical household electricity consumption, identifies usage patterns, develops predictive models, and generates energy-saving recommendations.

The project uses the Individual Household Electric Power Consumption dataset and incorporates historical weather information to investigate the relationship between environmental conditions and energy consumption.

It combines traditional machine learning, statistical time-series forecasting, deep learning, data visualization, and an interactive Streamlit dashboard into a unified workflow.

### Core Capabilities

- Historical electricity consumption analysis
- Data preprocessing and quality assessment
- Exploratory Data Analysis (EDA)
- Time-based and lag-based feature engineering
- Comparative machine learning model development
- Statistical time-series forecasting
- LSTM-based deep learning forecasting
- Weather-based energy prediction
- Model evaluation and performance visualization
- Intelligent energy consumption recommendations
- Interactive Streamlit dashboard

---

## 🎯 2. Project Objectives

1. Analyze historical household electricity consumption data.
2. Identify hourly, daily, monthly, and seasonal consumption patterns.
3. Perform data cleaning, preprocessing, and feature engineering.
4. Develop multiple machine learning regression models.
5. Compare statistical, machine learning, and deep learning forecasting approaches.
6. Investigate the contribution of historical weather features to energy prediction.
7. Evaluate predictive performance using standard regression metrics.
8. Generate consumption insights and energy-saving recommendations.
9. Present analytical results through an interactive dashboard.

---

## 📊 3. Dataset Description

### Primary Dataset: Individual Household Electric Power Consumption

**Source:** UCI Machine Learning Repository

The dataset contains electricity consumption measurements from a single household in Sceaux, France.

| Dataset Attribute | Description |
|---|---|
| Dataset | Individual Household Electric Power Consumption |
| Source | UCI Machine Learning Repository |
| Location | Sceaux, France |
| Collection Period | December 2006 – November 2010 |
| Original Frequency | One-minute measurements |
| Processed Frequency | Hourly |
| Prediction Target | `Energy_kWh` |

The original dataset contains household electricity measurements, including active power, reactive power, voltage, intensity, and sub-metering information.

### Processed Dataset

The feature-engineered hourly dataset contains:

- **31,428 hourly records**
- **15 columns**
- Target variable: `Energy_kWh`
- Time-based and engineered consumption features

### Weather Dataset

Historical weather information was obtained from the Open-Meteo Historical Weather API using ERA5-Land data.

Weather-related features include:

- Temperature
- Relative humidity

The weather data is used to investigate whether environmental conditions provide additional predictive information.

**Data sources:**
- UCI: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
- Open-Meteo: https://open-meteo.com/

---

## 🛠️ 4. Technology Stack

| Category | Tools and Technologies |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Gradient Boosting | XGBoost |
| Deep Learning | TensorFlow, Keras |
| Statistical Forecasting | Statsmodels, Prophet |
| Model Evaluation | MAE, MSE, RMSE, R², MAPE, WAPE, sMAPE |
| Web Application | Streamlit |
| Model Serialization | Joblib, Pickle |
| Development Environment | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |

---

## 🔄 5. Project Methodology

The project follows a structured end-to-end machine learning workflow.

### Step 1: Data Collection

- Collect historical household electricity consumption data.
- Obtain historical weather information.
- Understand dataset attributes and measurement frequencies.

### Step 2: Data Preprocessing

- Inspect data types and dataset structure.
- Handle missing values.
- Identify duplicate records and timestamp inconsistencies.
- Analyze incomplete hourly and daily observations.
- Prepare data for forecasting experiments.

### Step 3: Exploratory Data Analysis

Perform statistical and visual analysis to understand:

- Hourly energy consumption patterns
- Daily consumption trends
- Monthly consumption variations
- Seasonal energy usage
- Weekday versus weekend consumption
- Consumption distributions
- Outliers and unusual consumption patterns
- Feature correlations

### Step 4: Feature Engineering

Develop meaningful predictive features, including:

- Hour
- Day of the week
- Month
- Cyclical time features
- Historical energy consumption lags
- Weather-related variables
- Other time-dependent features

### Step 5: Model Development

Develop and evaluate different machine learning, statistical forecasting, and deep learning approaches.

### Step 6: Model Evaluation

Compare model predictions against actual energy consumption using appropriate evaluation metrics.

### Step 7: Forecasting

Generate energy consumption predictions and analyze future demand patterns.

### Step 8: Recommendation Generation

Analyze high-consumption periods and generate practical energy-saving recommendations.

### Step 9: Dashboard Development

Integrate analytical outputs and recommendation functionality into an interactive Streamlit application.

---

## 🤖 6. Machine Learning Models

### 6.1 Regression Models

Five regression algorithms were implemented and compared:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. XGBoost Regressor

### Model Performance Comparison

The following results were obtained in the original machine learning model comparison.

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| XGBoost | 0.3346 | 0.2302 | 0.4798 | 0.5802 |
| Gradient Boosting | 0.3422 | 0.2377 | 0.4875 | 0.5665 |
| Random Forest | 0.3437 | 0.2431 | 0.4930 | 0.5567 |
| Linear Regression | 0.3716 | 0.2696 | 0.5192 | 0.5084 |
| Decision Tree | 0.3795 | 0.3150 | 0.5613 | 0.4255 |

### Best Performing Model: XGBoost

XGBoost achieved:

- MAE: **0.3346**
- MSE: **0.2302**
- RMSE: **0.4798**
- R²: **0.5802**

Among the five evaluated regression models, XGBoost achieved the lowest MAE and RMSE and the highest R².

---

## ⏱️ 7. Advanced Time-Series Forecasting

Advanced forecasting experiments were conducted to investigate statistical and deep learning approaches to energy consumption prediction.

### Statistical Forecasting Models

- ARIMA
- SARIMA
- Tuned SARIMA
- Prophet

### Deep Learning Model

**Long Short-Term Memory (LSTM)**

The LSTM model was developed to learn temporal dependencies from historical hourly electricity consumption sequences.

### Statistical Forecasting Results

The following models were compared using the same 89-day statistical forecasting holdout period.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Tuned SARIMA | 9.4488 | 10.5434 | -0.0085 |
| ARIMA | 9.6891 | 10.6756 | -0.0340 |
| Seasonal Naive | 9.9809 | 11.2597 | -0.1502 |
| Original SARIMA | 10.5337 | 11.4981 | -0.1994 |
| Prophet | 11.7434 | 12.8688 | -0.5025 |

**Observation:** Tuned SARIMA achieved the lowest RMSE among the evaluated statistical forecasting models.

### LSTM Evaluation

The LSTM model was evaluated using a chronological hourly test set.

| Metric | Result |
|---|---:|
| MAE | 0.375068 |
| MSE | 0.322825 |
| RMSE | 0.568177 |
| R² | 0.413321 |
| MAPE | 36.483353% |
| WAPE | 37.598713% |
| sMAPE | 37.580784% |

The LSTM experiment used a 168-hour lookback window and a rolling one-step-ahead evaluation setup.

---

## 🌦️ 8. Weather-Based Energy Forecasting

A comparative experiment was conducted to evaluate the contribution of weather features to energy consumption prediction.

Two Random Forest models were evaluated:

- Baseline model: Without weather features
- Weather-based model: With weather features

Both models used the same Random Forest configuration and chronological evaluation split.

### Performance Comparison

| Evaluation Metric | Without Weather | With Weather |
|---|---:|---:|
| MAE | 0.3421 | 0.3418 |
| MSE | 0.2456 | 0.2440 |
| RMSE | 0.4956 | 0.4939 |
| R² | 0.5536 | 0.5566 |
| WAPE | 34.2982% | 34.2601% |
| sMAPE | 35.3546% | 35.3931% |

### Key Observation

The weather-based Random Forest model achieved approximately **0.34% relative RMSE improvement** compared with the baseline model.

This indicates a small improvement in the evaluated experiment when historical weather features are included.

The experiment uses observed historical weather values. It does not represent a future forecast using predicted weather inputs.

---

## 📈 9. Model Evaluation Metrics

The project uses the following metrics to assess model performance.

| Metric | Purpose |
|---|---|
| MAE | Measures average absolute prediction error |
| MSE | Measures average squared prediction error |
| RMSE | Measures prediction error with greater sensitivity to large errors |
| R² | Measures the proportion of variance explained by the model |
| MAPE | Measures average percentage prediction error |
| WAPE | Measures total absolute error relative to total actual consumption |
| sMAPE | Measures symmetric percentage prediction error |

Multiple evaluation metrics are used because no single metric completely describes forecasting performance.

---

## 💡 10. Intelligent Energy Recommendation System

A recommendation system was implemented to analyze household energy consumption and generate actionable insights.

### Main Capabilities

- Identify high-consumption observations using a 90th-percentile threshold.
- Detect peak energy consumption hours.
- Identify low-consumption hours.
- Analyze daily energy consumption.
- Compare weekday and weekend usage.
- Investigate temperature and consumption relationships.
- Generate energy-saving recommendations.
- Produce a high-consumption report.

### Observed Dataset Insights

| Analysis | Result |
|---|---|
| Average hourly energy consumption | 1.0894 kWh |
| High-consumption threshold | 2.3393 kWh |
| High-consumption records | 3,143 |
| Peak average consumption hour | 20:00 |
| Lowest average consumption hour | 03:00 |

The recommendations are based on whole-house consumption patterns. They do not identify individual appliance-level consumption.

---

## 🌐 11. Interactive Streamlit Dashboard

The project includes an interactive dashboard developed using Streamlit.

The dashboard connects the project's analytical and recommendation components through a graphical interface.

### Dashboard Capabilities

- Energy consumption analysis
- Historical usage visualization
- Forecasting result presentation
- Model performance analysis
- Weather-based forecasting results
- Energy consumption insights
- High-consumption analysis
- Energy-saving recommendations

### Run the Application

```bash
streamlit run website/app.py
```

The application can be accessed through the local URL displayed in the terminal.

---

## 📂 12. Repository Structure

```text
Smart-Energy-Consumption-Forecasting-Using-Machine-Learning/
│
├── dataset/
│   ├── energy_weather_merged.csv
│   ├── feature_engineered_hourly.csv
│   └── historical_weather_sceaux.csv
│
├── notebooks/
│   ├── 01_task.ipynb
│   └── 02_Advanced_Forecasting.ipynb
│
├── src/
│   └── recommendation_system.py
│
├── models/
│   ├── decision_tree_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── linear_regression_model.pkl
│   ├── lstm_energy_model.keras
│   ├── lstm_scaler.pkl
│   ├── minmax_scaler.pkl
│   ├── standard_scaler.pkl
│   ├── weather_random_forest.pkl
│   └── xgboost_model.pkl
│
├── evaluation/
│   ├── advanced_forecasting/
│   ├── feature_engineering/
│   ├── future_forecasting/
│   ├── model_development/
│   └── weather_forecasting/
│
├── visualizations/
│   ├── advanced_forecasting/
│   ├── eda/
│   ├── future_forecasting/
│   ├── insights_recommendations/
│   └── model_development/
│
├── website/
│   └── app.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** The original raw dataset, cleaned dataset, and one large model artifact are excluded from this GitHub repository because of individual file-size limitations. The files remain in the local project directory.

---

## ⚙️ 13. Installation and Execution

### Prerequisites

- Python 3.x
- Git
- VS Code or another Python IDE

### Step 1: Clone the Repository

```bash
git clone https://github.com/kancharla-vyshnavi/Smart-Energy-Consumption-Forecasting-Using-Machine-Learning.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Smart-Energy-Consumption-Forecasting-Using-Machine-Learning
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Launch the Streamlit Application

```bash
streamlit run website/app.py
```

**Important:** The dashboard requires its referenced dataset and trained model files. The excluded large files must be available locally if a particular application component depends on them.

---

## 🔍 14. Key Findings and Observations

1. Household energy consumption exhibits noticeable hourly, daily, monthly, and seasonal variations.
2. Peak consumption patterns can be identified through historical energy analysis.
3. XGBoost achieved the best results among the five original regression models evaluated.
4. Tuned SARIMA achieved the lowest RMSE among the evaluated statistical forecasting models.
5. LSTM demonstrated the application of sequence-based deep learning to hourly consumption data.
6. Historical weather features provided a small improvement in the evaluated Random Forest experiment.
7. Consumption analysis can support practical energy-saving recommendations.

**Evaluation note:** The statistical forecasting, LSTM, and weather-based experiments use different target periods and evaluation setups. Their raw RMSE values should not be directly ranked against one another.

---

## ⚠️ 15. Project Scope and Limitations

- The primary dataset represents a single household in Sceaux, France.
- Results may vary when applied to different households or geographical locations.
- Forecasting accuracy depends on data quality, feature availability, and evaluation methodology.
- Historical weather information is not equivalent to future weather forecasts.
- The recommendation system analyzes whole-house energy consumption rather than individual appliance usage.
- The dashboard is an analytical demonstration and is not connected to a live smart meter or utility control system.

---

## 🎓 16. Skills Demonstrated

This project demonstrates practical implementation of:

- Python programming
- Data cleaning and preprocessing
- Exploratory Data Analysis
- Time-series data analysis
- Feature engineering
- Regression modelling
- Machine learning model comparison
- Statistical forecasting
- Deep learning with LSTM
- Weather-based predictive modelling
- Model evaluation
- Data visualization
- Recommendation system development
- Streamlit application development
- Git and GitHub version control

---

## 👩‍💻 17. Author

**Vyshnavi**

B.Tech – Artificial Intelligence

**GitHub:** [@kancharla-vyshnavi](https://github.com/kancharla-vyshnavi)

**Project Repository:** [Smart Energy Consumption Forecasting Using Machine Learning](https://github.com/kancharla-vyshnavi/Smart-Energy-Consumption-Forecasting-Using-Machine-Learning)

---

## 🙏 18. Acknowledgements

- **UCI Machine Learning Repository** – Individual Household Electric Power Consumption dataset.
- **Open-Meteo** – Historical weather data.
- **Python Open-Source Community** – Machine learning, data science, and visualization libraries.

---

## 🌐 Live Website

https://smart-energy-consumption-forecasting-using-machine-learning-lz.streamlit.app/

<p align="center">
  <strong>⚡ SMART ENERGY CONSUMPTION FORECASTING</strong>
  <br/>
  <em>Transforming Historical Energy Data into Predictive Insights.</em>
  <br/><br/>
  ⭐ If you find this project useful, consider starring the repository!
</p>
