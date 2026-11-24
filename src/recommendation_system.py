# ============================================================
# SMART ENERGY CONSUMPTION FORECASTING
# TASK 5: ENERGY SAVING RECOMMENDATION SYSTEM
# ============================================================

import pandas as pd
import numpy as np


# ============================================================
# 1. ENERGY CONSUMPTION ANALYSIS
# ============================================================

def analyze_energy_consumption(df):

    data = df.copy()

    if "DateTime" in data.columns:
        data["DateTime"] = pd.to_datetime(data["DateTime"])
        data = data.set_index("DateTime")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Dataset must have a DateTime index.")

    if "Energy_kWh" not in data.columns:
        raise ValueError("Energy_kWh column is missing.")

    data = data.sort_index()

    data["Energy_kWh"] = pd.to_numeric(
        data["Energy_kWh"],
        errors="coerce"
    )

    data = data.dropna(subset=["Energy_kWh"])

    if data.empty:
        raise ValueError("No valid energy consumption records found.")

    # Identify high consumption using the 90th percentile
    high_threshold = data["Energy_kWh"].quantile(0.90)

    high_consumption = data[
        data["Energy_kWh"] >= high_threshold
    ].copy()

    # Average consumption by hour
    hourly_usage = (
        data.groupby(data.index.hour)["Energy_kWh"]
        .mean()
        .sort_values(ascending=False)
    )

    # Average consumption by weekday
    weekday_usage = (
        data.groupby(data.index.dayofweek)["Energy_kWh"]
        .mean()
    )

    # Daily energy consumption
    daily_usage = (
        data["Energy_kWh"]
        .resample("D")
        .sum(min_count=1)
        .dropna()
    )

    daily_threshold = daily_usage.quantile(0.90)

    high_usage_days = daily_usage[
        daily_usage >= daily_threshold
    ].sort_values(ascending=False)

    return {
        "data": data,
        "high_threshold": high_threshold,
        "high_consumption": high_consumption,
        "hourly_usage": hourly_usage,
        "weekday_usage": weekday_usage,
        "daily_usage": daily_usage,
        "daily_threshold": daily_threshold,
        "high_usage_days": high_usage_days
    }


# ============================================================
# 2. GENERATE ENERGY SAVING RECOMMENDATIONS
# ============================================================

def generate_recommendations(df):

    analysis = analyze_energy_consumption(df)

    recommendations = []

    data = analysis["data"]
    high_threshold = analysis["high_threshold"]
    hourly_usage = analysis["hourly_usage"]

    # --------------------------------------------------------
    # Recommendation 1: High consumption periods
    # --------------------------------------------------------

    high_records = analysis["high_consumption"]

    if not high_records.empty:

        recommendations.append({
            "Category": "High Consumption",
            "Finding": (
                f"{len(high_records)} hourly records are at or "
                f"above {high_threshold:.3f} kWh."
            ),
            "Recommendation": (
                "Review the appliances and equipment operating "
                "during these high-consumption periods. Switch "
                "off unnecessary equipment and avoid running "
                "multiple high-power appliances simultaneously."
            ),
            "Priority": "High"
        })

    # --------------------------------------------------------
    # Recommendation 2: Peak usage hours
    # --------------------------------------------------------

    peak_hour = int(hourly_usage.idxmax())
    peak_hour_value = float(hourly_usage.max())

    recommendations.append({
        "Category": "Peak Usage Hours",
        "Finding": (
            f"Highest average hourly consumption occurs at "
            f"{peak_hour:02d}:00, averaging "
            f"{peak_hour_value:.3f} kWh."
        ),
        "Recommendation": (
            "Consider shifting flexible activities such as "
            "laundry, dishwashing and other non-essential "
            "equipment usage to lower-consumption hours, "
            "where practical."
        ),
        "Priority": "High"
    })

    # --------------------------------------------------------
    # Recommendation 3: High consumption days
    # --------------------------------------------------------

    high_days = analysis["high_usage_days"]

    if not high_days.empty:

        peak_day = high_days.index[0]
        peak_day_value = float(high_days.iloc[0])

        recommendations.append({
            "Category": "Daily Consumption",
            "Finding": (
                f"Highest daily consumption in the top 10% "
                f"of days includes {peak_day.date()}, with "
                f"{peak_day_value:.2f} kWh."
            ),
            "Recommendation": (
                "Review daily appliance usage and identify "
                "equipment that may have been left running. "
                "Monitor consumption on similar high-usage days."
            ),
            "Priority": "High"
        })

    # --------------------------------------------------------
    # Recommendation 4: Low usage hours
    # --------------------------------------------------------

    lowest_hour = int(hourly_usage.idxmin())
    lowest_value = float(hourly_usage.min())

    recommendations.append({
        "Category": "Load Scheduling",
        "Finding": (
            f"Lowest average hourly consumption occurs at "
            f"{lowest_hour:02d}:00, averaging "
            f"{lowest_value:.3f} kWh."
        ),
        "Recommendation": (
            "Compare this lower-consumption period with peak "
            "hours. Where convenient, schedule flexible "
            "electricity usage during lower-demand periods."
        ),
        "Priority": "Medium"
    })

    # --------------------------------------------------------
    # Recommendation 5: Weekend and weekday comparison
    # --------------------------------------------------------

    weekday_data = data[
        data.index.dayofweek < 5
    ]["Energy_kWh"]

    weekend_data = data[
        data.index.dayofweek >= 5
    ]["Energy_kWh"]

    if not weekday_data.empty and not weekend_data.empty:

        weekday_avg = weekday_data.mean()
        weekend_avg = weekend_data.mean()

        if weekend_avg > weekday_avg:

            finding = (
                f"Average weekend hourly consumption "
                f"({weekend_avg:.3f} kWh) is higher than "
                f"weekday consumption ({weekday_avg:.3f} kWh)."
            )

            recommendation = (
                "Review weekend appliance usage and identify "
                "activities that increase consumption. "
                "Switch off unused devices when not required."
            )

        else:

            finding = (
                f"Average weekday hourly consumption "
                f"({weekday_avg:.3f} kWh) is higher than "
                f"weekend consumption ({weekend_avg:.3f} kWh)."
            )

            recommendation = (
                "Review weekday operating schedules and "
                "consider switching off equipment when "
                "it is not required."
            )

        recommendations.append({
            "Category": "Weekly Usage Pattern",
            "Finding": finding,
            "Recommendation": recommendation,
            "Priority": "Medium"
        })

    # --------------------------------------------------------
    # Recommendation 6: Temperature-related patterns
    # --------------------------------------------------------

    if "Temperature_C" in data.columns:

        data["Temperature_C"] = pd.to_numeric(
            data["Temperature_C"],
            errors="coerce"
        )

        temperature_data = data.dropna(
            subset=["Temperature_C", "Energy_kWh"]
        )

        if len(temperature_data) > 1:

            correlation = temperature_data[
                ["Temperature_C", "Energy_kWh"]
            ].corr().iloc[0, 1]

            if pd.notna(correlation):

                if correlation <= -0.30:

                    finding = (
                        f"Temperature and energy consumption show "
                        f"a negative correlation of {correlation:.3f}."
                    )

                    recommendation = (
                        "Review heating-related energy usage during "
                        "colder periods. Use appropriate thermostat "
                        "settings and avoid unnecessary heating."
                    )

                elif correlation >= 0.30:

                    finding = (
                        f"Temperature and energy consumption show "
                        f"a positive correlation of {correlation:.3f}."
                    )

                    recommendation = (
                        "Review cooling-related equipment usage "
                        "during warmer periods. Use efficient cooling "
                        "settings and reduce unnecessary operation."
                    )

                else:

                    finding = (
                        f"Temperature has a weak linear correlation "
                        f"with energy consumption ({correlation:.3f})."
                    )

                    recommendation = (
                        "Temperature alone does not show a strong "
                        "linear relationship with consumption. "
                        "Continue monitoring other usage patterns."
                    )

                recommendations.append({
                    "Category": "Weather-Based Insight",
                    "Finding": finding,
                    "Recommendation": recommendation,
                    "Priority": "Medium"
                })

    # --------------------------------------------------------
    # Recommendation 7: General energy saving
    # --------------------------------------------------------

    recommendations.append({
        "Category": "General Energy Saving",
        "Finding": (
            "Continuous monitoring can help identify "
            "unnecessary electricity usage."
        ),
        "Recommendation": (
            "Turn off unused lights and appliances, use "
            "energy-efficient equipment, maintain electrical "
            "devices and monitor consumption regularly."
        ),
        "Priority": "Medium"
    })

    return pd.DataFrame(recommendations)


# ============================================================
# 3. GENERATE HIGH CONSUMPTION REPORT
# ============================================================

def generate_high_consumption_report(df):

    analysis = analyze_energy_consumption(df)

    high_records = analysis["high_consumption"].copy()

    report = high_records[
        ["Energy_kWh"]
    ].copy()

    report["Hour"] = report.index.hour
    report["Date"] = report.index.date

    report = report.sort_values(
        "Energy_kWh",
        ascending=False
    )

    return report


# ============================================================
# 4. PRINT RECOMMENDATION SUMMARY
# ============================================================

def print_recommendation_summary(df):

    analysis = analyze_energy_consumption(df)

    recommendations = generate_recommendations(df)

    print("=" * 65)
    print("SMART ENERGY SAVING RECOMMENDATION SYSTEM")
    print("=" * 65)

    print("\nDATASET SUMMARY")
    print("-" * 50)

    print("Total hourly records:", len(analysis["data"]))

    print(
        "Average energy consumption:",
        round(analysis["data"]["Energy_kWh"].mean(), 4),
        "kWh"
    )

    print(
        "High consumption threshold:",
        round(analysis["high_threshold"], 4),
        "kWh"
    )

    print(
        "High consumption records:",
        len(analysis["high_consumption"])
    )

    print("\nENERGY SAVING RECOMMENDATIONS")
    print("=" * 65)

    for index, row in recommendations.iterrows():

        print(f"\n{index + 1}. {row['Category']}")
        print("Finding:", row["Finding"])
        print("Recommendation:", row["Recommendation"])
        print("Priority:", row["Priority"])
        print("-" * 65)

    print("\nRECOMMENDATION GENERATION COMPLETED")

    return recommendations


# ============================================================
# END OF RECOMMENDATION SYSTEM
# ============================================================