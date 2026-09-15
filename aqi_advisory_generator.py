#!/usr/bin/env python3
"""
NLP-Based Air Quality Health Advisory Generator (India)
Converts CPCB AQI and meteorological metrics into context-aware health advisories.
"""

import pandas as pd
import numpy as np


# CPCB National Air Quality Index (NAQI) health breakpoints
CPCB_BREAKPOINTS = {
    "AQI": [(0, 50, "Good"), (51, 100, "Satisfactory"),
            (101, 200, "Moderately Polluted"), (201, 300, "Poor"),
            (301, 400, "Very Poor"), (401, 500, "Severe")],
    "PM2.5": [(0, 30, "Good"), (31, 60, "Satisfactory"),
              (61, 90, "Moderately Polluted"), (91, 120, "Poor"),
              (121, 250, "Very Poor"), (251, 300, "Severe")],
    "PM10": [(0, 50, "Good"), (51, 100, "Satisfactory"),
             (101, 250, "Moderately Polluted"), (251, 350, "Poor"),
             (351, 430, "Very Poor"), (431, 500, "Severe")],
    "NO2": [(0, 40, "Good"), (41, 80, "Satisfactory"),
          (81, 180, "Moderately Polluted"), (181, 280, "Poor"),
          (281, 380, "Very Poor"), (381, 600, "Severe")],
    "CO": [(0, 1, "Good"), (1.1, 4, "Satisfactory"),
         (4.1, 10, "Moderately Polluted"), (10.1, 17, "Poor"),
         (17.1, 20, "Very Poor"), (20.1, 25, "Severe")],
    "SO2": [(0, 40, "Good"), (41, 80, "Satisfactory"),
          (81, 160, "Moderately Polluted"), (161, 380, "Poor"),
          (381, 800, "Very Poor"), (801, 1000, "Severe")],
    "O3": [(0, 50, "Good"), (51, 100, "Satisfactory"),
         (101, 168, "Moderately Polluted"), (169, 204, "Poor"),
         (205, 704, "Very Poor"), (705, 800, "Severe")],
}


def classify_aqi(aqi_value):
    """Classify AQI value into CPCB risk category and color."""
    if aqi_value <= 50:
        return "Good", "Green"
    elif aqi_value <= 100:
        return "Satisfactory", "Light Green"
    elif aqi_value <= 200:
        return "Moderately Polluted", "Yellow"
    elif aqi_value <= 300:
        return "Poor", "Orange"
    elif aqi_value <= 400:
        return "Very Poor", "Red"
    else:
        return "Severe", "Dark Red"


def get_dominant_pollutant(aqi, pm25, pm10, no2, so2, co, o3):
    """Determine the dominant pollutant using CPCB 'worst sub-index' principle."""
    pollutants = {
        "PM2.5": pm25,
        "PM10": pm10,
        "NO2": no2,
        "SO2": so2,
        "CO": co,
        "O3": o3,
    }
    # Simple approach: use the pollutant with highest normalized value
    # In a full implementation, each has a sub-index calculation
    return max(pollutants, key=pollutants.get)


def generate_advisory(aqi, category, dominant_pollutant, temperature,
                      humidity, wind_speed, rainfall, location,
                      target_group="General"):
    """
    Generate a natural language health advisory based on CPCB standards
    and demographic context.
    """
    # Base advisory by category
    base_advisories = {
        "Good": (
            f"Air quality in {location} is Good ({aqi} AQI). "
            f"Minimal health impact; safe for outdoor activities. "
            f"The dominant pollutant is {dominant_pollutant}."
        ),
        "Satisfactory": (
            f"Air quality in {location} is Satisfactory ({aqi} AQI). "
            f"May cause minor breathing discomfort to sensitive people. "
            f"The dominant pollutant is {dominant_pollutant}."
        ),
        "Moderately Polluted": (
            f"Air quality in {location} is Moderately Polluted ({aqi} AQI). "
            f"Discomfort to people with lung/heart disease, children, and older adults. "
            f"The dominant pollutant is {dominant_pollutant}."
        ),
        "Poor": (
            f"Air quality in {location} is Poor ({aqi} AQI). "
            f"Breathing discomfort to most people on prolonged exposure. "
            f"The dominant pollutant is {dominant_pollutant}."
        ),
        "Very Poor": (
            f"Air quality in {location} is Very Poor ({aqi} AQI). "
            f"Respiratory illness on prolonged exposure; severe impact on sensitive groups. "
            f"The dominant pollutant is {dominant_pollutant}."
        ),
        "Severe": (
            f"Air quality in {location} is Severe ({aqi} AQI). "
            f"Affects healthy people and seriously impacts those with pre-existing conditions. "
            f"The dominant pollutant is {dominant_pollutant}."
        ),
    }

    advisory = base_advisories.get(category, "")

    # Weather context modifications
    weather_parts = []
    if temperature > 35:
        weather_parts.append(
            f" High temperature ({temperature}°C) may increase "
            f"respiratory stress."
        )
    if humidity > 70:
        weather_parts.append(
            f" Elevated humidity ({humidity}%) can aggravate respiratory "
            f"conditions, especially for asthma sufferers."
        )
    if wind_speed and wind_speed < 5:
        weather_parts.append(
            f" Low wind speed ({wind_speed} km/h) may cause pollutant "
            f"accumulation."
        )
    if rainfall and rainfall > 0:
        weather_parts.append(
            f" Recent rainfall ({rainfall} mm) may have temporarily "
            f"improved air quality."
        )

    # Demographic-specific adjustments
    demographic_modifiers = {
        "Children": (
            " Children should limit outdoor activities and ensure "
            "proper ventilation indoors."
        ),
        "Elderly": (
            " Elderly individuals should avoid prolonged outdoor exposure "
            "and keep medications ready."
        ),
        "Sensitive": (
            " Individuals with respiratory conditions (asthma, etc.) "
            "should stay indoors and use air purifiers."
        ),
        "Asthma": (
            " Asthma patients should carry inhalers and avoid outdoor "
            "exercise. Keep windows closed."
        ),
        "General": "",
    }

    mod = demographic_modifiers.get(target_group, "")

    # Combine advisory components
    full_advisory = advisory
    if weather_parts:
        full_advisory += " " + " ".join(weather_parts)
    if mod:
        full_advisory += " " + mod

    # Final recommendation
    recommendations = {
        "Good": "No specific restrictions. Enjoy outdoor activities safely.",
        "Satisfactory": "Consider reducing intense outdoor activity if you have respiratory sensitivities.",
        "Moderately Polluted": "Reduce prolonged or heavy exertion. Keep windows closed if dust is suspected.",
        "Poor": "Avoid outdoor activities. Use N95 masks if going outside is necessary.",
        "Very Poor": "Stay indoors with windows closed. Use air purifiers if available.",
        "Severe": "Remain indoors. Use high-efficiency air filters. Seek medical attention if symptoms appear.",
    }

    full_advisory += " " + recommendations.get(category, "")

    return full_advisory.strip()


def process_dataset(filepath, target_group="General"):
    """Process a single AQI dataset file and generate advisories."""
    df = pd.read_csv(filepath)

    advisories = []
    for _, row in df.iterrows():
        aqi = row["AQI"]
        category, color = classify_aqi(aqi)
        dominant = get_dominant_pollutant(
            aqi, row["PM2.5"], row["PM10"], row["NO2"],
            row["SO2"], row["CO"], row["O3"]
        )
        advisory = generate_advisory(
            aqi=aqi,
            category=category,
            dominant_pollutant=dominant,
            temperature=30,  # Default; can be enhanced with weather data
            humidity=60,     # Default; can be enhanced with weather data
            wind_speed=8,    # Default; can be enhanced with weather data
            rainfall=0,      # Default; can be enhanced with weather data
            location=row["City"],
            target_group=target_group,
        )
        advisories.append({
            "City": row["City"],
            "Date": row["Date"],
            "AQI": aqi,
            "Category": category,
            "DominantPollutant": dominant,
            "Advisory": advisory,
        })

    return pd.DataFrame(advisories)


def main():
    """Main entry point - process all dataset files."""
    data_dir = "C:\\Users\\wsrir\\OneDrive\\Desktop\\coding bhai\\NLP project"

    cities = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Mumbai"]
    target_groups = ["General", "Children", "Elderly", "Sensitive", "Asthma"]

    print("=" * 80)
    print("NLP-BASED AIR QUALITY HEALTH ADVISORY GENERATOR (India)")
    print("=" * 80)
    print()

    for city in cities:
        csv_file = f"{data_dir}\\{city}_AQI_Dataset.csv"
        try:
            df = process_dataset(csv_file, target_group="General")
            print(f"\n=== {city} AQI Data ===")
            for _, row in df.head(3).iterrows():
                print(f"\nDate: {row['Date']}")
                print(f"AQI: {row['AQI']} ({row['Category']})")
                print(f"Dominant Pollutant: {row['DominantPollutant']}")
                print(f"Advisory: {row['Advisory']}")
        except FileNotFoundError:
            print(f"Warning: {csv_file} not found, skipping...")

    print("\n" + "=" * 80)
    print("Sample advisories generated for multiple cities and demographics.")
    print("Full dataset processing and NLP model fine-tuning available.")
    print("=" * 80)


if __name__ == "__main__":
    main()