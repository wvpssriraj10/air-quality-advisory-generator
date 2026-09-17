"""NLP-Based Air Quality Health Advisory Generator - Compact Dashboard"""
import streamlit as st
import numpy as np

# CPCB AQI Classification
def classify_aqi(aqi):
    if aqi <= 50: return 'Good'
    elif aqi <= 100: return 'Satisfactory'
    elif aqi <= 200: return 'Moderately Polluted'
    elif aqi <= 300: return 'Poor'
    elif aqi <= 400: return 'Very Poor'
    return 'Severe'

def generate_advisory(aqi, category, pollutant, temp, hum, wind, city, group):
    bases = {
        'Good': f"Air quality in {city} is Good ({aqi} AQI). Safe for outdoor activities.",
        'Satisfactory': f"Air quality in {city} is Satisfactory ({aqi} AQI). Minor discomfort to sensitive.",
        'Moderately Polluted': f"Air quality in {city} is Moderately Polluted ({aqi} AQI). Discomfort to lung/heart patients.",
        'Poor': f"Air quality in {city} is Poor ({aqi} AQI). Breathing discomfort on exposure.",
        'Very Poor': f"Air quality in {city} is Very Poor ({aqi} AQI). Respiratory illness risk.",
        'Severe': f"Air quality in {city} is Severe ({aqi} AQI). Affects healthy and seriously impacts pre-existing conditions."
    }
    adv = bases.get(category, "")
    adv += f" Pollutant: {pollutant}."
    if temp > 35: adv += f" Heat ({temp}°C) increases stress."
    if hum > 70: adv += f" Humidity ({hum}%) aggravates conditions."
    if wind and wind < 5: adv += f" Low wind ({wind} km/h) causes accumulation."
    demo = {'Children': 'Children: limit outdoor', 'Elderly': 'Elderly: avoid exposure', 'Sensitive': 'Sensitive: stay indoors', 'Asthma': 'Asthma: carry inhalers', 'General': ''}
    adv += f" {demo.get(group, '')}."
    rec = {'Good': 'No restrictions.', 'Satisfactory': 'Reduce intense activity.', 'Moderately Polluted': 'Reduce exertion.', 'Poor': 'Avoid outdoors.', 'Very Poor': 'Stay indoors.', 'Severe': 'Remain indoors.'}
    adv += f" {rec.get(category, '')}"
    return adv.strip()

st.set_page_config(page_title="AQI Advisory", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stSelectbox [data-baseweb="select"] > div {
        min-height: 38px;
        padding-top: 4px;
        padding-bottom: 4px;
    }
    .stSelectbox [data-baseweb="select"] span {
        line-height: 1.4;
        overflow: visible;
    }
    .stSuccess { font-size: 100%; padding: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🌫️ Air Quality Health Advisory")

with st.form("f", clear_on_submit=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        aqi = st.slider("AQI (0–500)", 0, 500, 150)
    with c2:
        city = st.selectbox("City", ["Delhi", "Mumbai", "Bangalore", "Chennai", "Hyderabad"])
    with c3:
        group = st.selectbox("User Group", ["General", "Children", "Elderly", "Sensitive", "Asthma"])

    c4, c5, c5b = st.columns(3)
    with c4:
        temp = st.slider("Temperature (°C)", 10, 45, 30)
    with c5:
        hum = st.slider("Humidity (%)", 20, 95, 60)
    with c5b:
        wind = st.slider("Wind Speed (km/h)", 0, 30, 8)

    dom = st.selectbox("Dominant Pollutant", ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"])

    submitted = st.form_submit_button("GO", use_container_width=True)

if submitted:
    cat = classify_aqi(aqi)
    adv = generate_advisory(aqi, cat, dom, temp, hum, wind, city, group)
    st.success(adv, icon="🩺")

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
