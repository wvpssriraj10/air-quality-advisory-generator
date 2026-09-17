"""NLP-Based Air Quality Health Advisory Generator - Compact Dashboard"""
import streamlit as st

CATEGORY_INFO = {
    'Good': {
        'color': 'Green',
        'impact': 'Minimal health impact; safe for outdoor activity.',
        'advice': 'No restrictions. Enjoy outdoor activities safely.',
    },
    'Satisfactory': {
        'color': 'Light Green',
        'impact': 'May cause minor breathing discomfort to sensitive people.',
        'advice': 'Reduce intense outdoor activity if you have respiratory sensitivities.',
    },
    'Moderately Polluted': {
        'color': 'Yellow',
        'impact': 'Discomfort to people with lung/heart disease, children, and older adults.',
        'advice': 'Reduce prolonged exertion. Keep windows closed if dust is suspected.',
    },
    'Poor': {
        'color': 'Orange',
        'impact': 'Breathing discomfort to most people on prolonged exposure.',
        'advice': 'Avoid outdoor activities. Use N95 masks if going outside is necessary.',
    },
    'Very Poor': {
        'color': 'Red',
        'impact': 'Respiratory illness on prolonged exposure; severe impact on sensitive groups.',
        'advice': 'Stay indoors with windows closed. Use air purifiers if available.',
    },
    'Severe': {
        'color': 'Dark Red',
        'impact': 'Affects healthy people and seriously impacts those with pre-existing conditions.',
        'advice': 'Remain indoors. Use high-efficiency air filters. Seek medical attention if symptoms appear.',
    },
}

GROUP_ADVICE = {
    'Children': 'Children should limit outdoor play, avoid heavy activity, and stay in well-ventilated indoor spaces.',
    'Elderly': 'Elderly individuals should avoid prolonged outdoor exposure and keep medications ready.',
    'Sensitive': 'People with respiratory or heart conditions should stay indoors and use air purifiers if available.',
    'Asthma': 'Asthma patients should carry inhalers, avoid outdoor exercise, and keep windows closed.',
    'General': 'Healthy adults can follow the general recommendation for the current AQI category.',
}

POLLUTANT_NOTES = {
    'PM2.5': 'Fine particles that penetrate deep into lungs and can enter the bloodstream.',
    'PM10': 'Coarse particles that irritate the eyes, nose, throat, and upper airways.',
    'NO2': 'Nitrogen dioxide linked to airway inflammation and worsened asthma symptoms.',
    'SO2': 'Sulfur dioxide that can trigger breathing difficulty, especially in sensitive groups.',
    'CO': 'Carbon monoxide reduces oxygen delivery; high levels are dangerous indoors and outdoors.',
    'O3': 'Ground-level ozone can irritate airways and reduce lung function during outdoor activity.',
}

def classify_aqi(aqi):
    if aqi <= 50: return 'Good'
    elif aqi <= 100: return 'Satisfactory'
    elif aqi <= 200: return 'Moderately Polluted'
    elif aqi <= 300: return 'Poor'
    elif aqi <= 400: return 'Very Poor'
    return 'Severe'

def build_result(aqi, category, pollutant, temp, hum, wind, city, group):
    info = CATEGORY_INFO[category]
    weather_notes = []
    if temp > 35:
        weather_notes.append(f'High temperature ({temp}°C) increases heat stress and respiratory strain.')
    elif temp < 15:
        weather_notes.append(f'Cool temperature ({temp}°C) — still limit outdoor exertion if AQI is elevated.')
    else:
        weather_notes.append(f'Temperature is moderate at {temp}°C.')

    if hum > 70:
        weather_notes.append(f'High humidity ({hum}%) can aggravate breathing discomfort.')
    elif hum < 30:
        weather_notes.append(f'Low humidity ({hum}%) may dry airways; stay hydrated.')
    else:
        weather_notes.append(f'Humidity is moderate at {hum}%.')

    if wind < 5:
        weather_notes.append(f'Low wind ({wind} km/h) can allow pollutants to accumulate near the ground.')
    else:
        weather_notes.append(f'Wind speed of {wind} km/h may help disperse some pollutants.')

    return {
        'city': city,
        'aqi': aqi,
        'category': category,
        'color': info['color'],
        'group': group,
        'pollutant': pollutant,
        'pollutant_note': POLLUTANT_NOTES.get(pollutant, ''),
        'impact': info['impact'],
        'advice': info['advice'],
        'group_advice': GROUP_ADVICE.get(group, GROUP_ADVICE['General']),
        'weather_notes': weather_notes,
        'temp': temp,
        'hum': hum,
        'wind': wind,
    }

st.set_page_config(page_title="AQI Advisory", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    div[data-testid="stSlider"] {
        padding-top: 0.35rem;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] {
        margin-top: 1.6rem;
    }
    div[data-baseweb="select"] > div {
        min-height: 44px !important;
        height: auto !important;
        padding-top: 8px !important;
        padding-bottom: 8px !important;
        display: flex !important;
        align-items: center !important;
    }
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {
        line-height: 1.5 !important;
        overflow: visible !important;
        text-overflow: unset !important;
        white-space: nowrap !important;
    }
    .field-label {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.15rem;
        color: rgba(250, 250, 250, 0.9);
    }
    .result-box {
        background: rgba(33, 131, 89, 0.18);
        border: 1px solid rgba(33, 131, 89, 0.45);
        border-radius: 10px;
        padding: 1rem 1.15rem;
        margin-top: 0.75rem;
        line-height: 1.55;
    }
    .result-box h3 {
        margin: 0 0 0.75rem 0;
        font-size: 1.15rem;
    }
    .result-box h4 {
        margin: 0.9rem 0 0.35rem 0;
        font-size: 0.95rem;
        color: rgba(250, 250, 250, 0.85);
    }
    .result-box p, .result-box li {
        margin: 0.25rem 0;
        font-size: 0.95rem;
    }
    .result-box ul {
        margin: 0.2rem 0 0.2rem 1.1rem;
        padding: 0;
    }
    .meta-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem 1.25rem;
        margin-bottom: 0.35rem;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌫️ Air Quality Health Advisory")

with st.form("f", clear_on_submit=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<p class="field-label">AQI (0–500)</p>', unsafe_allow_html=True)
        aqi = st.slider("AQI", 0, 500, 150, label_visibility="collapsed")
    with c2:
        st.markdown('<p class="field-label">City</p>', unsafe_allow_html=True)
        city = st.selectbox("City", ["Delhi", "Mumbai", "Bangalore", "Chennai", "Hyderabad"], label_visibility="collapsed")
    with c3:
        st.markdown('<p class="field-label">User Group</p>', unsafe_allow_html=True)
        group = st.selectbox("User Group", ["General", "Children", "Elderly", "Sensitive", "Asthma"], label_visibility="collapsed")

    c4, c5, c5b = st.columns(3)
    with c4:
        st.markdown('<p class="field-label">Temperature (°C)</p>', unsafe_allow_html=True)
        temp = st.slider("Temperature", 10, 45, 30, label_visibility="collapsed")
    with c5:
        st.markdown('<p class="field-label">Humidity (%)</p>', unsafe_allow_html=True)
        hum = st.slider("Humidity", 20, 95, 60, label_visibility="collapsed")
    with c5b:
        st.markdown('<p class="field-label">Wind Speed (km/h)</p>', unsafe_allow_html=True)
        wind = st.slider("Wind Speed", 0, 30, 8, label_visibility="collapsed")

    st.markdown('<p class="field-label">Dominant Pollutant</p>', unsafe_allow_html=True)
    dom = st.selectbox("Dominant Pollutant", ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"], label_visibility="collapsed")

    submitted = st.form_submit_button("GO", use_container_width=True)

if submitted:
    cat = classify_aqi(aqi)
    r = build_result(aqi, cat, dom, temp, hum, wind, city, group)
    weather_html = "".join(f"<li>{note}</li>" for note in r['weather_notes'])
    st.markdown(f"""
<div class="result-box">
  <h3>🩺 Health Advisory — {r['city']}</h3>
  <div class="meta-row">
    <span><b>AQI:</b> {r['aqi']}</span>
    <span><b>Category:</b> {r['category']} ({r['color']})</span>
    <span><b>User Group:</b> {r['group']}</span>
  </div>
  <div class="meta-row">
    <span><b>Temperature:</b> {r['temp']}°C</span>
    <span><b>Humidity:</b> {r['hum']}%</span>
    <span><b>Wind:</b> {r['wind']} km/h</span>
  </div>
  <h4>Health Impact</h4>
  <p>{r['impact']}</p>
  <h4>Dominant Pollutant — {r['pollutant']}</h4>
  <p>{r['pollutant_note']}</p>
  <h4>Weather Context</h4>
  <ul>{weather_html}</ul>
  <h4>Advice for {r['group']}</h4>
  <p>{r['group_advice']}</p>
  <h4>Recommended Action</h4>
  <p>{r['advice']}</p>
</div>
""", unsafe_allow_html=True)
