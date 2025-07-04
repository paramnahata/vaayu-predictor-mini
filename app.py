import streamlit as st
import joblib
import pandas as pd
import folium
from streamlit_folium import st_folium

# Load ML model
model = joblib.load('model.pkl')

st.title("🌫️ PM2.5 UI Test - VaayuVigyaan")

# Inputs
aod = st.slider("AOD (Aerosol Optical Depth)", 0.1, 1.0, 0.3)
temp = st.number_input("Temperature (°C)", value=30)
humidity = st.slider("Humidity (%)", 0, 100, 60)

# Flag to trigger map
show_map = False
prediction = None

# Predict
if st.button("Predict PM2.5"):
    input_df = pd.DataFrame([[aod, temp, humidity]], columns=['aod', 'temp', 'humidity'])
    prediction = model.predict(input_df)[0]
    st.success(f"🎯 Predicted PM2.5: {prediction:.2f} µg/m³")
    show_map = True

    # Health Risk Bar
    if prediction < 50:
        st.markdown("🟢 **Low Risk** – Air is Good 🙂")
    elif prediction < 100:
        st.markdown("🟡 **Moderate Risk** – Acceptable, but sensitive groups be cautious 😷")
    elif prediction < 200:
        st.markdown("🟠 **Unhealthy for Sensitive Groups** – Reduce outdoor activity 😷")
    else:
        st.markdown("🔴 **High Risk** – Avoid going outside ❌🏃")

    # 🔽 City Comparison Table
    st.subheader("📊 PM2.5 Comparison with Nearby Cities")
    data = {
        "City": ["Your City", "Pune", "Mumbai", "Nagpur"],
        "PM2.5": [round(prediction, 2), 120, 90, 150]
    }
    df = pd.DataFrame(data)
    st.table(df)


# ✅ Map always below to avoid disappearing
if show_map and prediction is not None:
    st.subheader("🗺️ Your Location on Pollution Map")
    m = folium.Map(location=[19.8762, 75.3433], zoom_start=6)
    folium.Marker(
        location=[19.8762, 75.3433],
        popup=f"PM2.5: {prediction:.2f}",
        tooltip="Your Location",
        icon=folium.Icon(color='red' if prediction > 150 else 'orange' if prediction > 100 else 'green')
    ).add_to(m)
    st_folium(m, width=700, height=500)
