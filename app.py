import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Page Config
st.set_page_config(page_title="Match Predictor", page_icon="⚽", layout="centered")

# --- 1. Load the Exported Artifacts ---
@st.cache_resource # Caches the model so it doesn't reload on every click
def load_artifacts():
    model = tf.keras.models.load_model('models/football_model.keras')
    scaler = joblib.load('models/scaler.pkl')
    encoder = joblib.load('models/encoder.pkl')
    elo_dict = joblib.load('models/elo_dict.pkl')
    return model, scaler, encoder, elo_dict

model, scaler, encoder, elo_dict = load_artifacts()

# --- 2. Dashboard UI ---
st.title("⚽ Premier League Match Predictor")
st.markdown("Powered by Deep Learning & Elo Ratings")

# Layout with two columns
col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox("Select Home Team", encoder.classes_)
    
with col2:
    away_team = st.selectbox("Select Away Team", encoder.classes_)

# Slider for recent form
home_goals = st.slider("Home Team's Recent Avg Goals", min_value=0.0, max_value=5.0, value=1.5, step=0.1)

# --- 3. Inference & Visualization ---
if st.button("Predict Outcome"):
    if home_team == away_team:
        st.error("Home and Away teams must be different!")
    else:
        # Fetch Elo ratings
        home_elo = elo_dict.get(home_team, 1500.0)
        away_elo = elo_dict.get(away_team, 1500.0)
        
        # Prepare inputs
        home_id = encoder.transform([home_team])[0]
        away_id = encoder.transform([away_team])[0]
        
        # Scale and Predict
        features = np.array([[home_id, away_id, home_goals, home_elo, away_elo]])
        features_scaled = scaler.transform(features)
        probs = model.predict(features_scaled)[0]
        
        # Display Results
        st.subheader("Match Prediction Probabilities")
        
        # Create a dataframe for the bar chart
        chart_data = pd.DataFrame(
            {"Probability": [probs[2], probs[1], probs[0]]},
            index=[f"{home_team} Win", "Draw", f"{away_team} Win"]
        )
        
        # Render a sleek bar chart
        st.bar_chart(chart_data)
        
        # Display actual Elo ratings
        st.caption(f"Current Elo Ratings — {home_team}: {home_elo:.0f} | {away_team}: {away_elo:.0f}")