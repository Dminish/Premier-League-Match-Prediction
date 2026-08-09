import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Page Config
st.set_page_config(page_title="Match Predictor", page_icon="⚽", layout="centered")

# --- 1. Custom CSS for Premium Styling ---
st.markdown("""
<style>
    /* Style the Predict button with Premier League colors */
    div.stButton > button:first-child {
        background-color: #38003c;
        color: white;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        width: 100%;
        border: none;
        padding: 12px;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #e90052;
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Clean up the header font */
    .title-text {
        text-align: center;
        color: #38003c;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Load the Exported Artifacts ---
@st.cache_resource 
def load_artifacts():
    model = tf.keras.models.load_model('models/football_model.keras')
    scaler = joblib.load('models/scaler.pkl')
    encoder = joblib.load('models/encoder.pkl')
    elo_dict = joblib.load('models/elo_dict.pkl')
    return model, scaler, encoder, elo_dict

model, scaler, encoder, elo_dict = load_artifacts()

# --- 3. Dashboard UI ---
st.markdown("<h1 class='title-text'>⚽ Premier League AI Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: gray;'>Powered by Deep Learning & Dynamic Elo Ratings</p>", unsafe_allow_html=True)
st.divider()

# Set Man United as default if available
teams = list(encoder.classes_)
default_home = teams.index("Man United") if "Man United" in teams else 0
default_away = teams.index("Arsenal") if "Arsenal" in teams else 1

col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox("🏠 Home Team", teams, index=default_home)
    home_goals = st.slider(f"{home_team}'s Recent Avg Goals", min_value=0.0, max_value=5.0, value=1.8, step=0.1)
    
with col2:
    away_team = st.selectbox("✈️ Away Team", teams, index=default_away)

st.markdown("<br>", unsafe_allow_html=True)

# --- 4. Inference & Visualization ---
if st.button("🔮 Predict Match Outcome"):
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
        
        st.divider()
        st.markdown("<h3 style='text-align: center;'>Head-to-Head Analytics</h3>", unsafe_allow_html=True)
        
        # Display Elo Ratings using sleek Metric Cards
        elo_col1, elo_col2 = st.columns(2)
        elo_col1.metric(label=f"📈 {home_team} Elo Rating", value=f"{home_elo:.0f}")
        elo_col2.metric(label=f"📈 {away_team} Elo Rating", value=f"{away_elo:.0f}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Match Prediction Probabilities")
        
        # Display animated progress bars for probabilities
        st.write(f"🏠 **{home_team} Win:** {probs[2]:.1%}")
        st.progress(float(probs[2]))
        
        st.write(f"🤝 **Draw:** {probs[1]:.1%}")
        st.progress(float(probs[1]))
        
        st.write(f"✈️ **{away_team} Win:** {probs[0]:.1%}")
        st.progress(float(probs[0]))
