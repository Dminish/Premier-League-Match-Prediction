import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Page Config (Must be the first Streamlit command)
st.set_page_config(page_title="Match Predictor by Danish", page_icon="⚽", layout="centered")

# --- 1. Custom CSS for Stadium Background & Glassmorphism ---
st.markdown("""
<style>
    /* Cinematic Stadium Background */
    [data-testid="stAppViewContainer"] {
        background: url("https://images.unsplash.com/photo-1489944440615-453fc2b6a9a9?q=80&w=2000&auto=format&fit=crop") no-repeat center center fixed;
        background-size: cover;
    }
    
    /* Hide the top header bar so it looks cleaner */
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    
    /* Create a frosted glass floating container for the app */
    .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.6);
        max-width: 800px;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

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
    
    .title-text {
        text-align: center;
        color: #38003c;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        margin-bottom: 0px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Team Logo Dictionary ---
TEAM_LOGOS = {
    "Arsenal": "https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg",
    "Aston Villa": "https://upload.wikimedia.org/wikipedia/en/9/9f/Aston_Villa_logo.svg",
    "Bournemouth": "https://upload.wikimedia.org/wikipedia/en/e/e5/AFC_Bournemouth_%282013%29.svg",
    "Brentford": "https://upload.wikimedia.org/wikipedia/en/2/2a/Brentford_FC_crest.svg",
    "Brighton": "https://upload.wikimedia.org/wikipedia/en/f/fd/Brighton_%26_Hove_Albion_logo.svg",
    "Burnley": "https://upload.wikimedia.org/wikipedia/en/6/62/Burnley_F.C._Logo.svg",
    "Chelsea": "https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg",
    "Crystal Palace": "https://upload.wikimedia.org/wikipedia/en/a/a2/Crystal_Palace_FC_logo_%282022%29.svg",
    "Everton": "https://upload.wikimedia.org/wikipedia/en/7/7c/Everton_FC_logo.svg",
    "Fulham": "https://upload.wikimedia.org/wikipedia/en/e/eb/Fulham_FC_%28shield%29.svg",
    "Liverpool": "https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg",
    "Luton Town": "https://upload.wikimedia.org/wikipedia/en/9/9d/Luton_Town_logo.svg",
    "Man City": "https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg",
    "Man United": "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg",
    "Newcastle": "https://upload.wikimedia.org/wikipedia/en/5/56/Newcastle_United_Logo.svg",
    "Nott'm Forest": "https://upload.wikimedia.org/wikipedia/en/e/e5/Nottingham_Forest_F.C._logo.svg",
    "Sheffield United": "https://upload.wikimedia.org/wikipedia/en/9/9c/Sheffield_United_FC_logo.svg",
    "Tottenham": "https://upload.wikimedia.org/wikipedia/en/b/b4/Tottenham_Hotspur.svg",
    "West Ham": "https://upload.wikimedia.org/wikipedia/en/c/c2/West_Ham_United_FC_logo.svg",
    "Wolves": "https://upload.wikimedia.org/wikipedia/en/f/fc/Wolverhampton_Wanderers_crest.svg"
}

def get_logo(team_name):
    # Returns the team logo, or a generic Premier League logo if the team name doesn't perfectly match
    return TEAM_LOGOS.get(team_name, "https://upload.wikimedia.org/wikipedia/en/e/e2/English_Premier_League_logo.svg")


# --- 3. Load the Exported Artifacts ---
@st.cache_resource 
def load_artifacts():
    model = tf.keras.models.load_model('models/football_model.keras')
    scaler = joblib.load('models/scaler.pkl')
    encoder = joblib.load('models/encoder.pkl')
    elo_dict = joblib.load('models/elo_dict.pkl')
    return model, scaler, encoder, elo_dict

model, scaler, encoder, elo_dict = load_artifacts()

# --- 4. Dashboard UI ---
st.markdown("<h1 class='title-text'>⚽ Match Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px; color: #e90052; font-weight: bold;'>Powered by Deep Learning & Elo Ratings</p>", unsafe_allow_html=True)
st.divider()

# Set Man United as default if available
teams = list(encoder.classes_)
default_home = teams.index("Man United") if "Man United" in teams else 0
default_away = teams.index("Liverpool") if "Liverpool" in teams else 1

# Display Logos Head-to-Head
logo_col1, logo_col2, logo_col3 = st.columns([1, 0.2, 1])
home_team = st.selectbox("🏠 Home Team", teams, index=default_home)
away_team = st.selectbox("✈️ Away Team", teams, index=default_away)

with logo_col1:
    st.markdown(f"<div style='text-align: center;'><img src='{get_logo(home_team)}' width='120'></div>", unsafe_allow_html=True)
with logo_col2:
    st.markdown("<h2 style='text-align: center; color: gray; margin-top: 30px;'>VS</h2>", unsafe_allow_html=True)
with logo_col3:
    st.markdown(f"<div style='text-align: center;'><img src='{get_logo(away_team)}' width='120'></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
home_goals = st.slider(f"{home_team}'s Recent Avg Goals", min_value=0.0, max_value=5.0, value=1.8, step=0.1)
st.markdown("<br>", unsafe_allow_html=True)

# --- 5. Inference & Visualization ---
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
