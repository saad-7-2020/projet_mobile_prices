# Script de l' interface de l'application web avec streamlit

import streamlit as st
import numpy as np
import joblib

# Chargement du modèle et du scaler
model = joblib.load("modele.pkl")
scaler = joblib.load("scaler.pkl")

# Inerface de l' application web
st.title("Prédiction du prix d\'un téléphone") # Titre de la page web
st.subheader("Realisé par Ait Lhadj Lhacen Saad") # Subheader

# Demende des caractéristiques du téléphone
with st.form("formulaire caractéristiques"):

    col1, col2 = st.columns(2)

    with col1:
        battery_power = st.number_input("Battery Power", step= 1)
        clock_speed = st.number_input("Clock Speed")
        fc = st.number_input("Front Camera (MP)", step= 1)
        int_memory = st.number_input("Mémoire Interne (Go)", step= 1)
        mobile_wt = st.number_input("Poids du téléphone (g)", step= 1)
        pc = st.number_input("Camera Principale (MP)", step= 1)
        px_width = st.number_input("Largeur écran (px)", step= 1)
        ram = st.number_input("RAM (Mo)", step=1)
        sc_h = st.number_input("Hauteur écran (cm)", step= 1)
        talk_time = st.number_input("Talk time (h)")

    with col2:
        blue = st.selectbox("Bluetooth", [0, 1])
        dual_sim = st.selectbox("Dual SIM", [0, 1])
        four_g = st.selectbox("4G", [0, 1])
        m_dep = st.number_input("Mobile Depth (cm)")
        n_cores = st.number_input("Nombre de cœurs CPU", step=1)
        px_height = st.number_input("Hauteur écran (px)", step=1)
        sc_w = st.number_input("Largeur écran (cm)", step=1)
        three_g = st.selectbox("3G", [0, 1])
        touch_screen = st.selectbox("Touch Screen", [0, 1])
        wifi = st.selectbox("WiFi", [0, 1])

    submit_button = st.form_submit_button("Submit")
    
# Après le click sur le button "Submit"
if submit_button:
    caracteristiques = np.array([[battery_power, blue, clock_speed, dual_sim, fc, four_g,
                            int_memory, m_dep, mobile_wt, n_cores, pc, px_height, px_width,
                            ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi]])

    caracteristiques_normalizees = scaler.transform(caracteristiques)
    prediction = model.predict(caracteristiques_normalizees)[0]

# Message a affiché indique la classe du téléphone
    st.success(f"Le modèle prédit que le téléphone appartient à la classe : **{prediction}**")
