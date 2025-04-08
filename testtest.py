import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Lire les données
df = pd.read_csv("mobile_prices.csv")

# Séparer les features et la target
X = df.drop("price_range", axis=1)
y = df["price_range"]

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entraînement du modèle
model = LogisticRegression()
model.fit(X_train, y_train)

# Interface Streamlit
st.title("Prédiction de la classe ' un téléphone'")

with st.form("Formulaire"):
    battery_power = st.number_input("Battery Power")
    blue = st.selectbox("Bluetooth", [0, 1])
    clock_speed = st.number_input("Clock Speed")
    dual_sim = st.selectbox("Dual SIM", [0, 1])
    fc = st.number_input("Front Camera (MP)")
    four_g = st.selectbox("4G", [0, 1])
    int_memory = st.number_input("Mémoire Interne (Go)")
    m_dep = st.number_input("Mobile Depth (cm)")
    mobile_wt = st.number_input("Poids du téléphone (g)")
    n_cores = st.number_input("Nombre de cœurs CPU")
    pc = st.number_input("Camera Principale (MP)")
    px_height = st.number_input("Hauteur écran (px)")
    px_width = st.number_input("Largeur écran (px)")
    ram = st.number_input("RAM (Mo)")
    sc_h = st.number_input("Hauteur écran (cm)")
    sc_w = st.number_input("Largeur écran (cm)")
    talk_time = st.number_input("Autonomie en communication (h)")
    three_g = st.selectbox("3G", [0, 1])
    touch_screen = st.selectbox("Écran tactile", [0, 1])
    wifi = st.selectbox("WiFi", [0, 1])

    submit = st.form_submit_button("Prédire")

# Traitement de la prédiction
if submit:
    user_input = np.array([[battery_power, blue, clock_speed, dual_sim, fc, four_g,
                            int_memory, m_dep, mobile_wt, n_cores, pc, px_height, px_width,
                            ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi]])
    
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)[0]
    
    st.success(f"Le modèle prédit que le téléphone appartient à la classe de prix : **{prediction}**")

    # Optionnel : affichage des performances du modèle
    y_pred_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    st.write(f"**Précision du modèle sur les données de test** : {accuracy * 100:.2f}%")
