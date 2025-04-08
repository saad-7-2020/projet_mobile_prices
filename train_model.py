# Scripte d' entrainement du model (model avec regression logistique)

import pandas as pd
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib


# Chargement de DATA
df = pd.read_csv("mobile_prices.csv")


# Séparer les caractéristique et la variable cible
carcteristiques = df.drop("price_range", axis=1)
variable_cible = df["price_range"]



# Lors de la préparation du model, j' ai devisé les données en données d' entrainement (80%) et données de test (20%)
# X_train, X_test, y_train, y_test = train_test_split(carcteristiques, variable_cible, train_size= 0.8)


# Normalisation
scaler = RobustScaler()
carcteristiques_normalizees = scaler.fit_transform(carcteristiques)
# X_train = scaler.fit_trnsform(Xtrain) (Lors de la préparation)
# X_test = scaler.transform(X_test) (Lors de la préparation)


# Entraînement du modèle
model = LogisticRegression()
model.fit(carcteristiques_normalizees, variable_cible)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# le model donne des valeurs entre 0.95 et 0.9775 ce qui montre que le model est bien ajusté


# Sauvegarde du modèle et du scaler pour éviter le reentrainememnt du model
joblib.dump(model, "modele.pkl")
joblib.dump(scaler, "scaler.pkl")

