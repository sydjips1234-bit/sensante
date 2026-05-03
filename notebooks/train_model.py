"""
Lab 2 : Entraînement et sérialisation du modèle SénSanté
Tag Git : v1
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================================
# Étape 1 : Charger les données (avec encodage UTF-8 corrigé)
# ============================================================
df = pd.read_csv("data/patients_dakar.csv", encoding="utf-8")
# Si les accents restent bizarres, remplacez la ligne ci-dessus par :
# df = pd.read_csv("data/patients_dakar.csv", encoding="latin-1")

print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"Colonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

# ============================================================
# Étape 2 : Encodage des variables catégorielles
# ============================================================
le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys',
                'toux', 'fatigue', 'maux_tete', 'region_encoded']

X = df[feature_cols]
y = df['diagnostic']

print(f"Features shape : {X.shape}")
print(f"Cible shape    : {y.shape}")

# ============================================================
# Étape 3 : Séparer les données (train / test)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nEntraînement : {X_train.shape[0]} patients")
print(f"Test         : {X_test.shape[0]} patients")

# ============================================================
# Étape 4 : Entraîner un RandomForest
# ============================================================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\nModèle entraîné !")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Classes : {list(model.classes_)}")

# ============================================================
# Étape 5 : Évaluer le modèle
# ============================================================
y_pred = model.predict(X_test)

# Comparaison 10 premiers
comparison = pd.DataFrame({
    'Vrai diagnostic': y_test.values[:10],
    'Prédiction': y_pred[:10]
})
print("\nComparaison (10 premiers) :")
print(comparison)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {accuracy:.2%}")

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("\nMatrice de confusion :")
print(cm)

# Rapport détaillé
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# ============================================================
# Étape 5 bis : Visualisation (matrice de confusion graphique)
# ============================================================
os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Prédiction du modèle')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion - SénSanté')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150)
plt.show()   # commenter si pas d'affichage graphique

# ============================================================
# Étape 6 : Sérialiser le modèle + encodeurs + metadata
# ============================================================
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")

print("\nModèle sauvegardé : models/model.pkl")
print(f"Taille : {os.path.getsize('models/model.pkl') / 1024:.1f} Ko")
print("Encodeurs et métadonnées sauvegardés.")

# ============================================================
# Étape 7 : Test du modèle rechargé (simulation API)
# ============================================================
model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': True,
    'fatigue': True,
    'maux_tete': True,
    'region': 'Dakar'
}

sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

features = [
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']),
    int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']),
    region_enc
]

prediction = model_loaded.predict([features])[0]
probas = model_loaded.predict_proba([features])[0]

print(f"\n--- Nouveau patient ---")
print(f"Diagnostic prédit : {prediction}")
print("Probabilités :")
for classe, proba in zip(model_loaded.classes_, probas):
    print(f"  {classe}: {proba:.2%}")

# ============================================================
# Exercice 1 : Importance des features
# ============================================================
importances = model.feature_importances_
print("\n--- Importance des features ---")
for name, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
    print(f"  {name:20s} : {imp:.3f}")

# ============================================================
# Exercice 2 : Tester avec 3 patients (profils différents)
# ============================================================
print("\n--- Tests avec 3 patients (exercice 2) ---")
patients_test = [
    {'age': 10, 'sexe': 'M', 'temperature': 37.0, 'tension_sys': 100,
     'toux': False, 'fatigue': False, 'maux_tete': False, 'region': 'Thiès'},
    {'age': 45, 'sexe': 'F', 'temperature': 40.2, 'tension_sys': 120,
     'toux': True, 'fatigue': True, 'maux_tete': True, 'region': 'Dakar'},
    {'age': 70, 'sexe': 'M', 'temperature': 38.0, 'tension_sys': 130,
     'toux': True, 'fatigue': False, 'maux_tete': False, 'region': 'Ziguinchor'}
]

for i, patient in enumerate(patients_test, 1):
    sexe_enc = le_sexe_loaded.transform([patient['sexe']])[0]
    region_enc = le_region_loaded.transform([patient['region']])[0]
    feat = [patient['age'], sexe_enc, patient['temperature'],
            patient['tension_sys'], int(patient['toux']),
            int(patient['fatigue']), int(patient['maux_tete']), region_enc]
    pred = model_loaded.predict([feat])[0]
    proba = model_loaded.predict_proba([feat])[0]
    print(f"Patient {i} -> {pred} (confiance {proba.max():.2%})")

print("\n=== Fin du script ===")