# SenSante API - Assistant pre-diagnostic medical
# Lab 3 - Integration de Modeles IA - ESP / UCAD

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import os


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)


# ----------------------------
# Schemas Pydantic
# ----------------------------

class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sexe: str = Field(...)
    temperature: float = Field(..., ge=35.0, le=42.0)
    tension_sys: int = Field(..., ge=60, le=250)
    toux: bool = Field(...)
    fatigue: bool = Field(...)
    maux_tete: bool = Field(...)
    region: str = Field(...)


class DiagnosticOutput(BaseModel):
    diagnostic: str
    probabilite: float
    confiance: str
    message: str

class ExplainInput(BaseModel):
    diagnostic: str
    probabilite: float
    confiance: str

# ----------------------------
# Application FastAPI
# ----------------------------

app = FastAPI(
    title="SenSante API",
    description="Assistant pre-diagnostic medical pour le Senegal",
    version="0.3.0"
)

groq_api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=groq_api_key)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Chargement du modele
# ----------------------------

print("Chargement du modele...")

model = joblib.load("models/model.pkl")
le_sexe = joblib.load("models/encoder_sexe.pkl")
le_region = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

print(f"Modele charge : {list(model.classes_)}")


# ----------------------------
# Route Health
# ----------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "SenSante API is running"
    }


# ----------------------------
# Exercice 1 - Model Info
# ----------------------------

@app.get("/model-info")
def model_info():

    return {
        "type_modele": type(model).__name__,
        "nombre_arbres": len(model.estimators_),
        "classes": list(model.classes_),
        "nombre_features": len(feature_cols)
    }


# ----------------------------
# Route Predict
# ----------------------------

@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):

    # Encoder sexe
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Sexe invalide : {patient.sexe}. Utiliser M ou F."
        )

    # Encoder region
    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Region inconnue : {patient.region}"
        )

    # Construire les features
    features = np.array([[
        patient.age,
        sexe_enc,
        patient.temperature,
        patient.tension_sys,
        int(patient.toux),
        int(patient.fatigue),
        int(patient.maux_tete),
        region_enc
    ]])

    # Prediction
    diagnostic = model.predict(features)[0]

    # Probabilites
    probas = model.predict_proba(features)[0]
    proba_max = float(probas.max())

    # Niveau de confiance
    if proba_max >= 0.7:
        confiance = "haute"
    elif proba_max >= 0.4:
        confiance = "moyenne"
    else:
        confiance = "faible"

    # Messages selon diagnostic
    messages = {
        "palu": "Suspicion de paludisme. Consultez un medecin rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation recommandes.",
        "typh": "Suspicion de typhoide. Consultation medicale necessaire.",
        "sain": "Pas de pathologie detectee. Continuez a surveiller."
    }

    # Génération du conseil IA
    conseil_ia = generer_conseil_ia(
        diagnostic,
        confiance
)

    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=conseil_ia
)

@app.post("/explain")
def explain(data: ExplainInput):

    prompt = f"""
    Tu es un assistant médical sénégalais bienveillant.

    Explique simplement ce diagnostic à un patient.

    Diagnostic : {data.diagnostic}
    Probabilité : {data.probabilite}
    Niveau de confiance : {data.confiance}

    Donne :
    - une explication simple
    - les symptômes possibles
    - des conseils pratiques
    - quand consulter un médecin

    Réponse courte, claire et rassurante.
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.5,
        max_tokens=250
    )

    return {
        "explication": response.choices[0].message.content
    }

    try:

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un assistant médical sénégalais."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.5,
            max_tokens=200
        )

        explication = response.choices[0].message.content

        return {
            "explication": explication
        }

    except Exception as e:

        return {
            "explication": "Explication non disponible."
        }