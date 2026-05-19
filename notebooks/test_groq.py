from groq import Groq
from dotenv import load_dotenv
import os

# Charger .env
load_dotenv()

# Lire la clé API
api_key = os.getenv("GROQ_API_KEY")

# Créer le client Groq
client = Groq(api_key=api_key)

# Appel au modèle
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "Tu es un assistant médical sénégalais."
        },
        {
            "role": "user",
            "content": "Explique simplement ce qu'est le paludisme."
        }
    ],
    temperature=0.5,
    max_tokens=200
)

# Afficher réponse
print(response.choices[0].message.content)