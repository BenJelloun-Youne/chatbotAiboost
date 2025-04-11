# Assistant KPIs et DATA

Un chatbot intelligent pour analyser les données d'un centre d'appels en langage naturel.

## Fonctionnalités

- Analyse des performances des agents
- Suivi des équipes
- Gestion des bonus et récompenses
- Suivi de la satisfaction client
- Interface conversationnelle intuitive

## Installation

1. Cloner le repository :
```bash
git clone [URL_DU_REPO]
cd chatbotAiboost
```

2. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

4. Configurer les variables d'environnement :
- Créer un fichier `.env` à la racine du projet
- Ajouter votre clé API OpenAI :
```
OPENAI_API_KEY=votre_clé_api
```

## Utilisation

1. Lancer l'application :
```bash
streamlit run app.py
```

2. Accéder à l'interface web :
- Ouvrir votre navigateur à l'adresse : http://localhost:8501

## Exemples de questions

- "Combien d'agents avons-nous ?"
- "Quels sont nos meilleurs agents ?"
- "Comment performent nos équipes ?"
- "Qui a reçu le plus de bonus ?"
- "Quelle est la satisfaction client ?"

## Technologies utilisées

- Python
- Streamlit
- OpenAI API
- SQLite
- LangChain

## Licence

MIT
