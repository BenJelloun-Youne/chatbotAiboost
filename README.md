# Assistants KPIs et DATA

Une application Streamlit permettant d'analyser les KPIs et données d'un centre d'appels en langage naturel.

## Fonctionnalités

- Interrogation de base de données via questions en langage naturel
- Génération automatique de requêtes SQL
- Analyse et interprétation des résultats en français
- Visualisation des données sous forme de tableaux
- Interface utilisateur conviviale et intuitive

## Déploiement sur Streamlit Cloud

1. Créez un nouveau repository GitHub
2. Ajoutez tous les fichiers de ce projet au repository
3. Connectez-vous à [Streamlit Cloud](https://streamlit.io/cloud)
4. Déployez l'application en sélectionnant votre repository GitHub

## Structure du projet

```
├── app.py                   # Application principale Streamlit
├── requirements.txt         # Dépendances Python
├── call_center_full_extended.db   # Base de données SQLite
├── .streamlit/
│   └── config.toml          # Configuration Streamlit
└── README.md                # Documentation
```

## Développement local

1. Clonez ce repository
2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
3. Lancez l'application :
   ```bash
   streamlit run app.py
   ```

## Personnalisation

- Modifiez le fichier `app.py` pour adapter l'application à vos besoins
- Remplacez la base de données par la vôtre (ajustez les chemins et le schéma)
- Personnalisez l'interface utilisateur via le fichier `.streamlit/config.toml`
