import streamlit as st
import google.generativeai as genai
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
import pandas as pd
import re
import os

# Configuration de l'API Gemini avec la version correcte du modèle
genai.configure(api_key="AIzaSyCwWitJOAQDe8jsogTiPmep5ToOw_Vl-Rk")

# Liste des modèles disponibles pour vérification
def list_available_models():
    try:
        models = genai.list_models()
        available_models = [model.name for model in models]
        return available_models
    except Exception as e:
        return f"Erreur lors de la liste des modèles: {str(e)}"

# Connexion à la base de données SQLite
DB_PATH = "call_center_full_extended.db"
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

def get_schema(db):
    """Récupère les informations de schéma de la base de données."""
    return db.get_table_info()

def format_chat_history(chat_history, max_messages=5):
    """Formate l'historique de chat pour l'inclure dans le prompt."""
    formatted_history = []
    filtered_history = [msg for msg in chat_history 
                       if not (isinstance(msg, AIMessage) and "Bonjour" in msg.content)]
    
    for message in filtered_history[-max_messages:]:
        role = "Utilisateur" if isinstance(message, HumanMessage) else "Assistant"
        formatted_history.append(f"{role}: {message.content}")
    
    return "\n".join(formatted_history)

def execute_sql_query(query):
    """Exécute une requête SQL et gère les exceptions."""
    try:
        # Nettoyage de la requête
        clean_query = re.sub(r'```sql|```', '', query).strip()
        
        # Exécution de la requête
        result = db.run(clean_query)
        
        # Si le résultat est vide, retourner un message approprié
        if not result or result.strip() == "":
            return "Aucun résultat trouvé pour cette requête."
        
        return result
    except Exception as e:
        st.error(f"Erreur lors de l'exécution de la requête SQL: {str(e)}")
        return f"Erreur: {str(e)}"

def get_gemini_response(prompt, max_retries=3):
    """Obtient une réponse du modèle Gemini avec le modèle adapté."""
    # Utilisation du modèle gemini-1.0-pro à la place de gemini-pro
    model_name = "gemini-1.5-pro"  # Version la plus récente
    fallback_models = ["gemini-1.0-pro", "gemini-pro-vision"]  # Modèles de secours
    
    for attempt in range(max_retries):
        try:
            # Essayer d'abord avec le modèle principal
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            st.warning(f"Tentative {attempt+1} avec {model_name} échouée: {e}")
            
            # Si c'est une erreur de modèle non trouvé, essayer avec les modèles de secours
            if "not found" in str(e).lower() and attempt < len(fallback_models):
                try:
                    backup_model = genai.GenerativeModel(fallback_models[attempt])
                    response = backup_model.generate_content(prompt)
                    # Si cela fonctionne, mémoriser ce modèle pour les prochains appels
                    model_name = fallback_models[attempt]
                    return response.text.strip()
                except Exception as backup_error:
                    st.warning(f"Modèle de secours {fallback_models[attempt]} échoué: {backup_error}")
            
            # Si c'est la dernière tentative, renvoyer une erreur
            if attempt == max_retries - 1:
                # Essayer une dernière tentative avec un modèle basique comme fallback final
                try:
                    basic_model = genai.GenerativeModel("text-bison@001")
                    response = basic_model.generate_content(prompt)
                    return response.text.strip()
                except:
                    return f"Erreur: Impossible d'accéder aux modèles Gemini. Vérifiez votre clé API et les modèles disponibles."

def get_sql_prompt(schema, chat_history, question):
    """Crée le prompt pour la génération de requête SQL."""
    template = """
    Vous êtes un expert SQL. Convertissez cette question en requête SQL précise.
    
    Schéma de la base de données:
    {schema}
    
    Historique des conversations:
    {chat_history}
    
    Question utilisateur: {question}
    
    Répondez uniquement avec la requête SQL sans aucune explication ou texte supplémentaire.
    """
    
    return template.format(
        schema=schema,
        chat_history=format_chat_history(chat_history),
        question=question
    )

def get_nl_response_prompt(schema, question, sql_query, sql_result):
    """Crée le prompt pour la génération de réponse en langage naturel."""
    template = """
    Transformer le résultat SQL en réponse claire en français pour l'utilisateur.
    
    Schéma de la base de données:
    {schema}
    
    Question originale: {question}
    
    Requête SQL exécutée:
    {sql_query}
    
    Résultat de la requête:
    {sql_result}
    
    Répondez comme un assistant amical, en français, en expliquant les résultats de manière claire.
    Si les résultats sont vides ou nuls, expliquez pourquoi la requête n'a peut-être pas donné de résultats.
    """
    
    return template.format(
        schema=schema,
        question=question,
        sql_query=sql_query,
        sql_result=sql_result
    )

def display_sql_result_as_table(result):
    """Affiche le résultat SQL sous forme de tableau si possible."""
    try:
        # Tenter de convertir le résultat en DataFrame
        if isinstance(result, str):
            # Analyser le résultat sous forme de chaîne en lignes et colonnes
            lines = result.strip().split('\n')
            if len(lines) > 1:
                header = lines[0].split(',')
                data = [line.split(',') for line in lines[1:]]
                df = pd.DataFrame(data, columns=header)
                return df
        return None
    except Exception:
        return None

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Assistants KPIs et DATA", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Stylisation et UI
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTitle {
        color: #1E3A8A;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# En-tête
st.title("📊 Assistants KPIs et DATA")
st.markdown("Explorez vos données et KPIs du centre d'appels avec des questions en langage naturel.")

# Vérification des modèles disponibles (à exécuter une seule fois)
if 'available_models_checked' not in st.session_state:
    available_models = list_available_models()
    st.session_state.available_models = available_models
    st.session_state.available_models_checked = True

# Barre latérale avec informations sur le schéma
with st.sidebar:
    st.header("Informations sur la base de données")
    
    # Récupération et affichage du schéma
    schema_info = get_schema(db)
    with st.expander("Schéma de la base de données", expanded=False):
        st.code(schema_info, language="sql")
    
    # Options avancées
    st.subheader("Options")
    show_sql = st.checkbox("Afficher les requêtes SQL", value=True)
    show_results_as_table = st.checkbox("Afficher les résultats sous forme de tableau", value=True)
    
    # Afficher les modèles disponibles
    with st.expander("Modèles Gemini disponibles", expanded=False):
        if isinstance(st.session_state.available_models, list):
            for model in st.session_state.available_models:
                st.write(f"- {model}")
        else:
            st.write(st.session_state.available_models)
    
    # À propos
    st.markdown("---")
    st.markdown("### À propos")
    st.markdown("Cet assistant vous permet d'interroger votre base de données via des questions en langage naturel pour analyser vos KPIs et données.")

# Initialisation de l'historique
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage("Bonjour ! Je suis votre assistant KPIs et DATA. Comment puis-je vous aider à analyser vos données aujourd'hui ?"),
    ]

# Affichage de l'historique du chat
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for message in st.session_state.chat_history:
        with st.chat_message("ai" if isinstance(message, AIMessage) else "human"):
            st.markdown(message.content)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Input utilisateur
user_query = st.chat_input("Posez votre question sur les KPIs ou les données...")

# Traitement de la requête
if user_query:
    # Ajouter la question de l'utilisateur à l'historique
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    # Afficher la question
    with st.chat_message("human"):
        st.markdown(user_query)
    
    # Traiter la question et afficher la réponse
    with st.chat_message("ai"):
        with st.spinner("Analyse de votre question..."):
            # 1. Récupérer le schéma
            schema = get_schema(db)
            
            # 2. Générer la requête SQL
            sql_prompt = get_sql_prompt(schema, st.session_state.chat_history, user_query)
            sql_query = get_gemini_response(sql_prompt)
            
            if sql_query and not sql_query.startswith("Erreur:"):
                # Afficher la requête SQL si demandé
                if show_sql:
                    st.markdown("**Requête SQL générée:**")
                    st.code(sql_query, language="sql")
                
                # 3. Exécuter la requête
                sql_result = execute_sql_query(sql_query)
                
                # Essayer d'afficher les résultats sous forme de tableau
                if show_results_as_table:
                    df = display_sql_result_as_table(sql_result)
                    if df is not None and not df.empty:
                        st.markdown("**Résultats:**")
                        st.dataframe(df, use_container_width=True)
                
                # 4. Générer la réponse en langage naturel
                nl_prompt = get_nl_response_prompt(schema, user_query, sql_query, sql_result)
                response = get_gemini_response(nl_prompt)
                
                # 5. Afficher la réponse
                st.markdown("**Analyse:**")
                st.markdown(response)
                
                # 6. Ajouter la réponse à l'historique
                st.session_state.chat_history.append(AIMessage(content=response))
            else:
                error_msg = "Je n'ai pas pu générer une requête SQL valide pour votre question. Pourriez-vous reformuler votre question ou fournir plus de détails ?"
                if sql_query.startswith("Erreur:"):
                    error_msg = sql_query
                st.markdown(error_msg)
                st.session_state.chat_history.append(AIMessage(content=error_msg))
