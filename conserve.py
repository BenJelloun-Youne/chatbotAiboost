import streamlit as st
import google.generativeai as genai
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage

# Configuration de l'API Gemini
genai.configure(api_key="AIzaSyCwWitJOAQDe8jsogTiPmep5ToOw_Vl-Rk")

# Connexion automatique à la base de données SQLite
DB_PATH = "call_center_full_extended.db"
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

def get_schema(db):
    return db.get_table_info()

def format_chat_history(chat_history):
    formatted_history = []
    for message in chat_history:
        if isinstance(message, AIMessage) and "Bonjour" in message.content:
            continue
        role = "Utilisateur" if isinstance(message, HumanMessage) else "Assistant"
        formatted_history.append(f"{role}: {message.content}")
    return "\n".join(formatted_history[-5:])

def get_sql_chain(schema, chat_history, question):
    template = """
    Vous êtes un expert SQL. Convertissez cette question en requête SQL précise.
    Schéma : {schema}
    Historique : {chat_history}
    Question : {question}
    Requête SQL :
    """
    
    formatted_history = format_chat_history(chat_history)
    
    return template.format(
        schema=schema, 
        chat_history=formatted_history, 
        question=question
    )

def get_nl_response(sql_query, schema, sql_response):
    template = """
    Transformer le résultat SQL en réponse claire en français.
    Schéma : {schema}
    Requête : {sql_query}
    Résultat : {sql_response}
    Réponse: 
    """
    return template.format(sql_query=sql_query, schema=schema, sql_response=sql_response)

# Initialisation de l'historique
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage("Bonjour ! Je suis votre assistant SQL."),
    ]

def get_gemini_response(question, prompt, max_retries=3):
    for _ in range(max_retries):
        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content([prompt, question])
            return response.text.strip()
        except Exception as e:
            st.warning(f"Tentative échouée : {e}")
    return "Désolé, je n'ai pas pu générer de réponse."

def safe_add_to_history(chat_history, message):
    """Ajoute un message à l'historique de manière sécurisée"""
    if message and isinstance(message, str) and message.strip():
        chat_history.append(AIMessage(content=message))
    elif not message:
        chat_history.append(AIMessage(content="Je n'ai pas pu générer de réponse."))

# Configuration Streamlit
st.set_page_config(page_title="Assistant SQL", page_icon="💬")
st.header("Interrogez Votre Base de Données")

# Récupération du schéma
schema = get_schema(db)

# Affichage de l'historique
for message in st.session_state.chat_history:
    with st.chat_message("ai" if isinstance(message, AIMessage) else "human"):
        st.markdown(message.content)

# Champ de saisie
typing_user_query = st.chat_input("Posez votre question...")

if typing_user_query and typing_user_query.strip() != "":
    # Ajouter la question de l'utilisateur
    st.session_state.chat_history.append(HumanMessage(content=typing_user_query))
    
    with st.chat_message("human"):
        st.markdown(typing_user_query)
    
    with st.chat_message("ai"):
        # Générer la requête SQL
        prompt = get_sql_chain(schema, st.session_state.chat_history, typing_user_query)
        sql_query = get_gemini_response(typing_user_query, prompt)
        
        if sql_query:
            try:
                # Nettoyer la requête SQL
                sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
                
                # Exécuter la requête
                sql_response = db.run(sql_query)
                
                # Générer la réponse en langage naturel
                response_prompt = get_nl_response(sql_query, schema, sql_response)
                response = get_gemini_response(typing_user_query, response_prompt)
                
                st.markdown(response)
                # Ajouter la réponse à l'historique de manière sécurisée
                safe_add_to_history(st.session_state.chat_history, response)
                
            except Exception as e:
                error_response = f"Désolé, je n'ai pas pu exécuter la requête SQL, car la question n'est pas claire ou les données ne sont pas disponibles."
                st.markdown(error_response)
                safe_add_to_history(st.session_state.chat_history, error_response)
        else:
            error_response = "Désolé, je n'ai pas pu générer de requête SQL valide."
            st.markdown(error_response)
            safe_add_to_history(st.session_state.chat_history, error_response)
