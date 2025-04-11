import streamlit as st
from openai import OpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
import pandas as pd
import re
import time
import random
from datetime import datetime
import streamlit.components.v1 as components

# Configuration de la page
st.set_page_config(
    page_title="Assistant KPIs et DATA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e3f2fd;
    }
    .chat-message.assistant {
        background-color: #f5f5f5;
    }
    .stTextInput>div>div>input {
        background-color: white;
    }
    .thinking {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1rem;
    }
    .thinking-dots {
        display: flex;
        gap: 0.5rem;
    }
    .thinking-dot {
        width: 8px;
        height: 8px;
        background-color: #2196f3;
        border-radius: 50%;
        animation: bounce 1.4s infinite ease-in-out;
    }
    .thinking-dot:nth-child(1) { animation-delay: -0.32s; }
    .thinking-dot:nth-child(2) { animation-delay: -0.16s; }
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    </style>
    """, unsafe_allow_html=True)

# === Configuration API et base de données ===
try:
    # Configuration directe de l'API OpenAI
    OPENAI_API_KEY = "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"
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    st.error(f"Erreur lors de l'initialisation de l'API OpenAI : {str(e)}")

# Chemin vers votre base de données SQLite
try:
    DB_PATH = "call_center_full_extended.db"
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
except Exception as e:
    st.error(f"Erreur lors de la connexion à la base de données : {str(e)}")

# === Fonctions utilitaires SQL et chat ===

def get_schema(db):
    """Récupère les informations de schéma de la base de données."""
    return db.get_table_info()

def format_chat_history(chat_history, max_messages=5):
    """Formate l'historique de chat pour l'inclure dans le prompt."""
    formatted_history = []
    filtered_history = [
        msg for msg in chat_history
        if not (isinstance(msg, AIMessage) and "Bonjour" in msg.content)
    ]
    for message in filtered_history[-max_messages:]:
        role = "Utilisateur" if isinstance(message, HumanMessage) else "Assistant"
        formatted_history.append(f"{role}: {message.content}")
    return "\n".join(formatted_history)

def execute_sql_query(query):
    """Exécute une requête SQL et gère les exceptions."""
    try:
        # Nettoyage de la requête
        clean_query = re.sub(r'```sql|```', '', query).strip()
        result = db.run(clean_query)
        
        # Log de débogage
        st.write(f"Requête exécutée: {clean_query}")
        st.write(f"Résultat brut: {result}")
        
        # Vérification si le résultat est vide
        if not result:
            return "Aucun résultat trouvé pour cette requête."
            
        # Conversion du résultat en format texte
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], tuple):
                # Pour les requêtes de comptage simples (un seul nombre)
                if len(result) == 1 and len(result[0]) == 1:
                    return str(result[0][0])
                
                # Pour les autres requêtes avec plusieurs colonnes
                # Extraction des noms de colonnes de la requête
                column_names = []
                matches = re.findall(r'(?:AS\s+)(\w+)|COUNT\(\*\)\s+(?:AS\s+)?(\w+)', clean_query, re.IGNORECASE)
                if matches:
                    column_names = [m[0] or m[1] for m in matches if m[0] or m[1]]
                else:
                    # Si pas d'alias trouvés, extraire les noms des colonnes après SELECT
                    select_part = clean_query.upper().split('FROM')[0].replace('SELECT', '').strip()
                    columns = [col.strip().split()[-1] for col in select_part.split(',')]
                    column_names = [col.split('.')[-1] for col in columns]
                
                if not column_names:  # Si toujours pas de noms de colonnes, utiliser des noms génériques
                    column_names = [f"colonne_{i}" for i in range(len(result[0]))]
                
                # Création du résultat formaté
                formatted_result = ",".join(column_names) + "\n"
                for row in result:
                    formatted_result += ",".join(str(value) for value in row) + "\n"
                return formatted_result.strip()
            else:
                return str(result[0])
        else:
            return str(result)
            
    except Exception as e:
        st.error(f"Erreur lors de l'exécution de la requête SQL: {str(e)}")
        return f"Erreur: {str(e)}"

def generate_simple_sql(question, schema):
    """
    Génère une requête SQL simple basée sur des règles pour les cas communs.
    On se base ici sur des mots-clés pour déterminer la requête à générer.
    """
    question = question.lower().strip()
    
    # Requêtes sur les agents
    if any(pattern in question for pattern in ["combien d'agent", "nombre d'agent", "nombre agent"]):
        return "SELECT COUNT(*) FROM agents;"
    
    # Requêtes sur les performances des agents
    if (("top" in question and "perform" in question) or 
        ("meilleur" in question and "agent" in question)):
        return """
        SELECT a.agent_id, a.name, t.team_name, 
               SUM(p.sales) as total_sales, 
               SUM(p.appointments) as total_appointments,
               ROUND(AVG(p.satisfaction_score), 2) as avg_satisfaction
        FROM agents a
        JOIN performances p ON a.agent_id = p.agent_id
        JOIN teams t ON a.team_id = t.team_id
        GROUP BY a.agent_id
        ORDER BY total_sales DESC
        LIMIT 5;
        """
    
    # Requêtes sur les équipes
    if ("equipe" in question or "team" in question) and ("performance" in question or "resultat" in question):
        return """
        SELECT t.team_name, 
               COUNT(DISTINCT a.agent_id) as nombre_agents,
               SUM(p.sales) as total_ventes,
               ROUND(AVG(p.satisfaction_score), 2) as satisfaction_moyenne
        FROM teams t
        JOIN agents a ON t.team_id = a.team_id
        JOIN performances p ON a.agent_id = p.agent_id
        GROUP BY t.team_name
        ORDER BY total_ventes DESC;
        """
    
    # Requêtes sur les bonus
    if "bonus" in question:
        return """
        SELECT a.name, t.team_name,
               COUNT(b.bonus_id) as nombre_bonus,
               SUM(b.bonus_amount) as montant_total_bonus
        FROM agents a
        JOIN bonuses b ON a.agent_id = b.agent_id
        JOIN teams t ON a.team_id = t.team_id
        GROUP BY a.agent_id
        ORDER BY montant_total_bonus DESC
        LIMIT 5;
        """
    
    # Requêtes sur la satisfaction client
    if "satisfaction" in question:
        return """
        SELECT a.name, t.team_name,
               ROUND(AVG(p.satisfaction_score), 2) as satisfaction_moyenne,
               COUNT(p.performance_id) as nombre_evaluations
        FROM agents a
        JOIN performances p ON a.agent_id = p.agent_id
        JOIN teams t ON a.team_id = t.team_id
        GROUP BY a.agent_id
        ORDER BY satisfaction_moyenne DESC
        LIMIT 5;
        """
    
    # Requête par défaut (aperçu des performances globales)
    return """
    SELECT a.name, t.team_name,
           SUM(p.calls_made) as appels_total,
           SUM(p.sales) as ventes_total,
           SUM(p.appointments) as rdv_total,
           ROUND(AVG(p.satisfaction_score), 2) as satisfaction_moyenne
    FROM agents a
    JOIN performances p ON a.agent_id = p.agent_id
    JOIN teams t ON a.team_id = t.team_id
    GROUP BY a.agent_id
    ORDER BY ventes_total DESC
    LIMIT 5;
    """

def is_greeting_or_small_talk(text):
    """Détecte si le texte est une salutation, une question courte ou autre petite interaction."""
    text = text.lower().strip()
    greetings = ["bonjour", "salut", "hello", "coucou", "hey", "bonsoir", "bon matin"]
    small_talk = ["ça va", "comment ça va", "comment vas-tu", "comment va", "quoi de neuf"]
    thanks = ["merci", "thanks", "thank you", "je vous remercie", "je te remercie"]
    goodbyes = ["au revoir", "bye", "à bientôt", "à plus", "adieu", "bonne journée", "salut"]
    help_phrases = ["aide", "help", "besoin d'aide", "aidez-moi", "que peux-tu faire", "comment ça marche",
                    "que sais-tu faire", "comment utiliser", "utilisations possibles"]

    for phrase in greetings:
        if phrase in text:
            return True, "greeting"
    for phrase in small_talk:
        if phrase in text:
            return True, "small_talk"
    for phrase in thanks:
        if phrase in text:
            return True, "thanks"
    for phrase in goodbyes:
        if phrase in text:
            return True, "goodbye"
    for phrase in help_phrases:
        if phrase in text:
            return True, "help"
    if len(text.split()) < 4:
        return True, "short_question"
    return False, None

def get_simple_response(type_message):
    """Génère une réponse courte en fonction du type de message simple."""
    hour = datetime.now().hour
    if type_message == "greeting":
        if 5 <= hour < 12:
            responses = [
                "Bonjour ! Je suis là pour vous aider à analyser vos données. Voici ce que je peux faire :\n\n" +
                "1. Analyser les performances des agents et des équipes\n" +
                "2. Vérifier l'atteinte des objectifs\n" +
                "3. Examiner les retards et l'assiduité\n" +
                "4. Analyser les bonus et récompenses\n" +
                "5. Étudier la satisfaction client\n\n" +
                "Que souhaitez-vous savoir ?",
                
                "Bonjour ! Je peux vous aider à explorer vos KPIs. Par exemple, vous pouvez me demander :\n\n" +
                "- Combien d'agents avons-nous ?\n" +
                "- Quels sont nos meilleurs agents ?\n" +
                "- Comment performent nos équipes ?\n" +
                "- Qui a reçu le plus de bonus ?\n\n" +
                "Quelle information recherchez-vous ?"
            ]
        elif 12 <= hour < 18:
            responses = [
                "Bon après-midi ! Je suis là pour vous aider à analyser vos données. Voici ce que je peux faire :\n\n" +
                "1. Analyser les performances des agents et des équipes\n" +
                "2. Vérifier l'atteinte des objectifs\n" +
                "3. Examiner les retards et l'assiduité\n" +
                "4. Analyser les bonus et récompenses\n" +
                "5. Étudier la satisfaction client\n\n" +
                "Que souhaitez-vous savoir ?",
                
                "Bon après-midi ! Je peux vous aider à explorer vos KPIs. Par exemple, vous pouvez me demander :\n\n" +
                "- Combien d'agents avons-nous ?\n" +
                "- Quels sont nos meilleurs agents ?\n" +
                "- Comment performent nos équipes ?\n" +
                "- Qui a reçu le plus de bonus ?\n\n" +
                "Quelle information recherchez-vous ?"
            ]
        else:
            responses = [
                "Bonsoir ! Je suis là pour vous aider à analyser vos données. Voici ce que je peux faire :\n\n" +
                "1. Analyser les performances des agents et des équipes\n" +
                "2. Vérifier l'atteinte des objectifs\n" +
                "3. Examiner les retards et l'assiduité\n" +
                "4. Analyser les bonus et récompenses\n" +
                "5. Étudier la satisfaction client\n\n" +
                "Que souhaitez-vous savoir ?",
                
                "Bonsoir ! Je peux vous aider à explorer vos KPIs. Par exemple, vous pouvez me demander :\n\n" +
                "- Combien d'agents avons-nous ?\n" +
                "- Quels sont nos meilleurs agents ?\n" +
                "- Comment performent nos équipes ?\n" +
                "- Qui a reçu le plus de bonus ?\n\n" +
                "Quelle information recherchez-vous ?"
            ]
        return random.choice(responses)
    elif type_message == "small_talk":
        return "Je vais très bien, merci ! Je suis là pour vous aider à analyser vos données. Que souhaitez-vous savoir sur vos KPIs ?"
    elif type_message == "thanks":
        return "Je vous en prie ! N'hésitez pas si vous avez d'autres questions sur vos KPIs."
    elif type_message == "goodbye":
        return "Au revoir ! N'hésitez pas à revenir si vous avez besoin d'analyser vos données."
    elif type_message == "help":
        return (
            "Je suis votre assistant KPIs et DATA. Voici ce que je peux faire :\n\n"
            "1. Analyser les performances des agents et des équipes\n"
            "2. Vérifier l'atteinte des objectifs\n"
            "3. Examiner les retards et l'assiduité\n"
            "4. Analyser les bonus et récompenses\n"
            "5. Étudier la satisfaction client\n\n"
            "Par exemple, vous pouvez me demander :\n"
            "- Combien d'agents avons-nous ?\n"
            "- Quels sont nos meilleurs agents ?\n"
            "- Comment performent nos équipes ?\n"
            "- Qui a reçu le plus de bonus ?"
        )
    elif type_message == "short_question":
        return (
            "Je peux vous aider à analyser vos données, mais j'ai besoin de plus de détails. Par exemple, vous pouvez me demander :\n\n"
            "- Combien d'agents avons-nous ?\n"
            "- Quels sont nos meilleurs agents ?\n"
            "- Comment performent nos équipes ?\n"
            "- Qui a reçu le plus de bonus ?\n\n"
            "Quelle information recherchez-vous précisément ?"
        )
    else:
        return (
            "Je suis là pour vous aider à analyser vos données. Voici ce que je peux faire :\n\n"
            "1. Analyser les performances des agents et des équipes\n"
            "2. Vérifier l'atteinte des objectifs\n"
            "3. Examiner les retards et l'assiduité\n"
            "4. Analyser les bonus et récompenses\n"
            "5. Étudier la satisfaction client\n\n"
            "Que souhaitez-vous savoir ?"
        )

def get_gemini_response(prompt, max_retries=3, backoff_factor=2):
    """Génère une réponse en utilisant l'API OpenAI."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Vous êtes un assistant spécialisé dans l'analyse de données et la génération de requêtes SQL."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Erreur lors de la génération de la réponse: {str(e)}")
                return "Désolé, je n'ai pas pu générer de réponse. Veuillez réessayer."
            time.sleep(backoff_factor ** attempt)

@st.cache_resource
def load_open_source_model():
    """Charge et met en cache le modèle open source GPT-Neo via Hugging Face."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = "EleutherAI/gpt-neo-2.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def get_open_source_response(prompt, max_new_tokens=150):
    """
    Obtient une réponse en utilisant le modèle open source GPT-Neo.
    """
    try:
        tokenizer, model = load_open_source_model()
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        return "Erreur: Impossible d'obtenir une réponse du modèle open source. " + str(e)

def get_sql_prompt(schema, chat_history, question):
    """Crée le prompt pour générer une requête SQL à partir d'une question."""
    template = """
    Vous êtes un expert SQL. Convertissez cette question en une requête SQL précise.
    
    Schéma de la base de données:
    {schema}
    
    Historique des conversations:
    {chat_history}
    
    Question utilisateur: {question}
    
    Répondez uniquement avec la requête SQL sans explication.
    """
    return template.format(
        schema=schema,
        chat_history=format_chat_history(chat_history),
        question=question
    )

def get_nl_response_prompt(schema, question, sql_query, sql_result):
    """Crée le prompt pour transformer le résultat SQL en réponse naturelle."""
    template = """
    Transformer le résultat SQL ci-dessous en une réponse claire en français.
    
    Schéma de la base de données:
    {schema}
    
    Question originale: {question}
    
    Requête SQL exécutée:
    {sql_query}
    
    Résultat de la requête:
    {sql_result}
    
    Expliquez les résultats de manière simple et claire. Si le résultat est vide, indiquez-en la raison.
    """
    return template.format(
        schema=schema,
        question=question,
        sql_query=sql_query,
        sql_result=sql_result
    )

def generate_simple_response(question, sql_result):
    """
    Génère une réponse simplifiée en langage naturel à partir du résultat SQL.
    Cette fonction analyse la chaîne résultat pour identifier les informations clés.
    """
    try:
        lines = sql_result.strip().split('\n')
        # Si la requête concerne le comptage des agents
        if "nombre_agents" in sql_result and len(lines) == 2:
            count = lines[1].strip()
            return f"Il y a {count} agents au total dans le centre d'appels."
        
        # Pour les performances (top ou low performers)
        if ("top" in question or "meilleur" in question) or ("low" in question or "pire" in question):
            agents_list = []
            for i in range(1, min(6, len(lines))):
                cols = lines[i].split(',')
                if len(cols) >= 4:
                    agents_list.append(f"{cols[1].strip()} ({cols[3].strip()})")
            if "top" in question or "meilleur" in question:
                return "Voici les meilleurs agents:\n\n" + "\n".join(f"{idx+1}. {agent}" for idx, agent in enumerate(agents_list))
            else:
                return "Voici les agents avec les performances les plus faibles:\n\n" + "\n".join(f"{idx+1}. {agent}" for idx, agent in enumerate(agents_list))
        
        # Pour les performances des équipes
        if "team_name" in sql_result and "nombre_agents" in sql_result and len(lines) > 1:
            teams_performance = []
            for i in range(1, min(4, len(lines))):
                cols = lines[i].split(',')
                if len(cols) >= 3:
                    team_name = cols[0].strip()
                    agents_count = cols[1].strip()
                    total_sales = cols[2].strip()
                    teams_performance.append(f"{team_name}: {total_sales} ventes avec {agents_count} agents")
            return "Performances des équipes:\n\n" + "\n".join(teams_performance)
        
        # Pour les objectifs atteints
        if "objectif" in question and "atteint" in sql_result:
            achieved = sum(1 for line in lines[1:] if "Oui" in line)
            total = len(lines) - 1
            return f"{achieved} agents sur {total} ont atteint leurs objectifs de vente."
        
        # Pour la satisfaction client
        if "satisfaction" in question and "score_moyen_satisfaction" in sql_result:
            cols = lines[1].split(',')
            if len(cols) >= 4:
                return f"L'agent avec le meilleur score de satisfaction est {cols[1].strip()} avec un score de {cols[3].strip()}/5."
        
        # Réponse générique
        return "Voici les résultats de votre requête. Pour plus de détails, consultez le tableau affiché."
    except Exception as e:
        return "Voici les résultats de votre requête. Pour plus d'analyse, merci de préciser votre demande."

def display_sql_result_as_table(result):
    """Convertit le résultat SQL en DataFrame pour l'affichage en tableau."""
    try:
        if isinstance(result, str):
            lines = result.strip().split('\n')
            if len(lines) > 1:
                header = [col.strip() for col in lines[0].split(',')]
                data = [ [cell.strip() for cell in line.split(',')] for line in lines[1:] ]
                df = pd.DataFrame(data, columns=header)
                return df
        return None
    except Exception:
        return None

def get_query_type(question):
    """Détermine le type de requête à partir de la question."""
    question = question.lower().strip()
    
    # Mots-clés plus flexibles pour la détection
    performance_keywords = ["perf", "performance", "meilleur", "top", "bon", "excellent", "fort"]
    count_keywords = ["combien", "nombre", "total", "quantité"]
    team_keywords = ["équipe", "team", "groupe", "service"]
    bonus_keywords = ["bonus", "prime", "récompense", "gratification"]
    satisfaction_keywords = ["satisfaction", "client", "évaluation", "note", "score"]
    attendance_keywords = ["présence", "absent", "retard", "punctualité"]
    
    # Détection plus intelligente
    words = question.split()
    performance_count = sum(1 for word in words if word in performance_keywords)
    count_count = sum(1 for word in words if word in count_keywords)
    team_count = sum(1 for word in words if word in team_keywords)
    bonus_count = sum(1 for word in words if word in bonus_keywords)
    satisfaction_count = sum(1 for word in words if word in satisfaction_keywords)
    attendance_count = sum(1 for word in words if word in attendance_keywords)
    
    # Décision basée sur le nombre de mots-clés trouvés
    if performance_count > 0:
        return "performance"
    elif count_count > 0:
        return "count"
    elif team_count > 0:
        return "team"
    elif bonus_count > 0:
        return "bonus"
    elif satisfaction_count > 0:
        return "satisfaction"
    elif attendance_count > 0:
        return "attendance"
    
    # Si aucun mot-clé n'est trouvé, on essaie de deviner d'après le contexte
    if any(word in question for word in ["agent", "employé", "collaborateur"]):
        return "performance"
    elif any(word in question for word in ["vente", "chiffre", "résultat"]):
        return "performance"
    elif any(word in question for word in ["temps", "heure", "jour"]):
        return "attendance"
    
    return "default"

def format_response(result, query_type):
    """Formate la réponse en fonction du type de requête."""
    try:
        if not result:
            return "Aucun résultat trouvé."
            
        if isinstance(result, str):
            return result
            
        if isinstance(result, list):
            if not result:
                return "Aucun résultat trouvé."
                
            if query_type == "count":
                return f"Il y a actuellement {result[0]} agents dans le centre d'appels."
                
            elif query_type == "performance":
                response = "Voici les performances des agents :\n\n"
                for agent in result:
                    if isinstance(agent, (list, tuple)):
                        # Vérification de la longueur et des valeurs
                        name = str(agent[0]) if len(agent) > 0 and agent[0] is not None else "N/A"
                        team = str(agent[1]) if len(agent) > 1 and agent[1] is not None else "N/A"
                        calls = str(agent[2]) if len(agent) > 2 and agent[2] is not None else "N/A"
                        sales = str(agent[3]) if len(agent) > 3 and agent[3] is not None else "N/A"
                        appointments = str(agent[4]) if len(agent) > 4 and agent[4] is not None else "N/A"
                        satisfaction = str(agent[5]) if len(agent) > 5 and agent[5] is not None else "N/A"
                        
                        response += f"• {name} ({team}) :\n"
                        response += f"  - Appels : {calls}\n"
                        response += f"  - Ventes : {sales}\n"
                        response += f"  - RDV : {appointments}\n"
                        response += f"  - Satisfaction : {satisfaction}/5\n\n"
                return response
                
            elif query_type == "team":
                response = "Voici les performances par équipe :\n\n"
                for team in result:
                    if isinstance(team, (list, tuple)):
                        name = str(team[0]) if len(team) > 0 and team[0] is not None else "N/A"
                        agents = str(team[1]) if len(team) > 1 and team[1] is not None else "N/A"
                        sales = str(team[2]) if len(team) > 2 and team[2] is not None else "N/A"
                        satisfaction = str(team[3]) if len(team) > 3 and team[3] is not None else "N/A"
                        
                        response += f"• {name} :\n"
                        response += f"  - Nombre d'agents : {agents}\n"
                        response += f"  - Ventes totales : {sales}\n"
                        response += f"  - Satisfaction moyenne : {satisfaction}/5\n\n"
                return response
                
        return str(result)
    except Exception as e:
        return f"Une erreur s'est produite lors du formatage de la réponse : {str(e)}"

def generate_sql_query(question, query_type):
    """Génère la requête SQL appropriée selon le type de question."""
    if query_type == "count":
        return "SELECT COUNT(*) FROM agents;"
        
    elif query_type == "performance":
        return """
        SELECT a.name, t.team_name,
               SUM(p.calls_made) as appels_total,
               SUM(p.sales) as ventes_total,
               SUM(p.appointments) as rdv_total,
               ROUND(AVG(p.satisfaction_score), 2) as satisfaction_moyenne
        FROM agents a
        JOIN performances p ON a.agent_id = p.agent_id
        JOIN teams t ON a.team_id = t.team_id
        GROUP BY a.agent_id
        ORDER BY ventes_total DESC
        LIMIT 5;
        """
        
    elif query_type == "team":
        return """
        SELECT t.team_name,
               COUNT(DISTINCT a.agent_id) as nombre_agents,
               SUM(p.sales) as ventes_total,
               ROUND(AVG(p.satisfaction_score), 2) as satisfaction_moyenne
        FROM teams t
        JOIN agents a ON t.team_id = a.team_id
        JOIN performances p ON a.agent_id = p.agent_id
        GROUP BY t.team_id
        ORDER BY ventes_total DESC;
        """
        
    elif query_type == "bonus":
        return """
        SELECT a.name, t.team_name,
               COUNT(b.bonus_id) as nombre_bonus,
               SUM(b.bonus_amount) as montant_total
        FROM agents a
        JOIN bonuses b ON a.agent_id = b.agent_id
        JOIN teams t ON a.team_id = t.team_id
        GROUP BY a.agent_id
        ORDER BY montant_total DESC
        LIMIT 5;
        """
        
    elif query_type == "satisfaction":
        return """
        SELECT a.name, t.team_name,
               ROUND(AVG(p.satisfaction_score), 2) as satisfaction_moyenne,
               COUNT(p.performance_id) as nombre_evaluations
        FROM agents a
        JOIN performances p ON a.agent_id = p.agent_id
        JOIN teams t ON a.team_id = t.team_id
        GROUP BY a.agent_id
        HAVING nombre_evaluations >= 10
        ORDER BY satisfaction_moyenne DESC
        LIMIT 5;
        """
        
    elif query_type == "attendance":
        return """
        SELECT a.name, t.team_name,
               COUNT(att.attendance_id) as jours_travailles,
               SUM(CASE WHEN att.is_present = 0 THEN 1 ELSE 0 END) as absences,
               SUM(att.tardiness_minutes) as minutes_retard_total
        FROM agents a
        JOIN attendance att ON a.agent_id = att.agent_id
        JOIN teams t ON a.team_id = t.team_id
        GROUP BY a.agent_id
        ORDER BY minutes_retard_total DESC
        LIMIT 5;
        """
    
    return None

# Ajout de la gestion de l'historique
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def get_contextual_response(question, chat_history):
    """Génère une réponse contextuelle en fonction de l'historique de la conversation."""
    # Analyse du contexte de la conversation
    context = " ".join([msg["content"] for msg in chat_history[-3:]])  # Derniers 3 messages
    
    # Détection du type de conversation
    if any(word in question.lower() for word in ["bonjour", "salut", "hello", "coucou"]):
        return "Bonjour ! Je suis votre assistant pour l'analyse des données. Comment puis-je vous aider aujourd'hui ?"
    
    elif any(word in question.lower() for word in ["merci", "thanks", "thank you"]):
        return "Je vous en prie ! N'hésitez pas si vous avez d'autres questions."
    
    elif any(word in question.lower() for word in ["au revoir", "bye", "à plus"]):
        return "Au revoir ! N'hésitez pas à revenir si vous avez besoin d'autres analyses."
    
    # Si la question est courte ou vague
    if len(question.split()) < 3:
        return "Je peux vous aider à analyser vos données, mais j'ai besoin de plus de détails. Par exemple, vous pouvez me demander :\n\n" + \
               "- Combien d'agents avons-nous ?\n" + \
               "- Quels sont nos meilleurs agents ?\n" + \
               "- Comment performent nos équipes ?\n" + \
               "- Qui a reçu le plus de bonus ?\n\n" + \
               "Quelle information recherchez-vous précisément ?"
    
    return None

def process_user_input(user_input):
    """Traite l'entrée utilisateur et retourne une réponse appropriée."""
    try:
        # Vérification de la réponse contextuelle
        contextual_response = get_contextual_response(user_input, st.session_state.chat_history)
        if contextual_response:
            return contextual_response
            
        # Vérification si c'est une salutation ou une petite conversation
        is_small_talk, talk_type = is_greeting_or_small_talk(user_input)
        if is_small_talk:
            return get_simple_response(talk_type)
            
        # Récupération du schéma de la base de données
        schema = get_schema(db)
        
        # Détermination du type de requête
        query_type = get_query_type(user_input)
        
        # Génération de la requête SQL
        sql_query = generate_sql_query(user_input, query_type)
        if not sql_query:
            return "Je n'ai pas pu comprendre votre demande. Pouvez-vous reformuler votre question ?"
        
        # Exécution de la requête
        result = execute_sql_query(sql_query)
        
        # Formatage de la réponse
        formatted_response = format_response(result, query_type)
        
        # Ajout d'une analyse supplémentaire si nécessaire
        if query_type == "performance":
            formatted_response += "\n\nAnalyse : Ces agents se distinguent par leurs excellentes performances en termes de ventes et de satisfaction client."
        
        return formatted_response
        
    except Exception as e:
        return f"Désolé, une erreur s'est produite : {str(e)}"

def display_thinking_animation():
    """Affiche une animation de réflexion."""
    st.markdown("""
        <div class="thinking">
            <div class="thinking-dots">
                <div class="thinking-dot"></div>
                <div class="thinking-dot"></div>
                <div class="thinking-dot"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# === Configuration de l'interface Streamlit ===
st.title("📊 Assistant KPIs et DATA")
st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='color: #666; font-size: 1.1rem;'>
            Interrogez votre base de données en langage naturel pour obtenir des réponses sur vos KPIs.
        </p>
    </div>
""", unsafe_allow_html=True)

# Initialisation de la session
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Bonjour ! Je suis votre assistant KPIs et DATA. Comment puis-je vous aider ?"
    })

# Affichage des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f"""
            <div class="chat-message {message['role']}">
                {message['content']}
            </div>
        """, unsafe_allow_html=True)

# Zone de saisie utilisateur
if prompt := st.chat_input("Posez votre question ici..."):
    # Ajout du message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(f"""
            <div class="chat-message user">
                {prompt}
            </div>
        """, unsafe_allow_html=True)

    # Réponse de l'assistant avec animation de réflexion
    with st.chat_message("assistant"):
        # Afficher l'animation de réflexion
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("""
            <div class="thinking">
                <div class="thinking-dots">
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Traitement de la réponse
        response = process_user_input(prompt)
        
        # Remplacer l'animation par la réponse
        thinking_placeholder.empty()
        st.markdown(f"""
            <div class="chat-message assistant">
                {response}
            </div>
        """, unsafe_allow_html=True)
        
        # Ajout de la réponse à l'historique
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# Ajout d'une barre latérale avec des exemples de questions
with st.sidebar:
    st.markdown("""
        <div style='padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem;'>
            <h3>💡 Exemples de questions</h3>
            <ul style='list-style-type: none; padding-left: 0;'>
                <li style='margin-bottom: 0.5rem;'>• Combien d'agents avons-nous ?</li>
                <li style='margin-bottom: 0.5rem;'>• Quels sont nos meilleurs agents ?</li>
                <li style='margin-bottom: 0.5rem;'>• Comment performent nos équipes ?</li>
                <li style='margin-bottom: 0.5rem;'>• Qui a reçu le plus de bonus ?</li>
                <li style='margin-bottom: 0.5rem;'>• Quels sont les objectifs des agents ?</li>
                <li style='margin-bottom: 0.5rem;'>• Qui a le meilleur taux de présence ?</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
