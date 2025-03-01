import streamlit as st
import google.generativeai as genai
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
import pandas as pd
import re
import time
import random
from datetime import datetime

# === Configuration API et base de données ===

# Configurez votre API Gemini (remplacez par votre clé)
genai.configure(api_key="VOTRE_CLE_API_ICI")

# Chemin vers votre base de données SQLite
DB_PATH = "call_center_full_extended.db"
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

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
        if not result or result.strip() == "":
            return "Aucun résultat trouvé pour cette requête."
        return result
    except Exception as e:
        st.error(f"Erreur lors de l'exécution de la requête SQL: {str(e)}")
        return f"Erreur: {str(e)}"

def generate_simple_sql(question, schema):
    """
    Génère une requête SQL simple basée sur des règles pour les cas communs.
    On se base ici sur des mots-clés pour déterminer la requête à générer.
    """
    question = question.lower()
    # Requêtes sur les agents
    if "combien d'agent" in question or "nombre d'agent" in question:
        return "SELECT COUNT(*) as nombre_agents FROM agents;"
    
    if (("top" in question and "perform" in question) or 
        ("meilleur" in question and "agent" in question)):
        return """
        SELECT a.agent_id, a.name, a.position, t.team_name, 
               SUM(p.sales) as total_sales, 
               SUM(p.appointments) as total_appointments,
               ROUND(AVG(p.satisfaction_score), 2) as avg_satisfaction
        FROM agents a
        JOIN performances p ON a.agent_id = p.agent_id
        JOIN teams t ON a.team_id = t.team_id
        GROUP BY a.agent_id
        ORDER BY total_sales DESC, total_appointments DESC
        LIMIT 5;
        """
    
    if (("low" in question and "perform" in question) or 
        ("pire" in question and "agent" in question)):
        return """
        SELECT a.agent_id, a.name, a.position, t.team_name, 
               SUM(p.sales) as total_sales, 
               SUM(p.appointments) as total_appointments,
               ROUND(AVG(p.satisfaction_score), 2) as avg_satisfaction
        FROM agents a
        JOIN performances p ON a.agent_id = p.agent_id
        JOIN teams t ON a.team_id = t.team_id
        GROUP BY a.agent_id
        ORDER BY total_sales ASC, total_appointments ASC
        LIMIT 5;
        """
    
    # Requêtes sur les équipes
    if ("tableau" in question or "performance" in question) and "equipe" in question:
        return """
        SELECT t.team_name, 
               COUNT(DISTINCT a.agent_id) as nombre_agents,
               SUM(p.sales) as total_ventes,
               SUM(p.appointments) as total_rdv,
               ROUND(AVG(p.satisfaction_score), 2) as satisfaction_moyenne,
               SUM(p.calls_made) as total_appels
        FROM teams t
        JOIN agents a ON t.team_id = a.team_id
        JOIN performances p ON a.agent_id = p.agent_id
        GROUP BY t.team_name
        ORDER BY total_ventes DESC;
        """
    
    # Requêtes sur les objectifs
    if "objectif" in question and "atteint" in question:
        return """
        SELECT a.agent_id, a.name, t.team_name,
               SUM(p.sales) as ventes_realisees, 
               pg.sales_target as objectif_ventes,
               CASE WHEN SUM(p.sales) >= pg.sales_target THEN 'Oui' ELSE 'Non' END as objectif_atteint
        FROM agents a
        JOIN performances p ON a.agent_id = p.agent_id
        JOIN teams t ON a.team_id = t.team_id
        JOIN performance_goals pg ON a.agent_id = pg.agent_id
        GROUP BY a.agent_id
        ORDER BY (SUM(p.sales) * 1.0 / pg.sales_target) DESC;
        """
    
    # Requêtes sur présence et retards
    if "retard" in question or "absent" in question:
        return """
        SELECT a.agent_id, a.name, t.team_name,
               COUNT(att.attendance_id) as jours_travailles,
               SUM(CASE WHEN att.is_present = 0 THEN 1 ELSE 0 END) as absences,
               SUM(CASE WHEN att.tardiness_minutes > 0 THEN 1 ELSE 0 END) as jours_avec_retard,
               SUM(att.tardiness_minutes) as minutes_retard_total
        FROM agents a
        JOIN attendance att ON a.agent_id = att.agent_id
        JOIN teams t ON a.team_id = t.team_id
        GROUP BY a.agent_id
        ORDER BY minutes_retard_total DESC;
        """
    
    # Requêtes sur les bonus
    if "bonus" in question:
        return """
        SELECT a.agent_id, a.name, t.team_name,
               COUNT(b.bonus_id) as nombre_bonus,
               SUM(b.bonus_amount) as montant_total_bonus,
               GROUP_CONCAT(DISTINCT b.reason) as raisons
        FROM agents a
        JOIN bonuses b ON a.agent_id = b.agent_id
        JOIN teams t ON a.team_id = t.team_id
        GROUP BY a.agent_id
        ORDER BY montant_total_bonus DESC;
        """
    
    # Requêtes sur la satisfaction client
    if "satisfaction" in question:
        return """
        SELECT a.agent_id, a.name, t.team_name,
               ROUND(AVG(p.satisfaction_score), 2) as score_moyen_satisfaction,
               COUNT(p.performance_id) as nombre_evaluations
        FROM agents a
        JOIN performances p ON a.agent_id = p.agent_id
        JOIN teams t ON a.team_id = t.team_id
        GROUP BY a.agent_id
        ORDER BY score_moyen_satisfaction DESC;
        """
    
    # Requête générique (aperçu des performances globales)
    return """
    SELECT a.agent_id, a.name, t.team_name,
           SUM(p.calls_made) as appels_total,
           SUM(p.sales) as ventes_total,
           SUM(p.appointments) as rdv_total,
           ROUND(AVG(p.satisfaction_score), 2) as satisfaction_moyenne
    FROM agents a
    JOIN performances p ON a.agent_id = p.agent_id
    JOIN teams t ON a.team_id = t.team_id
    GROUP BY a.agent_id
    ORDER BY ventes_total DESC
    LIMIT 10;
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
                "Bonjour ! Comment puis-je vous aider avec vos données aujourd'hui ?",
                "Bonjour ! Je suis prêt à analyser vos KPIs. Que souhaitez-vous savoir ?"
            ]
        elif 12 <= hour < 18:
            responses = [
                "Bon après-midi ! Comment puis-je vous aider avec vos données aujourd'hui ?",
                "Bonjour ! Je suis prêt à analyser vos KPIs. Que souhaitez-vous savoir ?"
            ]
        else:
            responses = [
                "Bonsoir ! Comment puis-je vous aider avec vos données aujourd'hui ?",
                "Bonsoir ! Je suis prêt à analyser vos KPIs. Que souhaitez-vous savoir ?"
            ]
        return random.choice(responses)
    elif type_message == "small_talk":
        return random.choice([
            "Je vais très bien, merci ! Que souhaitez-vous analyser aujourd'hui ?",
            "Tout va bien ! Comment puis-je vous aider à explorer vos KPIs ?"
        ])
    elif type_message == "thanks":
        return random.choice([
            "Je vous en prie !",
            "Avec plaisir, n'hésitez pas si vous avez d'autres questions."
        ])
    elif type_message == "goodbye":
        return random.choice([
            "Au revoir ! Revenez quand vous voulez pour analyser vos données.",
            "À bientôt !"
        ])
    elif type_message == "help":
        return (
            "Je suis votre assistant KPIs et DATA. Voici ce que je peux faire :\n\n"
            "1. Analyser les performances (agents, équipes, etc.)\n"
            "2. Vérifier l'atteinte des objectifs\n"
            "3. Analyser la présence et les retards\n"
            "4. Examiner les bonus\n"
            "5. Analyser la satisfaction client\n\n"
            "Par exemple, vous pouvez demander : « Quels sont les meilleurs agents ? » ou « Montre-moi les performances des équipes »."
        )
    elif type_message == "short_question":
        return random.choice([
            "Pouvez-vous préciser votre demande afin que je puisse mieux vous aider ?",
            "Merci de donner plus de détails sur votre question."
        ])
    else:
        return "Comment puis-je vous aider avec vos données aujourd'hui ?"

def get_gemini_response(prompt, max_retries=3, backoff_factor=2):
    """
    Obtient une réponse du modèle Gemini.
    En cas d'erreur ou de dépassement de quota, on attend puis on réessaie.
    """
    models_to_try = ["gemini-pro", "gemini-1.0-pro", "text-bison@001"]
    for model_name in models_to_try:
        for attempt in range(max_retries):
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                if "429" in str(e) or "404" in str(e):
                    wait_time = backoff_factor ** attempt
                    st.warning(f"Modèle {model_name} indisponible ou quota atteint, attente de {wait_time} secondes...")
                    time.sleep(wait_time)
                    if attempt == max_retries - 1:
                        if "Convertissez cette question en requête SQL" in prompt:
                            question = prompt.split("Question utilisateur:")[-1].strip()
                            return generate_simple_sql(question, get_schema(db))
                        return "Je n'ai pas pu générer une réponse détaillée en raison des limites de quota."
                else:
                    st.warning(f"Tentative {attempt+1} avec {model_name} échouée: {e}")
                    if attempt == max_retries - 1:
                        break
    if "Convertissez cette question en requête SQL" in prompt:
        question = prompt.split("Question utilisateur:")[-1].strip()
        return generate_simple_sql(question, get_schema(db))
    return "Erreur: Impossible d'accéder aux modèles Gemini. Passage en mode simplifié."

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

# === Configuration de l'interface Streamlit ===

st.set_page_config(
    page_title="Assistant KPIs et DATA", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer le style
st.markdown("""
<style>
    .main { background-color: #f5f7f9; }
    .stApp { max-width: 1200px; margin: 0 auto; }
    .chat-container { background-color: white; border-radius: 10px; padding: 20px; margin-top: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .metric-card { background-color: white; border-radius: 5px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; }
    .metric-value { font-size: 24px; font-weight: bold; color: #1E3A8A; }
    .metric-label { font-size: 14px; color: #6B7280; }
</style>
""", unsafe_allow_html=True)

# Titre et description
st.title("📊 Assistant KPIs et DATA")
st.markdown("Interrogez votre base de données en langage naturel pour obtenir des réponses sur vos KPIs.")

# === Chargement et affichage des KPIs de base ===

if 'kpis_loaded' not in st.session_state:
    try:
        total_agents_query = "SELECT COUNT(*) as nombre_agents FROM agents;"
        total_teams_query = "SELECT COUNT(*) as nombre_equipes FROM teams;"
        total_sales_query = "SELECT SUM(sales) as total_ventes FROM performances;"
        avg_satisfaction_query = "SELECT ROUND(AVG(satisfaction_score), 2) as satisfaction_moyenne FROM performances;"
        
        total_agents = execute_sql_query(total_agents_query).strip().split('\n')[1]
        total_teams = execute_sql_query(total_teams_query).strip().split('\n')[1]
        total_sales = execute_sql_query(total_sales_query).strip().split('\n')[1]
        avg_satisfaction = execute_sql_query(avg_satisfaction_query).strip().split('\n')[1]
        
        st.session_state.total_agents = total_agents
        st.session_state.total_teams = total_teams
        st.session_state.total_sales = total_sales
        st.session_state.avg_satisfaction = avg_satisfaction
        st.session_state.kpis_loaded = True
    except Exception as e:
        st.error(f"Erreur lors du chargement des KPIs: {str(e)}")
        st.session_state.kpis_loaded = False

if st.session_state.get("kpis_loaded", False):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.total_agents}</div><div class="metric-label">Agents</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.total_teams}</div><div class="metric-label">Équipes</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.total_sales}</div><div class="metric-label">Ventes Totales</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.avg_satisfaction}/5</div><div class="metric-label">Satisfaction Moyenne</div></div>', unsafe_allow_html=True)

# === Barre latérale avec informations sur le schéma et options ===

with st.sidebar:
    st.header("Informations sur la base de données")
    schema_info = """
    Tables:
    - agents (agent_id, name, position, team_id, work_hours)
    - attendance (attendance_id, agent_id, date, is_present, tardiness_minutes)
    - bonuses (bonus_id, agent_id, bonus_amount, reason)
    - performance_goals (goal_id, agent_id, calls_target, sales_target, appointments_target)
    - performances (performance_id, agent_id, date, calls_made, sales, appointments, answered_calls, qualified_leads, non_qualified_leads, pending_leads, call_result, satisfaction_score)
    - teams (team_id, team_name)
    """
    with st.expander("Schéma de la base de données", expanded=False):
        st.code(schema_info, language="sql")
    st.subheader("Options")
    show_sql = st.checkbox("Afficher les requêtes SQL", value=True)
    show_results_as_table = st.checkbox("Afficher les résultats sous forme de tableau", value=True)
    use_simple_mode = st.checkbox("Mode simplifié (sans API)", value=False, 
                                  help="Utiliser ce mode en cas de problème de quota avec l'API")
    st.subheader("Exemples de questions")
    st.markdown("""
    - Combien d'agents avons-nous au total ?
    - Quels sont les meilleurs agents ?
    - Montrez-moi les performances des équipes
    - Quels agents ont le plus de retard ?
    - Qui a reçu le plus de bonus ?
    - Quels agents ont atteint leurs objectifs ?
    - Qui a le meilleur score de satisfaction client ?
    """)
    st.markdown("---")
    st.markdown("### À propos")
    st.markdown("Cet assistant vous permet d'interroger votre base de données via des questions en langage naturel pour analyser vos KPIs.")

# === Historique du chat ===

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage("Bonjour ! Je suis votre assistant KPIs et DATA. Comment puis-je vous aider ?")]

with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            st.chat_message("assistant").markdown(message.content)
        else:
            st.chat_message("user").markdown(message.content)
    st.markdown('</div>', unsafe_allow_html=True)

# === Gestion de la requête utilisateur ===

user_query = st.chat_input("Posez votre question sur les KPIs ou les données...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.chat_message("user").markdown(user_query)
    
    with st.chat_message("assistant"):
        is_simple, type_message = is_greeting_or_small_talk(user_query)
        if is_simple:
            response = get_simple_response(type_message)
            st.markdown(response)
            st.session_state.chat_history.append(AIMessage(content=response))
        else:
            with st.spinner("Analyse de votre question..."):
                schema = schema_info  # Utilisation du schéma défini dans la sidebar
                # Génération de la requête SQL (mode API ou mode simple)
                if use_simple_mode:
                    sql_query = generate_simple_sql(user_query, schema)
                else:
                    sql_prompt = get_sql_prompt(schema, st.session_state.chat_history, user_query)
                    sql_query = get_gemini_response(sql_prompt)
                
                # Vérification de la validité de la requête générée
                if (not sql_query or 
                    "limites de quota" in sql_query.lower() or 
                    "n'ai pas pu générer" in sql_query.lower() or 
                    sql_query.startswith("Erreur:")):
                    fallback_query = generate_simple_sql(user_query, schema)
                    st.markdown("**Requête SQL générée (mode secours):**")
                    st.code(fallback_query, language="sql")
                    sql_query = fallback_query
                else:
                    if show_sql:
                        st.markdown("**Requête SQL générée:**")
                        st.code(sql_query, language="sql")
                
                sql_result = execute_sql_query(sql_query)
                if show_results_as_table:
                    df = display_sql_result_as_table(sql_result)
                    if df is not None and not df.empty:
                        st.markdown("**Résultats:**")
                        st.dataframe(df, use_container_width=True)
                
                if use_simple_mode:
                    response = generate_simple_response(user_query, sql_result)
                else:
                    nl_prompt = get_nl_response_prompt(schema, user_query, sql_query, sql_result)
                    response = get_gemini_response(nl_prompt)
                    if response.startswith("Erreur:"):
                        response = generate_simple_response(user_query, sql_result)
                st.markdown("**Analyse:**")
                st.markdown(response)
                st.session_state.chat_history.append(AIMessage(content=response))
