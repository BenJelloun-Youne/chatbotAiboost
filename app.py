import streamlit as st
import google.generativeai as genai
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
import pandas as pd
import re
import os
import time
import random
from datetime import datetime, timedelta

# Configuration de l'API Gemini
genai.configure(api_key="AIzaSyCwWitJOAQDe8jsogTiPmep5ToOw_Vl-Rk")  # Remplacez par votre clé API si nécessaire

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

def generate_simple_sql(question, schema):
    """Génère une requête SQL simple basée sur des règles pour les cas communs."""
    question = question.lower()
    
    # Requêtes pour les agents
    if "combien d'agent" in question or "nombre d'agent" in question:
        return "SELECT COUNT(*) as nombre_agents FROM agents;"
    
    if ("top" in question and "perform" in question) or ("meilleur" in question and "agent" in question):
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
    
    if ("low" in question and "perform" in question) or ("pire" in question and "agent" in question):
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
    
    # Requêtes pour les équipes
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
    
    # Requêtes pour les objectifs et réalisations
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
    
    # Requêtes pour les présences et retards
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
    
    # Requêtes pour les bonus
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
    
    # Requêtes pour la satisfaction client
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
    
    # Requête générale pour avoir un aperçu des performances globales par agent
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
    """Détecte si le texte est une salutation ou une question simple."""
    text = text.lower().strip()
    
    # Définir les modèles de salutations et de questions simples
    greetings = ["bonjour", "salut", "hello", "coucou", "hey", "bonsoir", "bon matin"]
    small_talk = ["ça va", "comment ça va", "comment vas-tu", "comment va", "quoi de neuf"]
    thanks = ["merci", "thanks", "thank you", "je vous remercie", "je te remercie"]
    goodbyes = ["au revoir", "bye", "à bientôt", "à plus", "adieu", "bonne journée", "salut"]
    help_phrases = ["aide", "help", "besoin d'aide", "aidez-moi", "que peux-tu faire", "comment ça marche", 
                   "que sais-tu faire", "comment utiliser", "utilisations possibles"]
    
    # Vérifier si le texte contient une des expressions
    for greeting in greetings:
        if greeting in text or text == greeting:
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
    
    # Vérifier si c'est une question très courte (moins de 4 mots)
    if len(text.split()) < 4:
        return True, "short_question"
    
    return False, None

def get_simple_response(type_message):
    """Génère une réponse appropriée selon le type de message simple."""
    current_time = datetime.now()
    hour = current_time.hour
    
    # Réponses pour les salutations
    if type_message == "greeting":
        if 5 <= hour < 12:
            greetings = [
                "Bonjour ! Comment puis-je vous aider avec vos données aujourd'hui ?",
                "Bonjour ! Je suis prêt à analyser vos KPIs. Que souhaitez-vous savoir ?",
                "Bonjour ! Comment puis-je vous assister avec l'analyse des performances ?"
            ]
        elif 12 <= hour < 18:
            greetings = [
                "Bon après-midi ! Comment puis-je vous aider avec vos données aujourd'hui ?",
                "Bonjour ! Je suis prêt à analyser vos KPIs. Que souhaitez-vous savoir ?",
                "Bonjour ! Comment puis-je vous assister avec l'analyse des performances ?"
            ]
        else:
            greetings = [
                "Bonsoir ! Comment puis-je vous aider avec vos données aujourd'hui ?",
                "Bonsoir ! Je suis prêt à analyser vos KPIs. Que souhaitez-vous savoir ?",
                "Bonsoir ! Comment puis-je vous assister avec l'analyse des performances ?"
            ]
        return random.choice(greetings)
    
    # Réponses pour les questions sur l'état
    elif type_message == "small_talk":
        responses = [
            "Je vais très bien, merci ! Je suis prêt à vous aider avec vos analyses de données. Que voulez-vous savoir ?",
            "Tout va bien ! Je suis ici pour vous aider à explorer vos KPIs. Qu'aimeriez-vous analyser ?",
            "Je suis opérationnel et prêt à vous assister ! Comment puis-je vous aider avec vos données aujourd'hui ?"
        ]
        return random.choice(responses)
    
    # Réponses pour les remerciements
    elif type_message == "thanks":
        responses = [
            "Je vous en prie ! N'hésitez pas si vous avez d'autres questions sur vos données.",
            "C'est un plaisir de vous aider. Y a-t-il autre chose que vous souhaiteriez analyser ?",
            "De rien ! Je suis là pour vous aider à comprendre vos KPIs. Avez-vous d'autres questions ?"
        ]
        return random.choice(responses)
    
    # Réponses pour les au revoir
    elif type_message == "goodbye":
        responses = [
            "Au revoir ! N'hésitez pas à revenir si vous avez besoin d'analyser vos données.",
            "À bientôt ! Je serai là si vous avez besoin d'aide avec vos KPIs.",
            "Bonne journée ! Revenez quand vous souhaitez explorer vos performances."
        ]
        return random.choice(responses)
    
    # Réponses pour les demandes d'aide
    elif type_message == "help":
        return """
        Je suis votre assistant KPIs et DATA. Voici comment je peux vous aider :
        
        1. **Analyser les performances** - Demandez-moi les meilleurs agents, les performances par équipe, etc.
        2. **Vérifier les objectifs** - Je peux vous dire quels agents ont atteint leurs objectifs
        3. **Analyser la présence** - Consultez les données d'absences et de retards
        4. **Examiner les bonus** - Voyez qui a reçu des bonus et pourquoi
        5. **Analyser la satisfaction client** - Découvrez les scores de satisfaction
        
        Essayez de me poser une question comme "Qui sont les meilleurs agents ?" ou "Montre-moi les performances des équipes".
        """
    
    # Réponses pour les questions très courtes
    elif type_message == "short_question":
        responses = [
            "Pourriez-vous me donner plus de détails sur ce que vous souhaitez savoir ? Je peux vous aider à analyser vos KPIs, les performances des agents, les équipes, etc.",
            "Je serais ravi de vous aider. Pourriez-vous préciser votre question sur les données que vous souhaitez analyser ?",
            "Pour mieux vous aider, j'aurais besoin de plus de détails. Que voulez-vous savoir précisément sur vos KPIs ou vos données ?"
        ]
        return random.choice(responses)
    
    # Réponse par défaut si le type n'est pas reconnu
    else:
        return "Je suis votre assistant KPIs et DATA. Comment puis-je vous aider à analyser vos données aujourd'hui ?"

def get_gemini_response(prompt, max_retries=3, backoff_factor=2):
    """Obtient une réponse du modèle Gemini avec gestion de quota."""
    # Liste des modèles à essayer
    models_to_try = ["gemini-pro", "gemini-1.0-pro", "text-bison@001"]
    
    for model_name in models_to_try:
        for attempt in range(max_retries):
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                # Si erreur de quota (429) ou de modèle indisponible (404)
                if "429" in str(e) or "404" in str(e):
                    wait_time = backoff_factor ** attempt
                    st.warning(f"Limite de quota atteinte ou modèle indisponible, attente de {wait_time} secondes...")
                    time.sleep(wait_time)
                    if attempt == max_retries - 1:
                        # En cas de quota dépassé, utiliser la génération de SQL simple
                        if "Convertissez cette question en requête SQL" in prompt:
                            # Extraire la question de l'utilisateur du prompt
                            schema = get_schema(db)
                            question = prompt.split("Question utilisateur:")[-1].strip()
                            return generate_simple_sql(question, schema)
                        else:
                            # Pour les prompts non-SQL, générer une réponse simple
                            return "Je n'ai pas pu générer une réponse détaillée en raison des limites de quota. Voici une réponse basique basée sur les données disponibles."
                else:
                    st.warning(f"Tentative {attempt+1} avec {model_name} échouée: {e}")
                    # Si ce n'est pas une erreur de quota et c'est la dernière tentative avec ce modèle, continuez au modèle suivant
                    if attempt == max_retries - 1:
                        break
    
    # Si tous les modèles échouent, retourner un message d'erreur et utiliser une approche de secours
    # Vérifier si c'est une requête SQL et utiliser la méthode simple le cas échéant
    if "Convertissez cette question en requête SQL" in prompt:
        schema = get_schema(db)
        question = prompt.split("Question utilisateur:")[-1].strip()
        return generate_simple_sql(question, schema)
    
    return "Erreur: Impossible d'accéder aux modèles Gemini en raison des limites de quota. J'utilise un mode simplifié pour vous répondre."

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

def generate_simple_response(question, sql_result):
    """Générer une réponse simple sans utiliser l'API."""
    try:
        question = question.lower()
        lines = sql_result.strip().split('\n')
        header = lines[0].split(',')
        
        # Pour les requêtes de comptage d'agents
        if "nombre_agents" in sql_result and len(lines) == 2:
            count = lines[1].strip()
            return f"Il y a {count} agents au total dans le centre d'appels."
        
        # Pour les performances des agents (top performers)
        if "top" in question or "meilleur" in question:
            if len(lines) > 1 and "agent_id" in sql_result:
                best_agents = []
                for i in range(1, min(6, len(lines))):
                    columns = lines[i].split(',')
                    # Extraire le nom et l'équipe
                    if len(columns) >= 4:
                        agent_name = columns[1].strip()
                        team_name = columns[3].strip()
                        best_agents.append(f"{agent_name} ({team_name})")
                
                return f"Voici les meilleurs agents en termes de performance: \n\n1. {best_agents[0]}\n2. {best_agents[1] if len(best_agents) > 1 else ''}\n3. {best_agents[2] if len(best_agents) > 2 else ''}\n4. {best_agents[3] if len(best_agents) > 3 else ''}\n5. {best_agents[4] if len(best_agents) > 4 else ''}"
        
        # Pour les performances des agents (low performers)
        if "low" in question or "pire" in question:
            if len(lines) > 1 and "agent_id" in sql_result:
                low_agents = []
                for i in range(1, min(6, len(lines))):
                    columns = lines[i].split(',')
                    # Extraire le nom et l'équipe
                    if len(columns) >= 4:
                        agent_name = columns[1].strip()
                        team_name = columns[3].strip()
                        low_agents.append(f"{agent_name} ({team_name})")
                
                return f"Voici les agents ayant les performances les plus faibles: \n\n1. {low_agents[0]}\n2. {low_agents[1] if len(low_agents) > 1 else ''}\n3. {low_agents[2] if len(low_agents) > 2 else ''}\n4. {low_agents[3] if len(low_agents) > 3 else ''}\n5. {low_agents[4] if len(low_agents) > 4 else ''}"
        
        # Pour les performances des équipes
        if "team_name" in sql_result and "nombre_agents" in sql_result and len(lines) > 1:
            teams_performance = []
            for i in range(1, min(4, len(lines))):
                columns = lines[i].split(',')
                if len(columns) >= 6:
                    team_name = columns[0].strip()
                    agents_count = columns[1].strip()
                    total_sales = columns[2].strip()
                    teams_performance.append(f"{team_name}: {total_sales} ventes avec {agents_count} agents")
            
            return f"Performances des équipes:\n\n{teams_performance[0]}\n{teams_performance[1] if len(teams_performance) > 1 else ''}\n{teams_performance[2] if len(teams_performance) > 2 else ''}"
        
        # Pour les objectifs atteints
        if "objectif" in question and "atteint" in sql_result:
            achieved_goals = 0
            total_agents = len(lines) - 1
            for i in range(1, len(lines)):
                columns = lines[i].split(',')
                if len(columns) >= 6 and columns[5].strip() == 'Oui':
                    achieved_goals += 1
            
            return f"{achieved_goals} agents sur {total_agents} ont atteint leurs objectifs de vente. Voici les détails dans le tableau."
        
        # Pour les retards et absences
        if "retard" in question or "absent" in question:
            if "minutes_retard_total" in sql_result:
                return f"J'ai analysé les données de présence et retards pour les agents. Les détails sont présentés dans le tableau ci-dessus. Vous pouvez voir les minutes de retard cumulées et le nombre d'absences pour chaque agent."
        
        # Pour les bonus
        if "bonus" in question and "montant_total_bonus" in sql_result:
            return f"Voici l'analyse des bonus accordés aux agents. Le tableau montre le montant total des bonus et les raisons pour chaque agent."
        
        # Pour la satisfaction client
        if "satisfaction" in question and "score_moyen_satisfaction" in sql_result:
            best_agent_columns = lines[1].split(',')
            if len(best_agent_columns) >= 4:
                best_agent = best_agent_columns[1].strip()
                best_score = best_agent_columns[3].strip()
                return f"L'agent avec le meilleur score de satisfaction client est {best_agent} avec un score moyen de {best_score}/5. Les détails pour tous les agents sont dans le tableau."
        
        # Réponse générique
        return "Voici les résultats de votre requête. Le tableau ci-dessus présente les données demandées. Pour une analyse plus détaillée, vous pouvez me poser des questions spécifiques sur ces résultats."
        
    except Exception as e:
        return "Voici les résultats de votre requête. Pour une analyse plus détaillée, n'hésitez pas à me poser des questions spécifiques."

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
    .metric-card {
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 14px;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# En-tête
st.title("📊 Assistants KPIs et DATA")
st.markdown("Explorez vos données et KPIs du centre d'appels avec des questions en langage naturel.")

# Charger les données pour les KPIs si ce n'est pas déjà fait
if 'kpis_loaded' not in st.session_state:
    try:
        # Requêtes pour les KPIs de base
        total_agents_query = "SELECT COUNT(*) as nombre_agents FROM agents;"
        total_teams_query = "SELECT COUNT(*) as nombre_equipes FROM teams;"
        total_sales_query = "SELECT SUM(sales) as total_ventes FROM performances;"
        avg_satisfaction_query = "SELECT ROUND(AVG(satisfaction_score), 2) as satisfaction_moyenne FROM performances;"
        
        # Exécuter les requêtes
        total_agents = execute_sql_query(total_agents_query).strip().split('\n')[1]
        total_teams = execute_sql_query(total_teams_query).strip().split('\n')[1]
        total_sales = execute_sql_query(total_sales_query).strip().split('\n')[1]
        avg_satisfaction = execute_sql_query(avg_satisfaction_query).strip().split('\n')[1]
        
        # Stocker les résultats
        st.session_state.total_agents = total_agents
        st.session_state.total_teams = total_teams
        st.session_state.total_sales = total_sales
        st.session_state.avg_satisfaction = avg_satisfaction
        st.session_state.kpis_loaded = True
    except Exception as e:
        st.error(f"Erreur lors du chargement des KPIs: {str(e)}")
        st.session_state.kpis_loaded = False

# Afficher les KPIs en haut
if 'kpis_loaded' in st.session_state and st.session_state.kpis_loaded:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{st.session_state.total_agents}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Agents</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{st.session_state.total_teams}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Équipes</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{st.session_state.total_sales}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Ventes Totales</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{st.session_state.avg_satisfaction}/5</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Satisfaction Moyenne</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Barre latérale avec informations sur le schéma
    with st.sidebar:
        st.header("Informations sur la base de données")
        
        # Récupération et affichage du schéma
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
        
        # Options avancées
        st.subheader("Options")
        show_sql = st.checkbox("Afficher les requêtes SQL", value=True)
        show_results_as_table = st.checkbox("Afficher les résultats sous forme de tableau", value=True)
        use_simple_mode = st.checkbox("Mode simplifié (sans API)", value=False, 
                                   help="Utiliser ce mode en cas de problèmes de quota avec l'API")
        
        # Exemples de questions
        st.subheader("Exemples de questions")
        st.markdown("""
        - Combien d'agents avons-nous au total?
        - Quels sont les meilleurs agents?
        - Montrez-moi les performances des équipes
        - Quels agents ont le plus de retard?
        - Qui a reçu le plus de bonus?
        - Quels agents ont atteint leurs objectifs?
        - Qui a le meilleur score de satisfaction client?
        """)
        
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
            # Vérifier si c'est une salutation ou une question simple
            is_simple, type_message = is_greeting_or_small_talk(user_query)
            
            if is_simple:
                # Générer une réponse simple pour les salutations et petites discussions
                response = get_simple_response(type_message)
                st.markdown(response)
                st.session_state.chat_history.append(AIMessage(content=response))
            else:
                # Traiter comme une question normale sur les données
                with st.spinner("Analyse de votre question..."):
                    # 1. Récupérer le schéma
                    schema = schema_info
                    
                    # 2. Générer la requête SQL (avec ou sans API selon le mode)
                    if use_simple_mode:
                        sql_query = generate_simple_sql(user_query, schema)
                    else:
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
                        if use_simple_mode:
                            response = generate_simple_response(user_query, sql_result)
                        else:
                            nl_prompt = get_nl_response_prompt(schema, user_query, sql_query, sql_result)
                            response = get_gemini_response(nl_prompt)
                            # Si l'API échoue, utiliser la méthode simple
                            if response.startswith("Erreur:"):
                                response = generate_simple_response(user_query, sql_result)
                        
                        # 5. Afficher la réponse
                        st.markdown("**Analyse:**")
                        st.markdown(response)
                        
                        # 6. Ajouter la réponse à l'historique
                        st.session_state.chat_history.append(AIMessage(content=response))
                    else:
                        # Utiliser la méthode simple en cas d'échec
                        fallback_query = generate_simple_sql(user_query, schema)
                        
                        if show_sql:
                            st.markdown("**Requête SQL générée (mode secours):**")
                            st.code(fallback_query, language="sql")
                        
                        sql_result = execute_sql_query(fallback_query)
                        
                        if show_results_as_table:
                            df = display_sql_result_as_table(sql_result)
                            if df is not None and not df.empty:
                                st.markdown("**Résultats:**")
                                st.dataframe(df, use_container_width=True)
                        
                        response = generate_simple_response(user_query, sql_result)
                        st.markdown("**Analyse (mode simplifié):**")
                        st.markdown(response)
                        
                        st.session_state.chat_history.append(AIMessage(content=response))
