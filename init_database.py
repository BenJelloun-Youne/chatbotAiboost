import sqlite3

def init_database():
    try:
        # Lecture du fichier SQL
        with open('init_db.sql', 'r') as sql_file:
            sql_script = sql_file.read()

        # Connexion à la base de données
        conn = sqlite3.connect('call_center_full_extended.db')
        cursor = conn.cursor()

        # Exécution du script SQL
        cursor.executescript(sql_script)

        # Validation des changements
        conn.commit()
        print("Base de données initialisée avec succès!")

    except Exception as e:
        print(f"Erreur lors de l'initialisation de la base de données: {str(e)}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    init_database() 