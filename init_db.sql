-- Suppression des tables existantes
DROP TABLE IF EXISTS attendance;
DROP TABLE IF EXISTS performances;
DROP TABLE IF EXISTS bonuses;
DROP TABLE IF EXISTS performance_goals;
DROP TABLE IF EXISTS agents;
DROP TABLE IF EXISTS teams;

-- Création des tables
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    team_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agents (
    agent_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    position TEXT NOT NULL,
    team_id INTEGER,
    work_hours TEXT,
    FOREIGN KEY (team_id) REFERENCES teams (team_id)
);

CREATE TABLE IF NOT EXISTS performance_goals (
    goal_id INTEGER PRIMARY KEY,
    agent_id INTEGER,
    calls_target INTEGER,
    sales_target INTEGER,
    appointments_target INTEGER,
    FOREIGN KEY (agent_id) REFERENCES agents (agent_id)
);

CREATE TABLE IF NOT EXISTS bonuses (
    bonus_id INTEGER PRIMARY KEY,
    agent_id INTEGER,
    bonus_amount REAL,
    reason TEXT,
    bonus_date TEXT,
    FOREIGN KEY (agent_id) REFERENCES agents (agent_id)
);

CREATE TABLE IF NOT EXISTS performances (
    performance_id INTEGER PRIMARY KEY,
    agent_id INTEGER,
    date TEXT,
    calls_made INTEGER,
    sales INTEGER,
    appointments INTEGER,
    answered_calls INTEGER,
    qualified_leads INTEGER,
    non_qualified_leads INTEGER,
    pending_leads INTEGER,
    call_result TEXT,
    satisfaction_score REAL,
    FOREIGN KEY (agent_id) REFERENCES agents (agent_id)
);

CREATE TABLE IF NOT EXISTS attendance (
    attendance_id INTEGER PRIMARY KEY,
    agent_id INTEGER,
    date TEXT,
    is_present INTEGER,
    tardiness_minutes INTEGER,
    FOREIGN KEY (agent_id) REFERENCES agents (agent_id)
);

-- Insertion de données de test
INSERT INTO teams (team_name) VALUES 
('Équipe A'),
('Équipe B'),
('Équipe C'),
('Équipe D');

INSERT INTO agents (name, position, team_id, work_hours) VALUES 
('Jean Dupont', 'Senior Agent', 1, '9:00-17:00'),
('Marie Martin', 'Junior Agent', 1, '9:00-17:00'),
('Pierre Durand', 'Senior Agent', 2, '10:00-18:00'),
('Sophie Bernard', 'Team Lead', 2, '9:00-17:00'),
('Lucas Petit', 'Junior Agent', 3, '8:00-16:00'),
('Emma Dubois', 'Senior Agent', 3, '9:00-17:00'),
('Thomas Moreau', 'Team Lead', 4, '9:00-17:00'),
('Julie Lambert', 'Senior Agent', 4, '10:00-18:00');

-- Ajout de performances sur plusieurs jours
INSERT INTO performances (agent_id, date, calls_made, sales, appointments, answered_calls, qualified_leads, satisfaction_score) VALUES 
(1, '2024-03-01', 50, 5, 3, 45, 10, 4.5),
(1, '2024-03-02', 48, 4, 2, 43, 9, 4.3),
(1, '2024-03-03', 52, 6, 4, 47, 11, 4.6),
(2, '2024-03-01', 45, 3, 2, 40, 8, 4.2),
(2, '2024-03-02', 42, 2, 1, 38, 7, 4.1),
(2, '2024-03-03', 46, 4, 3, 41, 9, 4.3),
(3, '2024-03-01', 55, 6, 4, 50, 12, 4.8),
(3, '2024-03-02', 58, 7, 5, 52, 13, 4.9),
(3, '2024-03-03', 53, 5, 3, 48, 11, 4.7),
(4, '2024-03-01', 40, 4, 3, 35, 7, 4.3),
(4, '2024-03-02', 42, 5, 4, 38, 8, 4.4),
(4, '2024-03-03', 45, 6, 5, 41, 9, 4.5),
(5, '2024-03-01', 48, 4, 2, 43, 9, 4.1),
(5, '2024-03-02', 45, 3, 2, 40, 8, 4.0),
(5, '2024-03-03', 50, 5, 3, 45, 10, 4.2),
(6, '2024-03-01', 52, 5, 3, 47, 10, 4.4),
(6, '2024-03-02', 55, 6, 4, 50, 11, 4.5),
(6, '2024-03-03', 50, 4, 3, 45, 9, 4.3),
(7, '2024-03-01', 45, 4, 3, 40, 8, 4.2),
(7, '2024-03-02', 48, 5, 4, 43, 9, 4.3),
(7, '2024-03-03', 50, 6, 5, 45, 10, 4.4),
(8, '2024-03-01', 50, 5, 3, 45, 10, 4.3),
(8, '2024-03-02', 52, 6, 4, 47, 11, 4.4),
(8, '2024-03-03', 48, 4, 3, 43, 9, 4.2);

-- Ajout de bonus
INSERT INTO bonuses (agent_id, bonus_amount, reason, bonus_date) VALUES 
(1, 200, 'Objectif dépassé', '2024-03-01'),
(1, 150, 'Excellente performance', '2024-03-03'),
(3, 300, 'Meilleur vendeur du mois', '2024-03-01'),
(3, 200, 'Record de ventes', '2024-03-02'),
(4, 150, 'Excellence service client', '2024-03-01'),
(6, 100, 'Progression remarquable', '2024-03-02'),
(7, 250, 'Leadership exceptionnel', '2024-03-01'),
(8, 180, 'Performance constante', '2024-03-03');

-- Ajout d'objectifs
INSERT INTO performance_goals (agent_id, calls_target, sales_target, appointments_target) VALUES 
(1, 150, 15, 9),
(2, 120, 10, 6),
(3, 160, 18, 12),
(4, 130, 15, 12),
(5, 120, 10, 6),
(6, 150, 15, 9),
(7, 140, 15, 10),
(8, 150, 15, 9);

-- Ajout de données de présence
INSERT INTO attendance (agent_id, date, is_present, tardiness_minutes) VALUES 
(1, '2024-03-01', 1, 0),
(1, '2024-03-02', 1, 5),
(1, '2024-03-03', 1, 0),
(2, '2024-03-01', 1, 10),
(2, '2024-03-02', 1, 0),
(2, '2024-03-03', 0, 0),
(3, '2024-03-01', 1, 0),
(3, '2024-03-02', 1, 0),
(3, '2024-03-03', 1, 0),
(4, '2024-03-01', 1, 0),
(4, '2024-03-02', 1, 0),
(4, '2024-03-03', 1, 0),
(5, '2024-03-01', 1, 15),
(5, '2024-03-02', 1, 0),
(5, '2024-03-03', 1, 5),
(6, '2024-03-01', 1, 0),
(6, '2024-03-02', 1, 0),
(6, '2024-03-03', 1, 0),
(7, '2024-03-01', 1, 0),
(7, '2024-03-02', 1, 0),
(7, '2024-03-03', 1, 0),
(8, '2024-03-01', 1, 0),
(8, '2024-03-02', 1, 0),
(8, '2024-03-03', 1, 0); 