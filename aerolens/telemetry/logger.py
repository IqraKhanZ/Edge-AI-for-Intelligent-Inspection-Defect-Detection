import os
import sqlite3
import time

class TelemetryLogger:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inspection_log.db")
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inspection_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                aircraft_id TEXT,
                zone TEXT,
                defect_class TEXT,
                confidence REAL,
                severity_score REAL,
                urgency_band TEXT,
                image_path TEXT
            )
        """)
        conn.commit()
        conn.close()

    def log_detection(self, aircraft_id, zone, defect_class, confidence, severity_score, urgency_band, image_path):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO inspection_records 
            (timestamp, aircraft_id, zone, defect_class, confidence, severity_score, urgency_band, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, aircraft_id, zone, defect_class, confidence, severity_score, urgency_band, image_path))
        conn.commit()
        conn.close()

    def get_history(self, limit=100, offset=0):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM inspection_records 
            ORDER BY timestamp DESC 
            LIMIT ? OFFSET ?
        """, (limit, offset))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
