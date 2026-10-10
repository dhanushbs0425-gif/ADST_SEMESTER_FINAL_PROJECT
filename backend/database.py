"""
SeedSense AI - Database Manager (SQLite)
Stores analysis history and batch seed predictions.
"""

import sqlite3
import json
import os
from datetime import datetime

from backend.config import DATABASE_PATH

def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Ensure correct schema by checking columns or re-creating
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_history'")
    table_exists = cursor.fetchone()
    
    if table_exists:
        cursor.execute("PRAGMA table_info(analysis_history)")
        columns = [col['name'] for col in cursor.fetchall()]
        if 'date' not in columns or 'healthy_percentage' not in columns:
            cursor.execute("DROP TABLE IF EXISTS seed_details")
            cursor.execute("DROP TABLE IF EXISTS analysis_history")
    
    # Table 1: Batch Analysis Summary
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            filename TEXT NOT NULL,
            seed_name TEXT NOT NULL,
            condition TEXT NOT NULL,
            defect_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            total_seeds INTEGER NOT NULL,
            healthy_count INTEGER NOT NULL,
            defective_count INTEGER NOT NULL,
            healthy_percentage REAL NOT NULL,
            model_used TEXT NOT NULL,
            annotated_filename TEXT NOT NULL,
            species_distribution TEXT,
            defect_distribution TEXT
        )
    ''')
    
    # Table 2: Individual Seed Details for Batch Analysis
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS seed_details (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER NOT NULL,
            seed_index INTEGER NOT NULL,
            seed_name TEXT NOT NULL,
            condition TEXT NOT NULL,
            defect_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            FOREIGN KEY (batch_id) REFERENCES analysis_history (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def save_analysis(result):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = os.path.basename(result.get('original_image_path', 'uploaded_image.jpg'))
    
    cursor.execute('''
        INSERT INTO analysis_history (
            date, filename, seed_name, condition, defect_type, confidence,
            total_seeds, healthy_count, defective_count, healthy_percentage,
            model_used, annotated_filename, species_distribution, defect_distribution
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        date_str,
        filename,
        result.get('primary_seed_name', 'Unknown'),
        result.get('primary_condition', 'Healthy'),
        result.get('primary_defect_type', 'None'),
        result.get('primary_confidence', 0.0),
        result.get('total_seeds', 1),
        result.get('healthy_count', 0),
        result.get('defective_count', 0),
        result.get('healthy_percentage', 0.0),
        result.get('model_used', 'DL'),
        result.get('annotated_image_filename', 'annotated_result.jpg'),
        json.dumps(result.get('species_distribution', {})),
        json.dumps(result.get('defect_distribution', {}))
    ))
    
    batch_id = cursor.lastrowid
    
    # Insert individual seed predictions
    predictions = result.get('predictions', [])
    for idx, p in enumerate(predictions):
        cursor.execute('''
            INSERT INTO seed_details (
                batch_id, seed_index, seed_name, condition, defect_type, confidence
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            batch_id,
            idx + 1,
            p.get('seed_name', 'Unknown'),
            p.get('condition', 'Healthy'),
            p.get('defect_type', 'None'),
            p.get('confidence', 0.0)
        ))
        
    conn.commit()
    conn.close()
    return batch_id

def get_history():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM analysis_history ORDER BY id DESC')
    rows = cursor.fetchall()
    history = [dict(r) for r in rows]
    conn.close()
    return history

def get_analysis_by_id(record_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM analysis_history WHERE id = ?', (record_id,))
    row = cursor.fetchone()
    if row is None:
        conn.close()
        return None
        
    analysis = dict(row)
    
    cursor.execute('SELECT * FROM seed_details WHERE batch_id = ? ORDER BY seed_index ASC', (record_id,))
    details = [dict(r) for r in cursor.fetchall()]
    
    analysis['seeds'] = details
    conn.close()
    return analysis

# Initialize database on module import
init_db()
