import os
import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DB_FILE = "notes.db"
INDEX_FILE = "notes_index.faiss"

# Load or initialize the FAISS index
def load_index():
    if os.path.exists(INDEX_FILE):
        return faiss.read_index(INDEX_FILE)
    else:
        dimension = model.get_sentence_embedding_dimension()
        index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
        return index

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS notes(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              content TEXT NOT NULL              
              )''')
    conn.commit()
    conn.close()

def get_all_notes():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, content FROM notes")
    rows = c.fetchall()
    conn.close()
    return rows

def add_note(content):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO notes (content) VALUES (?)", (content,))
    conn.commit()
    note_id = c.lastrowid
    print(f"Note ID: {note_id}")
    
    embed = model.encode([content])[0]
    index = load_index()
    index.add_with_ids(np.array([embed]).astype('float32'), np.array([note_id]))
    faiss.write_index(index, INDEX_FILE)
    
    print("✅ Note saved.")

# Example usage:
init_db()  # Initialize the database
add_note("This is a test note.")  # Add a note with content
