import os
import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, request, jsonify, redirect, url_for
import time

# Configuration
DB_FILE = "web_notes.db"
INDEX_FILE = "web_notes_index.faiss"

# Initialize Flask app
app = Flask(__name__)

# Initialize the SentenceTransformer model
print("Loading SentenceTransformer model...")
start_time = time.time()
model = SentenceTransformer('all-MiniLM-L6-v2')
load_time = time.time() - start_time
print(f"Model loaded in {load_time:.1f} seconds")

def load_index():
    """Load or initialize the FAISS index"""
    if os.path.exists(INDEX_FILE):
        return faiss.read_index(INDEX_FILE)
    else:
        dimension = model.get_sentence_embedding_dimension()
        index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
        return index

def init_db():
    """Initialize the SQLite database"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS notes(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              content TEXT NOT NULL,
              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP             
              )''')
    conn.commit()
    conn.close()

def get_all_notes():
    """Retrieve all notes from the database"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, content, created_at FROM notes ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def add_note(content):
    """Add a new note to the database and FAISS index"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO notes (content) VALUES (?)", (content,))
    conn.commit()
    note_id = c.lastrowid
    conn.close()
    
    # Generate embedding and add to FAISS index
    embed = model.encode([content])[0]
    index = load_index()
    index.add_with_ids(np.array([embed]).astype('float32'), np.array([note_id]))
    faiss.write_index(index, INDEX_FILE)
    
    return note_id

def search_notes(query, k=5):
    """Search for similar notes using semantic similarity"""
    index = load_index()
    
    if index.ntotal == 0:
        return []
    
    # Generate embedding for the query
    query_embed = model.encode([query])[0]
    
    # Search for similar embeddings
    scores, ids = index.search(np.array([query_embed]).astype('float32'), k)
    
    # Get the actual notes from the database
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    results = []
    for i, note_id in enumerate(ids[0]):
        if note_id != -1:  # -1 means no result found
            c.execute("SELECT id, content, created_at FROM notes WHERE id = ?", (int(note_id),))
            row = c.fetchone()
            if row:
                results.append({
                    'id': row[0],
                    'content': row[1],
                    'created_at': row[2],
                    'similarity_score': float(scores[0][i])
                })
    
    conn.close()
    return results

@app.route('/')
def index():
    """Home page - show all notes"""
    notes = get_all_notes()
    return render_template('index.html', notes=notes)

@app.route('/add', methods=['GET', 'POST'])
def add_note_page():
    """Add note page"""
    if request.method == 'POST':
        content = request.form.get('content', '').strip()
        if content:
            note_id = add_note(content)
            return redirect(url_for('index'))
        else:
            return render_template('add_note.html', error='Please enter some content for the note.')
    
    return render_template('add_note.html')

@app.route('/search')
def search_page():
    """Search page"""
    query = request.args.get('q', '').strip()
    results = []
    
    if query:
        results = search_notes(query)
    
    return render_template('search.html', query=query, results=results)

@app.route('/api/add_note', methods=['POST'])
def api_add_note():
    """API endpoint to add a note"""
    data = request.get_json()
    if not data or 'content' not in data:
        return jsonify({'error': 'Content is required'}), 400
    
    content = data['content'].strip()
    if not content:
        return jsonify({'error': 'Content cannot be empty'}), 400
    
    try:
        note_id = add_note(content)
        return jsonify({'success': True, 'note_id': note_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search')
def api_search():
    """API endpoint to search notes"""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    try:
        results = search_notes(query)
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/notes')
def api_get_notes():
    """API endpoint to get all notes"""
    try:
        notes = get_all_notes()
        notes_list = []
        for note in notes:
            notes_list.append({
                'id': note[0],
                'content': note[1],
                'created_at': note[2]
            })
        return jsonify({'notes': notes_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize database
    init_db()
    print("Database initialized")
    print("Starting Flask web server...")
    print("Access the app at: http://localhost:5000")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
