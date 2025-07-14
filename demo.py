import os
import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DB_FILE = "demo_notes.db"
INDEX_FILE = "demo_index.faiss"
model = SentenceTransformer("all-MiniLM-L6-v2")


def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS notes(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL
    )''')
    conn.commit()
    conn.close()


def load_index():
    if os.path.exists(INDEX_FILE):
        return faiss.read_index(INDEX_FILE)
    else:
        dim = model.get_sentence_embedding_dimension()
        return faiss.IndexIDMap(faiss.IndexFlatL2(dim))


def add_note(content):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO notes (content) VALUES (?)", (content,))
    conn.commit()
    note_id = c.lastrowid
    conn.close()

    embedding = model.encode([content])[0].astype('float32')
    index = load_index()
    index.add_with_ids(np.array([embedding]), np.array([note_id]))
    faiss.write_index(index, INDEX_FILE)
    print(f"✅ [{note_id}] Note saved and indexed.")


def search_notes(query, k=3):
    embedding = model.encode([query])[0].astype('float32')
    index = load_index()

    if index.ntotal == 0:
        print("⚠️ Index is empty.")
        return []

    D, I = index.search(np.array([embedding]), k)

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    results = []
    for note_id in I[0]:
        c.execute("SELECT content FROM notes WHERE id=?", (int(note_id),))
        row = c.fetchone()
        if row:
            results.append((note_id, row[0]))
    conn.close()
    return results


if __name__ == "__main__":
    init_db()

    # Longer journal-style notes
    demo_notes = [
        "Met Priya for coffee at the bookstore near college. We discussed the project idea and decided to meet again next week to finalize the proposal.",
        "Had a Zoom call with Alex and the rest of the remote team. We finalized the budget plan and talked about the upcoming product demo.",
        "Went to the park alone. Took a notebook and wrote ideas for my personal blog. Felt peaceful and productive. No distractions.",
        "Visited grandma in the afternoon. She told me stories from her youth. We had lunch together, and I fixed her TV afterwards.",
        "Caught up with Ravi after almost a year. We talked about our lives, careers, and how things have changed since college. A refreshing chat.",
        "Started reading ‘Sapiens’ again. The chapter on the Agricultural Revolution still fascinates me. Humans trading freedom for stability is a powerful theme."
    ]

    for note in demo_notes:
        add_note(note)

    print("\n🔍 Searching for similar notes to: 'Who did I meet yesterday?'")
    matches = search_notes("Who told me stories about her youth?", k=5)
    for note_id, content in matches:
        print(f"\n📌 [{note_id}] {content}")
