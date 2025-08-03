import os
import sqlite3
import faiss
import numpy as np
import time
from sentence_transformers import SentenceTransformer

DB_FILE = "enhanced_demo_notes.db"
INDEX_FILE = "enhanced_demo_index.faiss"

# Initialize model with loading feedback
print("loading sentencetransformer model (this may take 30-60 seconds on first run)...")
start_time = time.time()
model = SentenceTransformer("all-MiniLM-L6-v2")
load_time = time.time() - start_time
print(f"model loaded in {load_time:.1f} seconds")


def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS notes(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        category TEXT
    )''')
    conn.commit()
    conn.close()


def load_index():
    if os.path.exists(INDEX_FILE):
        return faiss.read_index(INDEX_FILE)
    else:
        dim = model.get_sentence_embedding_dimension()
        return faiss.IndexIDMap(faiss.IndexFlatL2(dim))


def add_note(content, category=None):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO notes (content, category) VALUES (?, ?)", (content, category))
    conn.commit()
    note_id = c.lastrowid
    conn.close()

    # Show progress for embedding generation
    print(f"generating embedding for note {note_id}...", end=" ", flush=True)
    start_time = time.time()
    embedding = model.encode([content])[0].astype('float32')
    embed_time = time.time() - start_time
    
    index = load_index()
    index.add_with_ids(np.array([embedding]), np.array([note_id]))
    faiss.write_index(index, INDEX_FILE)
    print(f"done ({embed_time:.2f}s)")
    return note_id


def search_notes(query, k=5):
    embedding = model.encode([query])[0].astype('float32')
    index = load_index()

    if index.ntotal == 0:
        print("index is empty.")
        return []

    D, I = index.search(np.array([embedding]), k)

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    results = []
    for i, (distance, note_id) in enumerate(zip(D[0], I[0])):
        c.execute("SELECT content, category FROM notes WHERE id=?", (int(note_id),))
        row = c.fetchone()
        if row:
            results.append((note_id, row[0], row[1], distance))
    conn.close()
    return results


def setup_enhanced_demo_data():
    """Load comprehensive test data for accuracy evaluation"""
    
    enhanced_demo_notes = [
        # Technology & Programming (3 notes)
        ("Learned about Python decorators today. They're functions that modify other functions. Very useful for logging and authentication.", "technology"),
        ("Debugging JavaScript async/await issues. The problem was mixing callbacks with promises. Need to be consistent.", "technology"),
        ("Setting up Docker containers for microservices architecture. Each service gets its own container for better isolation.", "technology"),
        
        # Work & Meetings (3 notes)
        ("Team standup meeting with Sarah, Mike, and Jennifer. Discussed sprint planning and delivery timeline for Q3.", "work"),
        ("One-on-one with manager about career progression. Need to focus on leadership skills and technical architecture.", "work"),
        ("Client presentation went well. They approved the wireframes and want to proceed with development phase.", "work"),
        
        # Health & Fitness (3 notes)
        ("Morning run in Central Park. 5 miles in 42 minutes. Weather was perfect, sunny but cool.", "health"),
        ("Yoga class with instructor Lisa. Focused on flexibility and core strength. Feeling much more balanced.", "health"),
        ("Nutritionist appointment. Need to increase protein intake and reduce processed foods. Goal is 1g protein per lb bodyweight.", "health"),
        
        # Learning & Books (3 notes)
        ("Reading 'Atomic Habits' by James Clear. The 1% improvement concept is powerful for building consistent routines.", "learning"),
        ("Finished online course on machine learning. Linear regression and decision trees make sense now. Ready for neural networks.", "learning"),
        ("Book club discussion about 'Sapiens'. Fascinating how agriculture changed human society and social structures.", "learning"),
        
        # Family & Personal (3 notes)
        ("Video call with Mom and Dad. They're planning to visit next month. Need to clean the guest room and stock groceries.", "personal"),
        ("Dinner with college friends at that new Italian restaurant downtown. Caught up on everyone's life changes.", "personal"),
        ("Took grandmother to doctor appointment. Everything looks good for her age. She told stories about her childhood during the drive.", "personal"),
        
        # Finance & Planning (3 notes)
        ("Financial planning session with advisor. Reviewed 401k allocations and discussed Roth IRA conversion strategy.", "finance"),
        ("Monthly budget review. Overspent on dining out again. Need to meal prep more and limit restaurant visits.", "finance"),
        ("Investment portfolio rebalancing. Moving some growth stocks to index funds for better diversification.", "finance"),
        
        # Travel & Experiences (2 notes)
        ("Weekend trip to San Francisco. Visited Golden Gate Bridge and Alcatraz. Amazing views but very windy and cold.", "travel"),
        ("Planning summer vacation to Japan. Need to research visa requirements, book flights, and learn basic Japanese phrases.", "travel")
    ]
    
    print(f"loading {len(enhanced_demo_notes)} notes with embeddings...")
    print("progress: ", end="", flush=True)
    
    total_time = time.time()
    for i, (content, category) in enumerate(enhanced_demo_notes):
        progress = f"{i+1}/{len(enhanced_demo_notes)}"
        print(f"[{progress}]", end=" ", flush=True)
        
        # Add note without individual progress messages
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO notes (content, category) VALUES (?, ?)", (content, category))
        conn.commit()
        note_id = c.lastrowid
        conn.close()

        embedding = model.encode([content])[0].astype('float32')
        index = load_index()
        index.add_with_ids(np.array([embedding]), np.array([note_id]))
        faiss.write_index(index, INDEX_FILE)
    
    total_time = time.time() - total_time
    print(f"\nloaded {len(enhanced_demo_notes)} notes in {total_time:.1f} seconds")


def run_accuracy_tests():
    """Run comprehensive accuracy tests with expected results"""
    
    test_cases = [
        {
            "query": "programming languages and coding",
            "expected_categories": ["technology"],
            "expected_count": 3,
            "description": "Technology & Programming"
        },
        {
            "query": "who did I meet or talk to?",
            "expected_categories": ["work", "personal"],
            "expected_count": 4,
            "description": "People & Relationships"
        },
        {
            "query": "exercise and physical activity",
            "expected_categories": ["health"],
            "expected_count": 3,
            "description": "Health & Wellness"
        },
        {
            "query": "books and studying",
            "expected_categories": ["learning"],
            "expected_count": 3,
            "description": "Learning & Education"
        },
        {
            "query": "money and investments",
            "expected_categories": ["finance"],
            "expected_count": 3,
            "description": "Financial Topics"
        }
    ]
    
    print("\n" + "="*80)
    print("accuracy test results")
    print("="*80)
    
    total_tests = 0
    passed_tests = 0
    
    for test in test_cases:
        print(f"\ntest: {test['description']}")
        print(f"query: '{test['query']}'")
        
        # Time the search operation
        search_start = time.time()
        results = search_notes(test['query'], k=5)
        search_time = time.time() - search_start
        
        # Count relevant results in top 5
        relevant_count = 0
        for note_id, content, category, distance in results:
            if category in test['expected_categories']:
                relevant_count += 1
        
        # Calculate accuracy
        accuracy = (relevant_count / test['expected_count']) * 100
        
        print(f"expected: {test['expected_count']} relevant results")
        print(f"found: {relevant_count} relevant results in top 5")
        print(f"accuracy: {accuracy:.1f}%")
        print(f"search time: {search_time:.3f} seconds")
        
        # Show results
        print("top 5 results:")
        for i, (note_id, content, category, distance) in enumerate(results, 1):
            relevant_mark = "correct" if category in test['expected_categories'] else "wrong"
            print(f"   {i}. {relevant_mark} [{category}] {content[:80]}... (distance: {distance:.3f})")
        
        # Pass/Fail criteria (80% accuracy threshold)
        if accuracy >= 80:
            print("passed")
            passed_tests += 1
        else:
            print("failed")
        
        total_tests += 1
    
    print("\n" + "="*80)
    print(f"final results: {passed_tests}/{total_tests} tests passed ({(passed_tests/total_tests)*100:.1f}%)")
    print("="*80)


def interactive_search():
    """Interactive search interface"""
    print("\ninteractive search mode")
    print("type your queries (or 'quit' to exit):")
    
    while True:
        query = input("\n> ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
            
        results = search_notes(query, k=3)
        
        if not results:
            print("no results found.")
            continue
            
        print(f"\ntop 3 results for: '{query}'")
        for i, (note_id, content, category, distance) in enumerate(results, 1):
            print(f"\n{i}. [{note_id}] [{category}] (similarity: {1-distance:.3f})")
            print(f"   {content}")


if __name__ == "__main__":
    print("enhanced nlp notes demo - accuracy testing")
    print("=" * 50)
    
    # Initialize
    init_db()
    
    # Check if we need to load demo data
    index = load_index()
    if index.ntotal == 0:
        setup_enhanced_demo_data()
    else:
        print(f"found existing data: {index.ntotal} notes in index")
    
    # Run accuracy tests
    run_accuracy_tests()
    
    # Interactive mode
    interactive_search()
