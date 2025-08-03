import psutil
import time
import threading
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import numpy as np

def monitor_cpu_usage(duration, interval=0.1):
    """Monitor CPU usage over time"""
    cpu_data = []
    memory_data = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_mb = psutil.virtual_memory().used / (1024 * 1024)
        cpu_data.append(cpu_percent)
        memory_data.append(memory_mb)
        time.sleep(interval)
    
    return cpu_data, memory_data

def test_embedding_performance():
    """Test CPU usage during different operations"""
    
    print("CPU Performance Testing")
    print("=" * 50)
    
    # Test 1: Model Loading
    print("\nTest 1: Model Loading")
    print("Starting CPU monitor...")
    
    cpu_data = []
    memory_data = []
    
    def cpu_monitor():
        nonlocal cpu_data, memory_data
        start_time = time.time()
        while time.time() - start_time < 10:  # Monitor for 10 seconds
            cpu_data.append(psutil.cpu_percent(interval=0.1))
            memory_data.append(psutil.virtual_memory().used / (1024 * 1024))
    
    # Start monitoring
    monitor_thread = threading.Thread(target=cpu_monitor, daemon=True)
    monitor_thread.start()
    
    # Load model (this will spike CPU)
    print("Loading model...")
    start_time = time.time()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    load_time = time.time() - start_time
    
    # Wait for monitoring to finish
    time.sleep(2)
    
    print(f"Model loaded in {load_time:.2f} seconds")
    print(f"Peak CPU during loading: {max(cpu_data):.1f}%")
    print(f"Average CPU during loading: {np.mean(cpu_data):.1f}%")
    print(f"Memory after loading: {memory_data[-1]:.0f} MB")
    
    # Test 2: Embedding Generation
    print("\nTest 2: Embedding Generation (Multiple Notes)")
    
    test_texts = [
        "This is a test note about programming and software development.",
        "Meeting with team members to discuss project timeline and deliverables.",
        "Morning exercise routine including running and strength training.",
        "Reading a book about machine learning and artificial intelligence.",
        "Financial planning session with investment portfolio review."
    ]
    
    cpu_before = psutil.cpu_percent(interval=1)
    memory_before = psutil.virtual_memory().used / (1024 * 1024)
    
    print("Generating embeddings for 5 notes...")
    embedding_times = []
    
    for i, text in enumerate(test_texts, 1):
        start_time = time.time()
        embedding = model.encode([text])
        embed_time = time.time() - start_time
        embedding_times.append(embed_time)
        
        cpu_during = psutil.cpu_percent(interval=None)
        print(f"   Note {i}: {embed_time:.3f}s (CPU: {cpu_during:.1f}%)")
    
    cpu_after = psutil.cpu_percent(interval=1)
    memory_after = psutil.virtual_memory().used / (1024 * 1024)
    
    print(f"\nAverage embedding time: {np.mean(embedding_times):.3f}s")
    print(f"CPU usage: {cpu_before:.1f}% -> {cpu_after:.1f}%")
    print(f"Memory usage: {memory_before:.0f} MB -> {memory_after:.0f} MB")
    
    # Test 3: Search Performance  
    print("\nTest 3: Search Query Performance")
    
    search_queries = [
        "programming software development",
        "team meeting project",
        "exercise fitness health",
        "books learning education",
        "money finance investment"
    ]
    
    search_times = []
    cpu_usages = []
    
    print("Testing search queries...")
    for i, query in enumerate(search_queries, 1):
        cpu_before = psutil.cpu_percent(interval=None)
        start_time = time.time()
        query_embedding = model.encode([query])
        search_time = time.time() - start_time
        cpu_after = psutil.cpu_percent(interval=None)
        
        search_times.append(search_time)
        cpu_usages.append(cpu_after)
        
        print(f"   Query {i}: {search_time:.3f}s (CPU: {cpu_after:.1f}%)")
    
    print(f"\nAverage search time: {np.mean(search_times):.3f}s")
    print(f"Average CPU during search: {np.mean(cpu_usages):.1f}%")
    
    # Test 4: Concurrent Operations
    print("\nTest 4: Concurrent Embedding Generation")
    
    def generate_embedding_batch(texts, thread_id):
        thread_times = []
        for text in texts:
            start_time = time.time()
            model.encode([text])
            thread_times.append(time.time() - start_time)
        return thread_times
    
    # Split texts into batches
    batch1 = test_texts[:3]
    batch2 = test_texts[3:]
    
    cpu_before = psutil.cpu_percent(interval=1)
    
    print("Running concurrent embedding generation...")
    start_time = time.time()
    
    # Run in parallel threads
    thread1 = threading.Thread(target=generate_embedding_batch, args=(batch1, 1))
    thread2 = threading.Thread(target=generate_embedding_batch, args=(batch2, 2))
    
    thread1.start()
    thread2.start()
    
    thread1.join()
    thread2.join()
    
    concurrent_time = time.time() - start_time
    cpu_after = psutil.cpu_percent(interval=1)
    
    print(f"Concurrent processing completed in {concurrent_time:.2f}s")
    print(f"CPU usage during concurrent ops: {cpu_before:.1f}% -> {cpu_after:.1f}%")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Model Loading: {load_time:.1f}s (High CPU)")
    print(f"Single Embedding: {np.mean(embedding_times)*1000:.0f}ms (Medium CPU)")
    print(f"Search Query: {np.mean(search_times)*1000:.0f}ms (Low CPU)")
    print(f"Memory Usage: ~{memory_after:.0f}MB after model load")
    
    return {
        'model_load_time': load_time,
        'avg_embedding_time': np.mean(embedding_times),
        'avg_search_time': np.mean(search_times),
        'memory_usage_mb': memory_after
    }

if __name__ == "__main__":
    # Install required packages: pip install psutil matplotlib
    try:
        results = test_embedding_performance()
        print(f"\nResults saved: {results}")
    except ImportError:
        print("Please install required packages: pip install psutil matplotlib")
    except Exception as e:
        print(f"Error: {e}")
