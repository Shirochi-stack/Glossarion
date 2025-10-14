#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnostic script to test if GlossaryManager is using true parallel processing.
This will help identify bottlenecks in sentence processing.
"""

import os
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import Counter
import re

# Simulate sentence processing task
def process_batch_thread(args):
    """Thread-based processing (GIL-limited)"""
    batch, batch_idx = args
    counter = Counter()
    
    # Simulate CPU-intensive regex work
    pattern = r'[가-힣]{2,4}'
    for sentence in batch:
        matches = re.findall(pattern, sentence)
        counter.update(matches)
    
    return counter, batch_idx

def process_batch_process(args):
    """Process-based processing (true parallelism)"""
    batch, batch_idx = args
    from collections import Counter
    import re
    
    counter = Counter()
    pattern = r'[가-힣]{2,4}'
    for sentence in batch:
        matches = re.findall(pattern, sentence)
        counter.update(matches)
    
    return counter, batch_idx

def test_parallel_processing(num_sentences=10000, num_workers=4):
    """Test both ThreadPoolExecutor and ProcessPoolExecutor"""
    
    print(f"\n{'='*60}")
    print(f"PARALLEL PROCESSING PERFORMANCE TEST")
    print(f"{'='*60}")
    print(f"Test parameters:")
    print(f"  - Total sentences: {num_sentences:,}")
    print(f"  - Workers: {num_workers}")
    print(f"  - CPU cores: {os.cpu_count()}")
    print(f"{'='*60}\n")
    
    # Generate test data
    print("Generating test data...")
    test_sentences = [
        f"이것은 테스트 문장입니다 {i} 번째 줄입니다. 김철수와 박영희가 대화를 나눕니다." * 3
        for i in range(num_sentences)
    ]
    print(f"Generated {len(test_sentences):,} sentences\n")
    
    # Calculate batch size (similar to your code)
    if num_sentences < 50000:
        batch_size = 500
    elif num_sentences < 200000:
        batch_size = 1000
    else:
        batch_size = 2000
    
    batches = [(test_sentences[i:i+batch_size], idx) 
               for idx, i in enumerate(range(0, len(test_sentences), batch_size))]
    print(f"Split into {len(batches)} batches of ~{batch_size} sentences\n")
    
    # Test 1: Sequential processing
    print("=" * 60)
    print("TEST 1: Sequential Processing (Baseline)")
    print("=" * 60)
    start = time.time()
    
    counter = Counter()
    pattern = r'[가-힣]{2,4}'
    for sentence in test_sentences:
        matches = re.findall(pattern, sentence)
        counter.update(matches)
    
    sequential_time = time.time() - start
    sequential_rate = num_sentences / sequential_time
    print(f"Time: {sequential_time:.2f}s")
    print(f"Rate: {sequential_rate:.0f} sentences/sec")
    print(f"Found {len(counter)} unique terms\n")
    
    # Test 2: ThreadPoolExecutor
    print("=" * 60)
    print("TEST 2: ThreadPoolExecutor (Python's default)")
    print("=" * 60)
    start = time.time()
    
    all_counters = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_batch_thread, batch) for batch in batches]
        
        for future in as_completed(futures):
            counter, idx = future.result()
            all_counters.append(counter)
    
    thread_time = time.time() - start
    thread_rate = num_sentences / thread_time
    thread_speedup = sequential_time / thread_time
    print(f"Time: {thread_time:.2f}s")
    print(f"Rate: {thread_rate:.0f} sentences/sec")
    print(f"Speedup vs sequential: {thread_speedup:.2f}x")
    print(f"Efficiency: {(thread_speedup / num_workers * 100):.1f}%")
    
    if thread_speedup < 1.2:
        print("⚠️  WARNING: ThreadPoolExecutor shows minimal speedup (GIL limitation)")
    print()
    
    # Test 3: ProcessPoolExecutor
    print("=" * 60)
    print("TEST 3: ProcessPoolExecutor (True Parallelism)")
    print("=" * 60)
    start = time.time()
    
    all_counters = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_batch_process, batch) for batch in batches]
        
        completed = 0
        for future in as_completed(futures):
            counter, idx = future.result()
            all_counters.append(counter)
            completed += 1
            if completed % 10 == 0:
                elapsed = time.time() - start
                rate = (completed * batch_size) / elapsed if elapsed > 0 else 0
                print(f"  Progress: {completed}/{len(batches)} batches | {rate:.0f} sent/sec")
    
    process_time = time.time() - start
    process_rate = num_sentences / process_time
    process_speedup = sequential_time / process_time
    print(f"\nTime: {process_time:.2f}s")
    print(f"Rate: {process_rate:.0f} sentences/sec")
    print(f"Speedup vs sequential: {process_speedup:.2f}x")
    print(f"Speedup vs threads: {thread_time / process_time:.2f}x")
    print(f"Efficiency: {(process_speedup / num_workers * 100):.1f}%")
    
    if process_speedup < 2:
        print("⚠️  WARNING: ProcessPoolExecutor not achieving good parallelism")
        print("    This could be due to:")
        print("    - Small dataset (overhead dominates)")
        print("    - Pickle overhead on Windows")
        print("    - System resource constraints")
    elif process_speedup >= num_workers * 0.7:
        print("✅ EXCELLENT: Near-linear speedup achieved!")
    elif process_speedup >= num_workers * 0.5:
        print("✅ GOOD: Decent parallelism achieved")
    else:
        print("⚠️  FAIR: Some parallelism, but room for improvement")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Sequential:  {sequential_rate:>8.0f} sent/sec (1.00x baseline)")
    print(f"Threads:     {thread_rate:>8.0f} sent/sec ({thread_speedup:.2f}x speedup)")
    print(f"Processes:   {process_rate:>8.0f} sent/sec ({process_speedup:.2f}x speedup)")
    print()
    
    # Recommendation
    if process_speedup > thread_speedup * 1.5:
        print("✅ RECOMMENDATION: Your system benefits from ProcessPoolExecutor")
        print("   GlossaryManager should use ProcessPoolExecutor for best performance")
    else:
        print("⚠️  RECOMMENDATION: Limited benefit from multiprocessing on this system")
        print("   Consider using ThreadPoolExecutor or investigate system constraints")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    # Test with different configurations
    print("\nThis test will help diagnose parallel processing performance.\n")
    
    # Auto-detect workers
    cpu_count = os.cpu_count() or 4
    test_workers = min(cpu_count, 12)
    
    print(f"Detected {cpu_count} CPU cores")
    print(f"Will test with {test_workers} workers\n")
    
    # Run test with moderate dataset size (similar to what you'd encounter)
    test_parallel_processing(num_sentences=50000, num_workers=test_workers)
    
    print("\nTest complete! Check the results above to see if true parallelism is working.")
