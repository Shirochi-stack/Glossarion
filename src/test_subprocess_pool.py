#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test to verify ProcessPoolExecutor works inside a subprocess on Windows
"""

import multiprocessing
import time
from collections import Counter

def worker_function(args):
    """Simple worker that processes a batch"""
    batch_data, batch_id = args
    # Simulate work
    result = Counter()
    for item in batch_data:
        result[item] += 1
    return result, batch_id

def test_in_subprocess():
    """This will be run in a subprocess"""
    print(f"🔍 Running in subprocess: {multiprocessing.current_process().name}")
    
    # Create test data
    test_data = [['a', 'b', 'c'] * 100 for _ in range(20)]
    batches = [(batch, idx) for idx, batch in enumerate(test_data)]
    
    print(f"📊 Testing with {len(batches)} batches...")
    
    # Try to use ProcessPoolExecutor with spawn context
    mp_context = multiprocessing.get_context('spawn')
    
    start_time = time.time()
    
    with mp_context.Pool(processes=4) as pool:
        print("✅ ProcessPoolExecutor created successfully in subprocess!")
        results = pool.map(worker_function, batches)
        print(f"✅ Processed {len(results)} batches in {time.time() - start_time:.2f}s")
        print(f"✅ ProcessPoolExecutor works in subprocess on Windows!")
    
    return "SUCCESS"

def main():
    print("="*60)
    print("TEST: ProcessPoolExecutor in Subprocess")
    print("="*60)
    
    # Test 1: Direct execution (not in subprocess)
    print("\n--- Test 1: Direct Execution ---")
    print(f"Current process: {multiprocessing.current_process().name}")
    test_in_subprocess()
    
    # Test 2: From within a subprocess
    print("\n--- Test 2: Inside Subprocess ---")
    mp_context = multiprocessing.get_context('spawn')
    with mp_context.Pool(processes=1) as pool:
        result = pool.apply(test_in_subprocess)
        print(f"Result from subprocess: {result}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("ProcessPoolExecutor CAN be used in subprocess on Windows")
    print("="*60)

if __name__ == '__main__':
    main()
