# advanced_duplicate_detection.py
"""
Advanced duplicate detection for glossary entries.
Uses multiple algorithms and takes the best match.
"""

def get_similarity_score(name1, name2, threshold=0.90):
    """
    Calculate similarity using multiple algorithms and return the best score.
    
    Args:
        name1: First name to compare
        name2: Second name to compare
        threshold: Minimum similarity threshold (0.0-1.0)
    
    Returns:
        float: Best similarity score from all algorithms (0.0-1.0)
    """
    if not name1 or not name2:
        return 0.0
    
    # Quick exact match check
    if name1.lower() == name2.lower():
        return 1.0
    
    scores = []
    
    # Try RapidFuzz first (fastest)
    try:
        from rapidfuzz import fuzz
        
        # Basic ratio
        ratio = fuzz.ratio(name1.lower(), name2.lower()) / 100.0
        scores.append(ratio)
        
        # Token sort (handles word order)
        token_sort = fuzz.token_sort_ratio(name1.lower(), name2.lower()) / 100.0
        scores.append(token_sort)
        
        # Partial ratio (substring matching)
        partial = fuzz.partial_ratio(name1.lower(), name2.lower()) / 100.0
        scores.append(partial)
        
    except ImportError:
        pass
    
    # Try TheFuzz/FuzzyWuzzy (more sophisticated)
    try:
        from thefuzz import fuzz as tfuzz
        
        # Token set ratio (best for name variations)
        token_set = tfuzz.token_set_ratio(name1, name2) / 100.0
        scores.append(token_set)
        
    except ImportError:
        pass
    
    # Try Jellyfish (phonetic matching for names)
    try:
        import jellyfish
        
        # Jaro-Winkler (designed for names, prioritizes prefix matches)
        jaro = jellyfish.jaro_winkler_similarity(name1, name2)
        scores.append(jaro)
        
    except ImportError:
        pass
    
    # Try TextDistance (additional algorithms)
    try:
        import textdistance
        
        # Jaro-Winkler from textdistance
        jw = textdistance.jaro_winkler.normalized_similarity(name1, name2)
        scores.append(jw)
        
        # DamerauLevenshtein (handles transpositions)
        dl = textdistance.damerau_levenshtein.normalized_similarity(name1, name2)
        scores.append(dl)
        
    except ImportError:
        pass
    
    # Fallback to difflib if no libraries available
    if not scores:
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
        scores.append(ratio)
    
    # Return the maximum score from all algorithms
    best_score = max(scores) if scores else 0.0
    
    return best_score


def find_duplicates_advanced(entries, threshold=0.90, debug=False):
    """
    Find duplicates using advanced multi-algorithm approach.
    
    Args:
        entries: List of dict entries with 'raw_name' field
        threshold: Similarity threshold (0.0-1.0)
        debug: Print debug information
    
    Returns:
        tuple: (deduplicated_entries, removed_count, duplicate_pairs)
    """
    from remove_honorifics import remove_honorifics  # Your existing function
    
    seen_entries = []  # List of (cleaned_name, original_entry)
    deduplicated = []
    duplicate_pairs = []  # Track what was merged
    removed_count = 0
    
    if debug:
        print(f"[AdvancedDedup] Processing {len(entries)} entries with threshold {threshold:.2f}")
    
    for idx, entry in enumerate(entries):
        raw_name = entry.get('raw_name', '')
        if not raw_name:
            continue
        
        # Clean the name (remove honorifics)
        cleaned_name = remove_honorifics(raw_name)
        
        # Check against all seen entries
        is_duplicate = False
        best_match_score = 0.0
        matched_with = None
        
        for seen_clean, seen_entry in seen_entries:
            # Get similarity score using multiple algorithms
            score = get_similarity_score(cleaned_name, seen_clean, threshold)
            
            if score >= threshold:
                is_duplicate = True
                if score > best_match_score:
                    best_match_score = score
                    matched_with = seen_entry.get('raw_name', '')
                break
        
        if is_duplicate:
            removed_count += 1
            duplicate_pairs.append({
                'duplicate': raw_name,
                'original': matched_with,
                'score': best_match_score
            })
            if debug and removed_count <= 10:
                print(f"[AdvancedDedup] Duplicate: '{raw_name}' matches '{matched_with}' (score: {best_match_score:.3f})")
        else:
            seen_entries.append((cleaned_name, entry))
            deduplicated.append(entry)
    
    if debug:
        print(f"[AdvancedDedup] Removed {removed_count} duplicates")
        print(f"[AdvancedDedup] Kept {len(deduplicated)} unique entries")
    
    return deduplicated, removed_count, duplicate_pairs


def get_available_algorithms():
    """Check which algorithms are available"""
    available = []
    
    try:
        import rapidfuzz
        available.append("RapidFuzz (Basic + Token)")
    except ImportError:
        pass
    
    try:
        import thefuzz
        available.append("TheFuzz (Token Set)")
    except ImportError:
        pass
    
    try:
        import jellyfish
        available.append("Jellyfish (Jaro-Winkler)")
    except ImportError:
        pass
    
    try:
        import textdistance
        available.append("TextDistance (Multiple)")
    except ImportError:
        pass
    
    if not available:
        available.append("difflib (Fallback)")
    
    return available


if __name__ == "__main__":
    from shutdown_utils import run_cli_main
    def _main():
        # Test the similarity scoring
        print("Available algorithms:")
        for algo in get_available_algorithms():
            print(f"  ✓ {algo}")
        
        print("\nTest cases:")
        test_pairs = [
            ("김상현", "김상현님"),
            ("Kim Sang-hyun", "Kim Sanghyun"),
            ("김상현", "김상혁"),
            ("Park Ji-sung", "Ji-sung Park"),
            ("田中太郎", "田中太郎さん"),
        ]
        
        for name1, name2 in test_pairs:
            score = get_similarity_score(name1, name2)
            status = "✓ MATCH" if score >= 0.90 else "✗ DIFFERENT"
            print(f"{status} '{name1}' vs '{name2}': {score:.3f}")
        return 0
    run_cli_main(_main)
