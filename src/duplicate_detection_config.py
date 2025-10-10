# duplicate_detection_config.py
"""
Configuration helper for duplicate detection algorithms.
Maps user selection to actual algorithm settings.
"""
import os

def get_duplicate_detection_config():
    """
    Get the duplicate detection configuration based on environment variable.
    
    Returns:
        dict: Configuration with 'algorithms', 'threshold', and 'description'
    """
    # Get selected algorithm from environment (set by GUI)
    selected = os.getenv('GLOSSARY_DUPLICATE_ALGORITHM', 'auto').lower()
    
    # Get base threshold from slider (can be overridden by preset)
    base_threshold = float(os.getenv('GLOSSARY_FUZZY_THRESHOLD', '0.90'))
    
    configs = {
        'auto': {
            'algorithms': ['basic', 'token_sort', 'partial', 'jaro_winkler'],
            'threshold': base_threshold,
            'description': 'Auto - Uses all algorithms, best score wins',
            'adjust_threshold': False  # Don't override user threshold
        },
        'strict': {
            'algorithms': ['basic'],
            'threshold': max(base_threshold, 0.95),  # Enforce minimum 95%
            'description': 'Strict - High precision, minimal merging',
            'adjust_threshold': True  # Override to at least 95%
        },
        'balanced': {
            'algorithms': ['token_sort', 'partial'],
            'threshold': base_threshold,
            'description': 'Balanced - Token + Partial matching',
            'adjust_threshold': False
        },
        'aggressive': {
            'algorithms': ['basic', 'token_sort', 'partial', 'jaro_winkler'],
            'threshold': min(base_threshold, 0.80),  # Lower to at most 80%
            'description': 'Aggressive - Maximum duplicate detection',
            'adjust_threshold': True  # Override to at most 80%
        },
        'basic': {
            'algorithms': ['basic'],
            'threshold': base_threshold,
            'description': 'Basic - Simple Levenshtein distance',
            'adjust_threshold': False
        }
    }
    
    return configs.get(selected, configs['auto'])


def calculate_similarity_with_config(name1, name2, config=None):
    """
    Calculate similarity between two names using configured algorithms.
    
    Args:
        name1: First name
        name2: Second name
        config: Configuration dict (from get_duplicate_detection_config())
    
    Returns:
        float: Best similarity score (0.0-1.0)
    """
    if not name1 or not name2:
        return 0.0
    
    # Quick exact match
    if name1.lower() == name2.lower():
        return 1.0
    
    # Get config if not provided
    if config is None:
        config = get_duplicate_detection_config()
    
    algorithms = config['algorithms']
    scores = []
    
    # Basic ratio (Levenshtein)
    if 'basic' in algorithms:
        try:
            from rapidfuzz import fuzz
            score = fuzz.ratio(name1.lower(), name2.lower()) / 100.0
            scores.append(score)
        except ImportError:
            from difflib import SequenceMatcher
            score = SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
            scores.append(score)
    
    # Token sort (word order insensitive)
    if 'token_sort' in algorithms:
        try:
            from rapidfuzz import fuzz
            score = fuzz.token_sort_ratio(name1.lower(), name2.lower()) / 100.0
            scores.append(score)
        except (ImportError, AttributeError):
            pass
    
    # Partial ratio (substring matching)
    if 'partial' in algorithms:
        try:
            from rapidfuzz import fuzz
            score = fuzz.partial_ratio(name1.lower(), name2.lower()) / 100.0
            scores.append(score)
        except (ImportError, AttributeError):
            pass
    
    # Jaro-Winkler (designed for names)
    if 'jaro_winkler' in algorithms:
        try:
            import jellyfish
            score = jellyfish.jaro_winkler_similarity(name1, name2)
            scores.append(score)
        except ImportError:
            pass
    
    # Return best score
    return max(scores) if scores else 0.0


def get_algorithm_display_info():
    """Get information about which algorithms are actually available."""
    available = []
    
    try:
        import rapidfuzz
        available.append("RapidFuzz")
    except ImportError:
        available.append("difflib (fallback)")
    
    try:
        import jellyfish
        available.append("Jaro-Winkler")
    except ImportError:
        pass
    
    return available


if __name__ == "__main__":
    # Test the configuration
    import os
    
    print("Testing duplicate detection configuration...\n")
    
    # Test all modes
    modes = ['auto', 'strict', 'balanced', 'aggressive', 'basic']
    
    for mode in modes:
        os.environ['GLOSSARY_DUPLICATE_ALGORITHM'] = mode
        os.environ['GLOSSARY_FUZZY_THRESHOLD'] = '0.90'
        
        config = get_duplicate_detection_config()
        print(f"{mode.upper()}:")
        print(f"  Description: {config['description']}")
        print(f"  Algorithms: {', '.join(config['algorithms'])}")
        print(f"  Threshold: {config['threshold']:.2f}")
        print()
    
    # Test similarity calculation
    print("\nTesting similarity calculations (AUTO mode):")
    os.environ['GLOSSARY_DUPLICATE_ALGORITHM'] = 'auto'
    
    test_pairs = [
        ("Kim Sang-hyun", "Kim Sanghyun"),
        ("Park Ji-sung", "Ji-sung Park"),
        ("김상현", "김상현님"),
        ("Catherine", "Katherine"),
    ]
    
    for name1, name2 in test_pairs:
        score = calculate_similarity_with_config(name1, name2)
        status = "✓ MATCH" if score >= 0.90 else "✗ DIFFERENT"
        print(f"{status} '{name1}' vs '{name2}': {score:.3f}")
    
    print(f"\nAvailable algorithms: {', '.join(get_algorithm_display_info())}")
