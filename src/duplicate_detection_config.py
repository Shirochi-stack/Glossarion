# duplicate_detection_config.py
"""
Configuration helper for duplicate detection algorithms.
Maps user selection to actual algorithm settings.

Performance note:
This module is used in tight loops during glossary deduplication. Optional dependencies
are imported once at module import time to avoid repeated ImportError exceptions.
"""
import os

# Optional dependencies (imported once to avoid per-comparison ImportError overhead)
try:
    from rapidfuzz import fuzz as _rf_fuzz  # type: ignore
    _HAS_RAPIDFUZZ = True
except (ImportError, OSError):
    _rf_fuzz = None
    _HAS_RAPIDFUZZ = False

try:
    import jellyfish as _jellyfish  # type: ignore
    _HAS_JELLYFISH = True
except (ImportError, OSError):
    _jellyfish = None
    _HAS_JELLYFISH = False


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
    partial_weight = float(os.getenv('GLOSSARY_PARTIAL_RATIO_WEIGHT', '0.45') or 0.45)
    partial_weight = max(0.0, min(1.0, partial_weight))
    
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
    
    config = configs.get(selected, configs['auto'])

    # Filter out algorithms that aren't available in this environment.
    # This avoids misleading config and prevents slow per-comparison ImportError handling.
    algorithms = list(config.get('algorithms', []))
    # Disable partial-ratio when weight is 0
    if partial_weight <= 0:
        algorithms = [a for a in algorithms if a != 'partial']
    if not _HAS_RAPIDFUZZ:
        algorithms = [a for a in algorithms if a not in ('token_sort', 'partial')]
    if not _HAS_JELLYFISH:
        algorithms = [a for a in algorithms if a != 'jaro_winkler']
    if not algorithms:
        algorithms = ['basic']

    # Return a copy so callers can safely mutate without affecting defaults.
    resolved = dict(config)
    resolved['algorithms'] = algorithms
    resolved['partial_ratio_weight'] = partial_weight
    return resolved


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

    # Normalize once (hot path)
    n1 = str(name1)
    n2 = str(name2)
    n1_lower = n1.lower()
    n2_lower = n2.lower()

    # Quick exact match
    if n1_lower == n2_lower:
        return 1.0

    # Get config if not provided
    if config is None:
        config = get_duplicate_detection_config()

    algorithms = config.get('algorithms', [])
    partial_weight = max(0.0, min(1.0, config.get('partial_ratio_weight', 1.0)))
    best = 0.0

    # Basic ratio (Levenshtein-like)
    if 'basic' in algorithms:
        if _HAS_RAPIDFUZZ:
            best = max(best, _rf_fuzz.ratio(n1_lower, n2_lower) / 100.0)
        else:
            from difflib import SequenceMatcher
            best = max(best, SequenceMatcher(None, n1_lower, n2_lower).ratio())

    # Token sort (word order insensitive) + partial ratio (substring matching)
    if _HAS_RAPIDFUZZ:
        if 'token_sort' in algorithms:
            best = max(best, _rf_fuzz.token_sort_ratio(n1_lower, n2_lower) / 100.0)
        if 'partial' in algorithms and partial_weight > 0:
            partial_score = _rf_fuzz.partial_ratio(n1_lower, n2_lower) / 100.0
            best = max(best, partial_score * partial_weight)

    # Jaro-Winkler (designed for names)
    if 'jaro_winkler' in algorithms and _HAS_JELLYFISH:
        best = max(best, _jellyfish.jaro_winkler_similarity(n1, n2))

    return best


def get_algorithm_display_info():
    """Get information about which algorithms are actually available."""
    available = []

    if _HAS_RAPIDFUZZ:
        available.append("RapidFuzz")
    else:
        available.append("difflib (fallback)")

    if _HAS_JELLYFISH:
        available.append("Jaro-Winkler")

    return available


if __name__ == "__main__":
    from shutdown_utils import run_cli_main
    def _main():
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
        return 0
    run_cli_main(_main)
