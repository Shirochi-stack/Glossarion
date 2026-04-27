import re, sys
sys.stdout.reconfigure(encoding='utf-8')

esc_re = re.compile(r'\\+[a-zA-Z]\[\d+\]|\\+[a-zA-Z](?![a-zA-Z])|\\+[{}GS$.|!><^]')

def is_escape_only(text):
    if not text or not text.strip():
        return True
    stripped = esc_re.sub('', text)
    return not stripped.strip()

# Test cases
tests = [
    ('\\c', True, 'bare color reset'),
    ('\\n', True, 'bare newline'),
    ('\\c[27]', True, 'color code'),
    ('\\n[1]', True, 'name reference'),
    ('\\c[27]\\c[0]', True, 'multiple codes'),
    ('\\c[27]Hello\\c[0]', False, 'code + text'),
    ('Hello world', False, 'normal text'),
    ('Cowgirl position', False, 'normal English'),
    ('お願いする', False, 'normal Japanese'),
    ('\\n[1] greeted the student.', False, 'code prefix with text'),
    ('', True, 'empty'),
    ('   ', True, 'whitespace'),
]

print("=== Validation filter tests ===")
all_pass = True
for text, expected, desc in tests:
    result = is_escape_only(text)
    status = "✅" if result == expected else "❌ FAIL"
    if result != expected:
        all_pass = False
    print(f"  {status} '{text}' -> escape_only={result} (expected {expected}) [{desc}]")

print(f"\n{'All tests passed!' if all_pass else 'SOME TESTS FAILED!'}")
