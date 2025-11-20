import re

content = """<p"N-no, that's—uh, not the point! Don't you get what matters?"</p>"""
content_lower = content.lower()

print(f"Content lower: {content_lower}")

pattern = r'<p\"[^>]*'
matches = re.findall(pattern, content_lower)
print(f"Matches for {pattern}: {matches}")
print(f"Count: {len(matches)}")

pattern2 = r'(?:^|>)\s*p>[^<]*</p>'
content2 = "p>Test</p>"
matches2 = re.findall(pattern2, content2.lower())
print(f"Matches for {pattern2}: {matches2}")
print(f"Count 2: {len(matches2)}")
