import os
import re

src_dir = r'C:\Users\omarn\Projects\Glossarion\src'
results = {}

for f in os.listdir(src_dir):
    if f.endswith('.py'):
        fp = os.path.join(src_dir, f)
        try:
            content = open(fp, 'r', encoding='utf-8').read()
            count = len(re.findall(r'\.(get|set)\(\)', content))
            if count > 0:
                results[f] = count
        except:
            pass

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
print('Files with .get()/.set() calls (Tkinter variables):')
print('='*60)
for fname, count in sorted_results[:20]:
    print(f'{fname}: {count} occurrences')
