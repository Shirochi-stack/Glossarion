import sys, os
sys.path.insert(0, r'c:\Users\ADMIN\Dev\Glossarion\src')
from epub_converter import EPUBCompiler

class MockCompiler(EPUBCompiler):
    def __init__(self): pass

ep = MockCompiler()
ep.log = lambda x: None
reused, rem = ep._cross_reference_from_other_file(
    {159: '시즌2 - 주변에 여자들이 너무 많음'}, 
    r'C:\Users\ADMIN\Documents\korean translation\[393761] 쫓아냈던 노예들이 강해져서 돌아왔다\TOC.txt', 
    'header'
)
print('Reused:', reused)
print('Remaining:', rem)
