import os
import re

def diagnose_epub_directory(directory="."):
    """Diagnose issues with EPUB output directory"""
    
    print(f"\n{'='*60}")
    print(f"EPUB Directory Diagnostic Tool")
    print(f"{'='*60}\n")
    
    # Get absolute path
    abs_path = os.path.abspath(directory)
    print(f"📁 Checking directory: {abs_path}")
    
    # Check if directory exists
    if not os.path.exists(abs_path):
        print(f"❌ ERROR: Directory does not exist!")
        return
    
    if not os.path.isdir(abs_path):
        print(f"❌ ERROR: Path is not a directory!")
        return
    
    # List contents
    try:
        contents = os.listdir(abs_path)
        print(f"✅ Directory is accessible")
        print(f"📊 Total items: {len(contents)}\n")
    except Exception as e:
        print(f"❌ ERROR: Cannot read directory: {e}")
        return
    
    # Categorize files
    html_files = []
    response_files = []
    css_files = []
    image_files = []
    directories = []
    other_files = []
    
    for item in contents:
        item_path = os.path.join(abs_path, item)
        
        if os.path.isdir(item_path):
            directories.append(item)
        elif item.endswith('.html'):
            html_files.append(item)
            if item.startswith('response_'):
                response_files.append(item)
        elif item.endswith('.css'):
            css_files.append(item)
        elif item.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.svg')):
            image_files.append(item)
        else:
            other_files.append(item)
    
    # Report findings
    print("📋 Directory Contents Summary:")
    print(f"   • HTML files: {len(html_files)}")
    print(f"   • Response files (translated chapters): {len(response_files)}")
    print(f"   • CSS files: {len(css_files)}")
    print(f"   • Image files: {len(image_files)}")
    print(f"   • Subdirectories: {len(directories)}")
    print(f"   • Other files: {len(other_files)}")
    
    # Check for required items
    print(f"\n📍 Checking Required Items:")
    
    # Check for metadata.json
    if 'metadata.json' in contents:
        print("   ✅ metadata.json found")
    else:
        print("   ❌ metadata.json NOT FOUND")
    
    # Check for response files
    if response_files:
        print(f"   ✅ {len(response_files)} translated chapter files found")
        
        # Analyze chapter numbers
        chapter_nums = []
        for f in response_files:
            m = re.match(r'response_(\d+)_', f)
            if m:
                chapter_nums.append(int(m.group(1)))
        
        if chapter_nums:
            chapter_nums.sort()
            print(f"   📖 Chapter range: {min(chapter_nums)} to {max(chapter_nums)}")
            
            # Check for missing chapters
            expected = set(range(min(chapter_nums), max(chapter_nums) + 1))
            actual = set(chapter_nums)
            missing = expected - actual
            if missing:
                print(f"   ⚠️ Missing chapters: {sorted(missing)}")
    else:
        print("   ❌ No response_*.html files found!")
        
        if html_files:
            print(f"\n   🔍 Found {len(html_files)} HTML files with different names:")
            for i, f in enumerate(html_files[:5]):
                print(f"      {i+1}. {f}")
            if len(html_files) > 5:
                print(f"      ... and {len(html_files) - 5} more")
    
    # Check subdirectories
    if directories:
        print(f"\n📂 Subdirectories found:")
        for d in directories:
            print(f"   • {d}/")
            
            # Check contents of important subdirectories
            if d in ['css', 'images', 'fonts']:
                try:
                    sub_contents = os.listdir(os.path.join(abs_path, d))
                    print(f"     Contains {len(sub_contents)} items")
                except:
                    print(f"     Cannot read contents")
    
    # Sample file check
    if response_files:
        print(f"\n🔍 Checking a sample chapter file...")
        sample_file = response_files[0]
        sample_path = os.path.join(abs_path, sample_file)
        
        try:
            with open(sample_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"   ✅ {sample_file} is readable")
                print(f"   📏 File size: {len(content):,} characters")
                
                # Check for basic HTML structure
                if '<html' in content.lower():
                    print("   ✅ Contains HTML tag")
                if '<body' in content.lower():
                    print("   ✅ Contains BODY tag")
                if '<p>' in content or '<p ' in content:
                    print("   ✅ Contains paragraph tags")
                    
        except Exception as e:
            print(f"   ❌ Cannot read {sample_file}: {e}")
    
    print(f"\n{'='*60}")
    print("Diagnostic complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    from shutdown_utils import run_cli_main
    def _main():
        import sys
        if len(sys.argv) > 1:
            diagnose_epub_directory(sys.argv[1])
        else:
            diagnose_epub_directory(".")
        return 0
    run_cli_main(_main)
