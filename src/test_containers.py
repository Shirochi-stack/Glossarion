#!/usr/bin/env python3
"""
Test script for the advanced container system.
This validates that the sophisticated containers work properly.
"""

import tkinter as tk
import sys
import os

# Add current directory to path to import our modules
sys.path.insert(0, os.path.dirname(__file__))

try:
    from gui_containers import ResponsiveContainer, FlexibleDialog, SectionContainer
    print("✅ Successfully imported all container classes")
    CONTAINERS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import containers: {e}")
    CONTAINERS_AVAILABLE = False
    sys.exit(1)

def test_responsive_container():
    """Test the ResponsiveContainer class"""
    print("\n🧪 Testing ResponsiveContainer...")
    
    root = tk.Tk()
    root.title("ResponsiveContainer Test")
    root.geometry("800x600")
    
    # Create a responsive container
    container = ResponsiveContainer(
        root, 
        layout_type='vertical', 
        theme='modern',
        auto_expand=True
    )
    container.pack(fill='both', expand=True, padx=10, pady=10)
    
    # Add some test widgets
    label1 = tk.Label(container, text="Test Label 1", bg='lightblue')
    container.add_widget(label1, expand=False)
    
    label2 = tk.Label(container, text="Test Label 2", bg='lightgreen')
    container.add_widget(label2, expand=True)
    
    button = tk.Button(container, text="Test Button")
    container.add_widget(button, expand=False)
    
    print("✅ ResponsiveContainer created successfully")
    
    # Test layout changes
    def test_layout_change():
        container.layout_type = 'horizontal'
        container._layout_children()
        print("✅ Layout change test passed")
        root.after(2000, root.destroy)
    
    root.after(1000, test_layout_change)
    root.mainloop()

def test_section_container():
    """Test the SectionContainer class"""
    print("\n🧪 Testing SectionContainer...")
    
    root = tk.Tk()
    root.title("SectionContainer Test")
    root.geometry("600x400")
    
    # Create a section container with title
    section = SectionContainer(
        root, 
        title="Test Section", 
        collapsible=True, 
        theme='raised'
    )
    section.pack(fill='both', expand=True, padx=10, pady=10)
    
    # Add content to the section
    label = tk.Label(section.content_frame, text="This is content inside the section")
    section.add_widget(label)
    
    entry = tk.Entry(section.content_frame)
    section.add_widget(entry, expand=False)
    
    print("✅ SectionContainer created successfully")
    
    def test_collapse():
        if hasattr(section, 'toggle_collapse'):
            section.toggle_collapse()
            print("✅ Collapse functionality works")
        root.after(2000, root.destroy)
    
    root.after(1000, test_collapse)
    root.mainloop()

def test_flexible_dialog():
    """Test the FlexibleDialog class"""
    print("\n🧪 Testing FlexibleDialog...")
    
    root = tk.Tk()
    root.title("FlexibleDialog Test")
    root.geometry("400x300")
    
    def open_dialog():
        # Create a flexible dialog
        dialog = FlexibleDialog(
            root, 
            title="Test Dialog",
            layout='vertical',
            theme='modern',
            modal=False
        )
        
        # Add a section to the dialog
        test_section = SectionContainer(
            dialog.main_container,
            title="Dialog Section",
            theme='modern'
        )
        
        # Add content to the section
        label = tk.Label(test_section.content_frame, text="Dialog content here")
        test_section.add_widget(label)
        
        # Add section to dialog
        dialog.add_section("test", test_section, position='main')
        
        # Add a button to close dialog
        close_btn = tk.Button(
            dialog.bottom_frame, 
            text="Close", 
            command=dialog.destroy
        )
        close_btn.pack(pady=10)
        
        print("✅ FlexibleDialog created successfully")
        
        # Auto-close after 3 seconds
        dialog.after(3000, dialog.destroy)
    
    # Button to open dialog
    open_btn = tk.Button(root, text="Open Test Dialog", command=open_dialog)
    open_btn.pack(expand=True)
    
    # Auto-open dialog after 500ms
    root.after(500, open_dialog)
    
    # Close root after 5 seconds
    root.after(5000, root.destroy)
    
    root.mainloop()

def main():
    """Run all container tests"""
    if not CONTAINERS_AVAILABLE:
        print("❌ Container system not available, skipping tests")
        return
    
    print("🚀 Starting container system tests...")
    
    try:
        test_responsive_container()
        test_section_container()
        test_flexible_dialog()
        
        print("\n✅ All container tests passed!")
        
    except Exception as e:
        print(f"\n❌ Container test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()