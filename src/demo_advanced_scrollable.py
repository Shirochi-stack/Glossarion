#!/usr/bin/env python3
"""
Demo of the enhanced setup_scrollable_advanced method.
This shows how to use the new sophisticated container system for scrollable dialogs.
"""

import tkinter as tk
from tkinter import ttk
import sys
import os

# Add current directory to path to import our modules
sys.path.insert(0, os.path.dirname(__file__))

# Import the translator GUI components
try:
    from translator_gui import WindowManager
    print("✅ Successfully imported WindowManager")
except ImportError as e:
    print(f"❌ Failed to import WindowManager: {e}")
    sys.exit(1)

def demo_basic_advanced_scrollable():
    """Demo basic advanced scrollable dialog"""
    print("\n🚀 Demo: Basic Advanced Scrollable Dialog")
    
    root = tk.Tk()
    root.title("Advanced Scrollable Demo")
    root.geometry("600x400")
    
    # Create window manager
    base_dir = os.path.dirname(__file__)
    wm = WindowManager(base_dir)
    
    def open_advanced_dialog():
        # Create advanced scrollable dialog
        dialog, main_container, fixed_containers = wm.setup_scrollable_advanced(
            root,
            title="Advanced Settings Dialog",
            layout='vertical',
            theme='modern',
            responsive=True
        )
        
        # Add content to the main scrollable container
        for i in range(20):
            section = wm.create_section_container(
                main_container,
                title=f"Settings Section {i + 1}",
                theme='raised'
            )
            
            # Add content to each section
            label = tk.Label(section.content_frame, text=f"Configuration options for section {i + 1}")
            if hasattr(section, 'add_widget'):
                section.add_widget(label)
            else:
                label.pack(padx=10, pady=5)
            
            entry = tk.Entry(section.content_frame, width=40)
            if hasattr(section, 'add_widget'):
                section.add_widget(entry, expand=False)
            else:
                entry.pack(padx=10, pady=5)
            
            # Add section to main container
            if hasattr(main_container, 'add_widget'):
                main_container.add_widget(section, expand=False)
            else:
                section.pack(fill='x', padx=10, pady=5)
        
        print("✅ Advanced scrollable dialog created")
    
    # Button to open dialog
    open_btn = tk.Button(root, text="Open Advanced Scrollable Dialog", command=open_advanced_dialog)
    open_btn.pack(expand=True)
    
    # Auto-open after 1 second
    root.after(1000, open_advanced_dialog)
    
    root.mainloop()

def demo_fixed_sections_dialog():
    """Demo advanced scrollable with fixed sections"""
    print("\n🚀 Demo: Advanced Scrollable with Fixed Sections")
    
    root = tk.Tk()
    root.title("Fixed Sections Demo")
    root.geometry("700x500")
    
    # Create window manager
    base_dir = os.path.dirname(__file__)
    wm = WindowManager(base_dir)
    
    def open_fixed_sections_dialog():
        # Create advanced scrollable dialog with fixed sections
        dialog, main_container, fixed_containers = wm.setup_scrollable_advanced(
            root,
            title="Configuration with Fixed Buttons",
            layout='vertical',
            theme='modern',
            fixed_sections=['buttons'],  # Create a fixed button section
            responsive=True
        )
        
        # Add scrollable content
        for i in range(15):
            section = wm.create_section_container(
                main_container,
                title=f"Scrollable Section {i + 1}",
                theme='modern',
                collapsible=True
            )
            
            # Add content
            text = f"This is scrollable content in section {i + 1}. " * 3
            label = tk.Label(section.content_frame, text=text, wraplength=400, justify='left')
            if hasattr(section, 'add_widget'):
                section.add_widget(label)
            else:
                label.pack(padx=10, pady=5)
            
            # Add to main container
            if hasattr(main_container, 'add_widget'):
                main_container.add_widget(section, expand=False)
            else:
                section.pack(fill='x', padx=10, pady=5)
        
        # Add buttons to fixed section
        if 'buttons' in fixed_containers:
            button_frame = fixed_containers['buttons']
            
            save_btn = tk.Button(button_frame, text="Save Settings")
            save_btn.pack(side='left', padx=10, pady=10)
            
            cancel_btn = tk.Button(button_frame, text="Cancel", command=dialog.destroy)
            cancel_btn.pack(side='left', padx=10, pady=10)
            
            reset_btn = tk.Button(button_frame, text="Reset to Defaults")
            reset_btn.pack(side='right', padx=10, pady=10)
        
        print("✅ Fixed sections dialog created")
    
    # Button to open dialog
    open_btn = tk.Button(root, text="Open Fixed Sections Dialog", command=open_fixed_sections_dialog)
    open_btn.pack(expand=True)
    
    # Auto-open after 1 second
    root.after(1000, open_fixed_sections_dialog)
    
    root.mainloop()

def demo_adaptive_layout():
    """Demo adaptive layout that changes based on dialog size"""
    print("\n🚀 Demo: Adaptive Layout Dialog")
    
    root = tk.Tk()
    root.title("Adaptive Layout Demo")
    root.geometry("800x600")
    
    # Create window manager
    base_dir = os.path.dirname(__file__)
    wm = WindowManager(base_dir)
    
    def open_adaptive_dialog():
        # Create adaptive dialog
        dialog, main_container, fixed_containers = wm.setup_scrollable_advanced(
            root,
            title="Adaptive Layout Dialog (Resize Me!)",
            layout='adaptive',  # This will change based on size
            theme='modern',
            responsive=True
        )
        
        # Add content that will adapt to different layouts
        for i in range(12):
            card = tk.Frame(main_container, bg='lightblue', relief='raised', bd=1)
            label = tk.Label(card, text=f"Card {i + 1}", bg='lightblue', font=('Arial', 10, 'bold'))
            label.pack(pady=10)
            
            desc = tk.Label(card, text="This card will rearrange\nbased on dialog size", 
                          bg='lightblue', justify='center')
            desc.pack(pady=5)
            
            if hasattr(main_container, 'add_widget'):
                main_container.add_widget(card, expand=False)
            else:
                card.pack(fill='x', padx=5, pady=5)
        
        # Add instruction label
        info_label = tk.Label(
            dialog.bottom_frame, 
            text="💡 Try resizing this dialog to see the adaptive layout in action!",
            fg='blue'
        )
        info_label.pack(pady=10)
        
        print("✅ Adaptive layout dialog created")
    
    # Button to open dialog
    open_btn = tk.Button(root, text="Open Adaptive Layout Dialog", command=open_adaptive_dialog)
    open_btn.pack(expand=True)
    
    # Auto-open after 1 second
    root.after(1000, open_adaptive_dialog)
    
    root.mainloop()

def main():
    """Run all demos"""
    print("🚀 Starting Advanced Scrollable Dialog Demos...")
    print("Each demo will run sequentially. Close the window to proceed to the next demo.")
    
    try:
        demo_basic_advanced_scrollable()
        demo_fixed_sections_dialog()
        demo_adaptive_layout()
        
        print("\n✅ All demos completed successfully!")
        print("\n📖 Summary:")
        print("   - Basic advanced scrollable: Smooth scrolling with modern containers")
        print("   - Fixed sections: Scrollable content with fixed button areas")
        print("   - Adaptive layout: Automatically adjusts based on dialog size")
        print("\n🎯 The enhanced setup_scrollable_advanced method provides:")
        print("   ✅ Sophisticated container system")
        print("   ✅ Responsive layouts")
        print("   ✅ Fixed and scrollable sections")
        print("   ✅ Advanced scrolling with acceleration")
        print("   ✅ Theme support")
        print("   ✅ Automatic layout adaptation")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
