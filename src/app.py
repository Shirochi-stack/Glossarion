#!/usr/bin/env python3
"""
Glossarion Hugging Face Spaces Entry Point
Launches the full Glossarion Web Interface
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Main entry point for Hugging Face Spaces"""
    
    # Set environment variables for Hugging Face Spaces
    os.environ['HF_SPACES'] = 'true'
    os.environ['GRADIO_SHARE'] = 'false'
    
    try:
        # Import and launch the existing Glossarion Web interface
        from glossarion_web import GlossarionWeb
        
        print("🚀 Starting Glossarion Web Interface on Hugging Face Spaces...")
        
        web_app = GlossarionWeb()
        app = web_app.create_interface()
        
        # Launch with Spaces-appropriate settings
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            favicon_path=None  # Skip favicon for Spaces
        )
        
    except Exception as e:
        print(f"❌ Error launching Glossarion Web: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: show error message
        import gradio as gr
        
        def error_interface():
            with gr.Blocks(title="Glossarion - Error") as error_app:
                gr.HTML(f"""
                <div style="text-align: center; padding: 50px;">
                    <h1>❌ Glossarion Failed to Load</h1>
                    <p><strong>Error:</strong> {str(e)}</p>
                    <p>Please check the logs for more details.</p>
                    <p>You may need to install missing dependencies or check module imports.</p>
                </div>
                """)
            return error_app
        
        error_app = error_interface()
        error_app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )

if __name__ == "__main__":
    main()