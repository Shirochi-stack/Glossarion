#!/usr/bin/env python3
"""Test the actual Gradio app to see what's happening"""

from app import GlossarionWeb
import threading
import time

def test_gradio_app():
    print("Starting actual Gradio app test...")
    
    # Create the app
    web_app = GlossarionWeb()
    app = web_app.create_interface()
    
    print(f"Initial config model: {web_app.get_config_value('model')}")
    print(f"Initial config batch_size: {web_app.get_config_value('batch_size')}")
    
    # Get the components we need to test
    # The app has many components, let's find the Settings tab components
    
    # Start the app in a thread so we can interact with it
    def run_app():
        app.launch(server_name="127.0.0.1", server_port=7861, share=False, show_error=True, prevent_thread_lock=True)
    
    thread = threading.Thread(target=run_app, daemon=True)
    thread.start()
    
    print("App started on http://127.0.0.1:7861")
    print("Waiting 5 seconds for server to start...")
    time.sleep(5)
    
    # The app is running, but we need to test the actual handlers
    # Let's check what happens when we call the handlers directly
    
    print("\nTesting manga model save handler...")
    # Find the actual handler function that's bound to manga_model.change
    # It should be one of the save functions
    
    print("\nPress Ctrl+C to stop the test")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping test...")

if __name__ == "__main__":
    test_gradio_app()