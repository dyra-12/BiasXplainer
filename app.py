"""Hugging Face Space entrypoint for BiasGuard Pro.

Exposes a `demo` Gradio Blocks instance so Spaces auto-detects and can
serve it. When executed directly, launches the app. This avoids using
`share=True` inside Spaces (unnecessary) and reuses the dashboard builder.
"""

from main import BiasGuardDashboard

# Construct the dashboard once; Spaces will look for a top-level `demo`.
dashboard = BiasGuardDashboard()
demo = dashboard.create_dashboard()

if __name__ == "__main__":
    # Local execution path.
    demo.launch(server_name="0.0.0.0", show_error=True, debug=False)
