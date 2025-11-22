"""Hugging Face Space entrypoint.

This file re-uses the existing `main.py` dashboard implementation and
invokes the same launch sequence. The Space will execute this file to
start the Gradio app.
"""

from main import main


def run():
    # Call the project's main launcher which constructs and starts the
    # Gradio Blocks dashboard.
    main()


if __name__ == "__main__":
    run()
