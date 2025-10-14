import os

# Model configuration
MODEL_PATH = './models'  # Change this to your actual model directory
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH, exist_ok=True)

# Check if model files exist in current directory as fallback
CURRENT_DIR_FILES = os.listdir('.')
MODEL_FILES = [f for f in CURRENT_DIR_FILES if f.endswith(('.safetensors', '.bin', '.json'))]

if MODEL_FILES and not any(os.path.exists(os.path.join(MODEL_PATH, f)) for f in MODEL_FILES):
    print(f"📁 Model files found in current directory: {MODEL_FILES}")
    MODEL_PATH = '.'  # Use current directory if model files are here