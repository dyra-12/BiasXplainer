#!/usr/bin/env python3
"""
Setup script for BiasGuard Pro
"""

import os
import sys


def setup_environment():
    """Setup the environment and check dependencies"""
    print("🔧 Setting up BiasGuard Pro...")

    # Check if we're in the right directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")

    # List files to help debug
    print("\n📁 Files in current directory:")
    for file in os.listdir("."):
        if file.endswith((".py", ".json", ".txt", ".safetensors", ".bin")):
            print(f"  - {file}")

    # Check for model files
    model_files = [
        f for f in os.listdir(".") if f.endswith((".safetensors", ".bin", ".json"))
    ]
    if model_files:
        print(f"\n✅ Found model files: {model_files}")
    else:
        print("\n⚠️  No model files found in current directory")
        print("The system will use a default model instead.")

    print("\n🚀 Setup complete! You can now run:")
    print("   python main.py")


if __name__ == "__main__":
    setup_environment()
