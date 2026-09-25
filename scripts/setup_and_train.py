#!/usr/bin/env python3
"""
Setup script to install dependencies and train the model
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def train_model():
    """Train the machine learning model"""
    print("\nTraining the loan approval model...")
    try:
        # Import and run the training script
        exec(open('train_model.py').read())
        print("✅ Model trained and saved successfully!")
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return False
    return True

def main():
    """Main setup function"""
    print("🏦 Loan Approval System Setup")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists('data.csv'):
        print("❌ data.csv not found. Make sure you're in the correct directory.")
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Train model
    if not train_model():
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\nTo run the application:")
    print("python app.py")
    print("\nThen open your browser to: http://localhost:5000")

if __name__ == "__main__":
    main()
