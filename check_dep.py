"""
Check if all required dependencies are installed in the current Python environment
Run this before starting the Streamlit app to verify everything is set up correctly
"""

import sys
import importlib.util

def check_module(module_name):
    """Check if a module is installed in the current environment"""
    is_installed = importlib.util.find_spec(module_name) is not None
    return is_installed

def main():
    # List of required modules
    required_modules = [
        "streamlit",
        "deepface",
        "numpy",
        "PIL",
        "tensorflow"
    ]
    
    # Check each module
    missing_modules = []
    installed_modules = []
    
    for module in required_modules:
        if check_module(module):
            installed_modules.append(module)
        else:
            missing_modules.append(module)
    
    # Print python version and environment info
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Python path: {sys.path}")
    print("\n")
    
    # Print results
    if missing_modules:
        print("❌ MISSING DEPENDENCIES")
        print("The following modules are missing from your environment:")
        for module in missing_modules:
            print(f"  - {module}")
        print("\nTo install them, run:")
        print(f"{sys.executable} -m pip install " + " ".join(missing_modules))
    else:
        print("✅ ALL DEPENDENCIES INSTALLED")
    
    # Print installed modules
    print("\nInstalled modules:")
    for module in installed_modules:
        print(f"  - {module}")
    
    # Special check for deepface
    if "deepface" in installed_modules:
        try:
            import deepface
            print(f"\nDeepFace version: {deepface.__version__}")
            
            # Test importing DeepFace from deepface
            try:
                from deepface import DeepFace
                print("✅ DeepFace successfully imported")
            except ImportError as e:
                print(f"❌ Could not import DeepFace from deepface: {e}")
        except Exception as e:
            print(f"❌ Error importing deepface: {e}")

if __name__ == "__main__":
    main()