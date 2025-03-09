import streamlit as st
import os
from deepface import DeepFace

# Create photo library structure
PHOTO_LIBRARY = "photo_library"
os.makedirs(PHOTO_LIBRARY, exist_ok=True)

# Sidebar navigation
section = st.sidebar.radio("App Sections", ["Library", "Identification", "Results"])

if section == "Library":
    st.header("📚 Photo Library Management")
    person_name = st.text_input("Enter person's name for uploads")
    
    # Multi-file upload with validation
    uploaded_files = st.file_uploader("Upload photos (JPEG/PNG)", 
                                    accept_multiple_files=True,
                                    type=["jpg", "jpeg", "png"])
    
    if uploaded_files and person_name:
        person_dir = os.path.join(PHOTO_LIBRARY, person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        for idx, file in enumerate(uploaded_files):
            # Save with original filename
            save_path = os.path.join(person_dir, file.name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
            
            # Verify face detection
            try:
                DeepFace.extract_faces(save_path)
                st.success(f"Saved: {file.name}")
            except ValueError:
                st.error(f"No face detected in {file.name}")
                os.remove(save_path)
