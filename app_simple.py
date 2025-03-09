"""
A simplified version of the face identification app with minimal dependencies.
Use this if you're having issues with the full app.
"""

import streamlit as st
import os
import tempfile
from PIL import Image

# Set page config - this must be the first Streamlit command
st.set_page_config(
    page_title="Face Identification App (Simple Version)",
    page_icon="👤",
    layout="wide"
)

# Try to import DeepFace
try:
    from deepface import DeepFace
    deepface_available = True
except ImportError:
    deepface_available = False

# Define constants
LIBRARY_PATH = "library"
if not os.path.exists(LIBRARY_PATH):
    os.makedirs(LIBRARY_PATH)

# Set app title
st.title("Face Identification App (Simple Version)")

# Display dependency status
if not deepface_available:
    st.error("⚠️ DeepFace is not installed in this environment")
    st.info("""
    To install DeepFace, run the following command in your terminal:
    ```
    pip install deepface tensorflow opencv-python-headless
    ```
    
    After installing, restart the Streamlit app.
    
    **If you've already installed it elsewhere:** You might be running Streamlit in a different Python environment than where DeepFace is installed.
    
    Run this to check your environment:
    ```
    python check_dependencies.py
    ```
    """)
    # Show environment details
    import sys
    st.write(f"Python executable: {sys.executable}")
else:
    st.success("✅ DeepFace is available")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Library", "Identification", "Results"])

# Library Section
with tab1:
    st.header("Photo Library")
    
    # Upload new images to library
    uploaded_files = st.file_uploader("Upload images to the library", 
                                     type=['jpg', 'jpeg', 'png'], 
                                     accept_multiple_files=True)
    
    if uploaded_files:
        with st.spinner("Saving images to library..."):
            for uploaded_file in uploaded_files:
                # Save the uploaded image to the library
                filepath = os.path.join(LIBRARY_PATH, uploaded_file.name)
                with open(filepath, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Saved {uploaded_file.name} to library!")
    
    # Display library images
    library_images = [os.path.join(LIBRARY_PATH, f) for f in os.listdir(LIBRARY_PATH) 
                     if os.path.isfile(os.path.join(LIBRARY_PATH, f))]
    
    if library_images:
        st.subheader(f"Library Images ({len(library_images)})")
        
        # Create a grid layout for displaying images
        cols_per_row = 4
        rows = (len(library_images) + cols_per_row - 1) // cols_per_row
        
        for row in range(rows):
            cols = st.columns(cols_per_row)
            
            for col_idx in range(cols_per_row):
                img_idx = row * cols_per_row + col_idx
                
                if img_idx < len(library_images):
                    with cols[col_idx]:
                        img_path = library_images[img_idx]
                        img = Image.open(img_path)
                        st.image(img, caption=os.path.basename(img_path), width=150)
                        
                        # Add delete button for each image
                        if st.button(f"Delete", key=f"delete_{img_idx}"):
                            os.remove(img_path)
                            st.success(f"Deleted {os.path.basename(img_path)}")
                            st.rerun()
    else:
        st.info("No images in the library. Please upload some images.")

# Main Section (Identification)
with tab2:
    st.header("Face Identification")
    
    if not deepface_available:
        st.warning("DeepFace is not available. Face identification will not work.")
    else:
        # Provide options for selecting the target image
        option = st.radio("Select the target image source:", 
                         ["Upload Image", "Use Camera"])
        
        target_image_path = None
        
        if option == "Upload Image":
            target_file = st.file_uploader("Upload an image to identify", 
                                          type=['jpg', 'jpeg', 'png'])
            
            if target_file:
                # Save the uploaded target image temporarily
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                temp_file.write(target_file.getbuffer())
                target_image_path = temp_file.name
                
                # Display the target image
                target_img = Image.open(target_image_path)
                st.image(target_img, caption="Target Image", width=300)
        
        elif option == "Use Camera":
            # Camera integration using Streamlit's camera_input
            camera_image = st.camera_input("Take a picture")
            
            if camera_image:
                # Save the captured image temporarily
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                temp_file.write(camera_image.getbuffer())
                target_image_path = temp_file.name
        
        # Process the target image for identification
        if target_image_path and st.button("Identify Face"):
            library_images = [os.path.join(LIBRARY_PATH, f) for f in os.listdir(LIBRARY_PATH) 
                             if os.path.isfile(os.path.join(LIBRARY_PATH, f))]
            
            if not library_images:
                st.warning("No images in the library. Please upload some images to the library first.")
                st.stop()
            
            with st.spinner("Finding matches..."):
                # Test all library images
                matches = []
                threshold = 0.8  # More permissive threshold for testing
                
                # Debug log
                st.write(f"Target image: {target_image_path}")
                st.write(f"Library images: {len(library_images)}")
                
                # Try to match with each library image
                for library_image in library_images:
                    try:
                        # Log which image we're processing
                        st.write(f"Processing: {os.path.basename(library_image)}")
                        
                        # Verify face
                        result = DeepFace.verify(
                            img1_path=target_image_path, 
                            img2_path=library_image,
                            enforce_detection=False
                        )
                        
                        # Log the results
                        st.write(f"  Distance: {result['distance']}")
                        st.write(f"  Verified: {result['verified']}")
                        
                        # Add to matches if distance is low (good match)
                        if result['verified'] or result['distance'] < threshold:
                            matches.append({
                                'image_path': library_image,
                                'distance': result['distance'],
                                'verified': result['verified']
                            })
                    except Exception as e:
                        st.write(f"Error processing {os.path.basename(library_image)}: {e}")
                
                # Display results
                if matches:
                    st.success(f"Found {len(matches)} matching faces!")
                    
                    # Sort matches by distance (best matches first)
                    matches.sort(key=lambda x: x['distance'])
                    
                    # Display each match
                    for i, match in enumerate(matches):
                        st.write(f"Match #{i+1}: {os.path.basename(match['image_path'])}")
                        st.write(f"Distance: {match['distance']}")
                        st.image(Image.open(match['image_path']), width=200)
                else:
                    st.warning("No matching faces found in the library.")
                    st.write("This could be due to:")
                    st.write("1. Face detection issues - ensure faces are clearly visible")
                    st.write("2. The threshold might be too strict")
                    st.write("3. The images might be too different")

# Results Section
with tab3:
    st.header("Identification Results")
    st.info("Run an identification in the Identification tab to see results here.")

# Footer
st.markdown("---")
st.markdown("Face Identification App using DeepFace and Streamlit (Simple Version)")