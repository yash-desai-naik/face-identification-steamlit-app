import streamlit as st

# Set page config - this must be the first Streamlit command
st.set_page_config(
    page_title="Face Identification App",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import other modules and dependencies after setting page config
import os
import numpy as np
from PIL import Image
import tempfile
import atexit
import psutil
from utils import (
    create_library_directory, 
    save_uploaded_image, 
    get_library_images, 
    delete_image, 
    analyze_face, 
    find_matching_faces, 
    sort_matches,
    deepface_available
)

# Check if DeepFace is installed
if not deepface_available:
    st.error("DeepFace library is not installed. Please run: pip install deepface")
    st.stop()

# Global variables
LIBRARY_PATH = create_library_directory()
temp_files = []

def cleanup_temp_files():
    """Clean up temporary files when the app exits."""
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            print(f"Error removing temporary file {temp_file}: {e}")

# Register the cleanup function to run at exit
atexit.register(cleanup_temp_files)

# Initialize session state variables
if 'target_image_path' not in st.session_state:
    st.session_state.target_image_path = None
if 'matches' not in st.session_state:
    st.session_state.matches = []
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.6
if 'model_name' not in st.session_state:
    st.session_state.model_name = 'VGG-Face'
if 'max_workers' not in st.session_state:
    # Default to 2 workers for limited resource systems
    st.session_state.max_workers = 2

# Set app title
st.title("Face Identification App")

# Sidebar for settings and information
with st.sidebar:
    st.title("Settings")
    
    # Model selection
    st.session_state.model_name = st.selectbox(
        "Face Recognition Model",
        ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"],
        index=0
    )
    
    # Threshold adjustment
    st.session_state.threshold = st.slider(
        "Matching Threshold", 
        0.0, 1.0, 0.6, 0.01,
        help="Lower threshold values result in more matches but may include false positives."
    )
    
    # Performance settings
    st.subheader("Performance Settings")
    
    # Get system memory info
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    available_gb = mem.available / (1024**3)
    
    st.info(f"System Memory: {total_gb:.1f}GB (Available: {available_gb:.1f}GB)")
    
    # Worker count - Set default based on available memory
    default_workers = min(max(1, int(available_gb)), 4)
    st.session_state.max_workers = st.slider(
        "Parallel Workers", 
        1, 8, default_workers,
        help="More workers may speed up identification but use more memory. For systems with 2GB RAM or less, keep this at 1-2."
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app uses DeepFace for face recognition and analysis.
    
    **Features:**
    - Upload images to create a photo library
    - Identify faces using uploaded images or camera
    - Analyze basic face attributes (age and gender)
    - Sort and filter matches
    - Parallel processing for improved performance
    """)

# Create tabs for the three main sections
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
                filepath = save_uploaded_image(uploaded_file, LIBRARY_PATH)
                st.success(f"Saved {uploaded_file.name} to library!")
    
    # Display and manage library images
    library_images = get_library_images(LIBRARY_PATH)
    
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
                            if delete_image(img_path):
                                st.success(f"Deleted {os.path.basename(img_path)}")
                                st.rerun()
    else:
        st.info("No images in the library. Please upload some images.")

# Main Section (Identification)
with tab2:
    st.header("Face Identification")
    
    # Provide options for selecting the target image
    option = st.radio("Select the target image source:", 
                     ["Upload Image", "Use Camera"])
    
    if option == "Upload Image":
        target_file = st.file_uploader("Upload an image to identify", 
                                      type=['jpg', 'jpeg', 'png'])
        
        if target_file:
            # Save the uploaded target image temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_file.write(target_file.getbuffer())
            st.session_state.target_image_path = temp_file.name
            temp_files.append(temp_file.name)
            
            # Display the target image
            target_img = Image.open(st.session_state.target_image_path)
            st.image(target_img, caption="Target Image", width=300)
    
    elif option == "Use Camera":
        # Camera integration using Streamlit's camera_input
        camera_image = st.camera_input("Take a picture")
        
        if camera_image:
            # Save the captured image temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_file.write(camera_image.getbuffer())
            st.session_state.target_image_path = temp_file.name
            temp_files.append(temp_file.name)
            
            # Display the captured image
            st.image(camera_image, caption="Captured Image", width=300)
    
    # Process the target image for identification
    if st.session_state.target_image_path and st.button("Identify Face"):
        library_images = get_library_images(LIBRARY_PATH)
        
        if library_images:
            debug_container = st.empty()
            debug_container.text("Setting up face recognition...")
            
            # Log memory usage before processing
            mem_before = psutil.virtual_memory()
            debug_container.text(f"Memory usage before: {mem_before.percent}% ({mem_before.available/(1024**3):.1f}GB available)")
            
            with st.spinner("Analyzing face..."):
                # Create a progress bar
                progress_bar = st.progress(0)
                
                # Debug information
                debug_container.text(f"Target image: {st.session_state.target_image_path}")
                debug_container.text(f"Model: {st.session_state.model_name}")
                debug_container.text(f"Threshold: {st.session_state.threshold}")
                debug_container.text(f"Library images: {len(library_images)}")
                debug_container.text(f"Parallel workers: {st.session_state.max_workers}")
                
                # Run matching with the specified number of parallel workers
                st.session_state.matches = find_matching_faces(
                    st.session_state.target_image_path, 
                    library_images,
                    threshold=st.session_state.threshold,
                    model_name=st.session_state.model_name,
                    progress_bar=progress_bar,
                    max_workers=st.session_state.max_workers
                )
                
                # Log memory usage after processing
                mem_after = psutil.virtual_memory()
                debug_container.text(f"Memory usage after: {mem_after.percent}% ({mem_after.available/(1024**3):.1f}GB available)")
                
                # Log output to the debug container
                debug_container.text(f"Matching complete. Found {len(st.session_state.matches)} matches")
                
                # Clear the progress bar
                progress_bar.empty()
                
                if st.session_state.matches:
                    st.success(f"Found {len(st.session_state.matches)} matching faces!")
                    st.rerun()
                else:
                    st.warning("No matching faces found in the library.")
                    
                    # Show debugging info
                    debug_container.markdown("### Debugging Information")
                    debug_container.text("No matches found. This could be due to:")
                    debug_container.text("1. Face detection issues in the images")
                    debug_container.text("2. The threshold might be too strict")
                    debug_container.text("3. Model compatibility issues")
                    debug_container.text("\nTry uploading clearer images with faces clearly visible.")
        else:
            st.warning("No images in the library. Please upload some images to the library first.")

# Results Section
with tab3:
    st.header("Identification Results")
    
    # Check if we have any matches
    if 'matches' in st.session_state and st.session_state.matches:
        matches = st.session_state.matches
        
        st.success(f"Found {len(matches)} matching faces!")
        
        # Simple sorting options
        sort_by = st.selectbox(
            "Sort results by:", 
            ["Best Match (Highest %)", "Worst Match (Lowest %)", "Age", "Gender"]
        )
        
        # Sort the matches based on selected criteria
        if sort_by == "Best Match (Highest %)":
            sorted_matches = sorted(matches, key=lambda x: x['distance'])  # Lower distance = better match
        elif sort_by == "Worst Match (Lowest %)":
            sorted_matches = sorted(matches, key=lambda x: x['distance'], reverse=True)
        elif sort_by == "Age":
            # Sort by age (youngest to oldest)
            sorted_matches = sorted(matches, key=lambda x: x['analysis'][0]['age'] if x.get('analysis') and len(x.get('analysis', [])) > 0 and 'age' in x['analysis'][0] else 999)
        elif sort_by == "Gender":
            # Sort by gender (Female then Male)
            def get_gender(match):
                if not match.get('analysis') or len(match.get('analysis', [])) == 0 or 'gender' not in match['analysis'][0]:
                    return ""
                gender_data = match['analysis'][0]['gender']
                if isinstance(gender_data, dict):
                    return "Female" if gender_data.get('Woman', 0) > gender_data.get('Man', 0) else "Male"
                elif isinstance(gender_data, str):
                    if gender_data.lower() in ["woman", "female"]:
                        return "Female"
                    else:
                        return "Male"
                return str(gender_data)
            sorted_matches = sorted(matches, key=get_gender)
        
        # Display matches in a grid layout
        cols_per_row = 3
        rows = (len(sorted_matches) + cols_per_row - 1) // cols_per_row
        
        for row in range(rows):
            cols = st.columns(cols_per_row)
            
            for col_idx in range(cols_per_row):
                match_idx = row * cols_per_row + col_idx
                
                if match_idx < len(sorted_matches):
                    match = sorted_matches[match_idx]
                    
                    with cols[col_idx]:
                        try:
                            img = Image.open(match['image_path'])
                            st.image(img, caption=os.path.basename(match['image_path']), width=200)
                        except Exception as e:
                            st.error(f"Error loading image: {e}")
                            continue
                        
                        st.write(f"**Match #{match_idx+1}**")
                        # Convert distance to percentage (lower distance = better match)
                        match_percentage = max(0, min(100, (1 - match['distance']) * 100))
                        st.write(f"Match: {match_percentage:.1f}%")
                        
                        # Only show age and gender if available
                        if match.get('analysis') and len(match['analysis']) > 0:
                            analysis = match['analysis'][0]
                            
                            if 'age' in analysis:
                                st.write(f"Age: {analysis['age']}")
                            
                            if 'gender' in analysis:
                                gender_data = analysis['gender']
                                
                                # Handle when gender is returned as a dictionary with probabilities
                                if isinstance(gender_data, dict):
                                    # Find gender with highest probability
                                    if 'Man' in gender_data and 'Woman' in gender_data:
                                        is_male = gender_data['Man'] > gender_data['Woman']
                                        gender = "Male" if is_male else "Female"
                                        confidence = gender_data['Man'] if is_male else gender_data['Woman']
                                        st.write(f"Gender: {gender} ({confidence:.1f}%)")
                                    else:
                                        # If the format is different, just show the raw data
                                        st.write(f"Gender: {gender_data}")
                                # Handle when gender is returned as a string
                                elif isinstance(gender_data, str):
                                    # Convert string gender to "Male" or "Female"
                                    if gender_data.lower() in ["man", "male"]:
                                        gender = "Male"
                                    elif gender_data.lower() in ["woman", "female"]:
                                        gender = "Female"
                                    else:
                                        gender = gender_data
                                    st.write(f"Gender: {gender}")
                                else:
                                    # For any other format, just display as is
                                    st.write(f"Gender: {gender_data}")
    else:
        st.info("No matches found. Please run an identification in the Identification tab.")

# Footer
st.markdown("---")
st.markdown("Face Identification App using DeepFace and Streamlit")