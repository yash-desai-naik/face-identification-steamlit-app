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
import cv2
from utils import (
    create_library_directory, 
    save_uploaded_image, 
    get_library_images, 
    delete_image, 
    find_matching_faces, 
    deepface_available,
    get_face_metadata,
    extract_and_save_faces_metadata
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
    st.session_state.threshold = 0.5  # Stricter default threshold
if 'model_name' not in st.session_state:
    st.session_state.model_name = 'Facenet512'  # Better default model
if 'max_workers' not in st.session_state:
    # Default to 4 workers for better performance on M-series chips
    st.session_state.max_workers = 4
if 'strict_mode' not in st.session_state:
    st.session_state.strict_mode = True  # Enable strict mode by default

# Set app title
st.title("Face Identification App")

# Sidebar for settings and information
with st.sidebar:
    st.title("Settings")
    
    # Model selection
    model_options = ["Facenet512", "ArcFace", "Facenet", "VGG-Face", "OpenFace", "DeepFace"]
    st.session_state.model_name = st.selectbox(
        "Face Recognition Model",
        model_options,
        index=model_options.index("Facenet512") if "Facenet512" in model_options else 0
    )
    
    # Threshold adjustment
    st.session_state.threshold = st.slider(
        "Matching Threshold", 
        0.0, 1.0, 0.5, 0.01,  # Stricter default threshold
        help="Lower threshold values result in more matches but may include false positives."
    )
    
    # Strict mode option
    st.session_state.strict_mode = st.checkbox(
        "Strict Matching", 
        value=True,
        help="Enable additional verification to reduce false positives"
    )
    
    # Performance settings
    st.subheader("Performance Settings")
    
    # Get system memory info
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    available_gb = mem.available / (1024**3)
    
    st.info(f"System Memory: {total_gb:.1f}GB (Available: {available_gb:.1f}GB)")
    
    # Worker count - Set default based on available memory
    default_workers = min(max(2, int(available_gb / 2)), 8)  # More aggressive parallelism for Apple Silicon
    st.session_state.max_workers = st.slider(
        "Parallel Workers", 
        2, 16, default_workers,
        help="More workers may speed up identification on powerful machines like Apple Silicon."
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app uses DeepFace for face recognition with improved accuracy.
    
    **Features:**
    - Upload images to create a photo library
    - Identify faces using uploaded images or camera
    - Detect and match faces in group photos
    - Advanced matching validation to reduce false positives
    - Optimized for Apple Silicon
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
                        
                        # Get face metadata
                        metadata = get_face_metadata(img_path, LIBRARY_PATH)
                        face_count = metadata.get('face_count', 0) if metadata else 0
                        
                        # Show badge for group photos
                        if face_count > 1:
                            st.image(img, caption=f"{os.path.basename(img_path)} ({face_count} faces)", width=150)
                        else:
                            st.image(img, caption=os.path.basename(img_path), width=150)
                        
                        # Add delete button for each image
                        if st.button(f"Delete", key=f"delete_{img_idx}"):
                            if delete_image(img_path):
                                st.success(f"Deleted {os.path.basename(img_path)}")
                                st.rerun()
                                
                        # Add analyze button for each image
                        if st.button(f"Extract Faces", key=f"analyze_{img_idx}"):
                            with st.spinner(f"Extracting faces from {os.path.basename(img_path)}..."):
                                metadata = extract_and_save_faces_metadata(img_path, LIBRARY_PATH)
                                if metadata:
                                    st.write(f"Found {metadata['face_count']} face(s)")
                                    # If it's a group photo, show extracted faces
                                    if metadata.get('face_files'):
                                        face_cols = st.columns(min(4, len(metadata['face_files'])))
                                        for i, face_file in enumerate(metadata['face_files']):
                                            with face_cols[i % 4]:
                                                face_img = Image.open(face_file['path'])
                                                st.image(face_img, caption=f"Face {i+1}", width=100)
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
            
            # Extract face from target image first
            with st.spinner("Analyzing target face..."):
                # Create a temporary directory for target face extraction
                temp_dir = tempfile.mkdtemp()
                try:
                    # Extract faces with multiple detection methods
                    target_metadata = extract_and_save_faces_metadata(st.session_state.target_image_path, temp_dir)
                    
                    # If no face found in target even with our robust methods, use the whole image
                    if not target_metadata or target_metadata['face_count'] == 0:
                        st.warning("Face detection is having trouble with this image. Proceeding with the whole image...")
                        # No need to stop, we'll use the whole image as is
                except Exception as e:
                    st.error(f"Error during face analysis: {e}")
                    # Continue with the original image even if face detection fails
            
            with st.spinner("Searching for matches..."):
                # Create a progress bar
                progress_bar = st.progress(0)
                
                # Debug information
                debug_container.text(f"Target image: {st.session_state.target_image_path}")
                debug_container.text(f"Model: {st.session_state.model_name}")
                debug_container.text(f"Threshold: {st.session_state.threshold}")
                debug_container.text(f"Library images: {len(library_images)}")
                debug_container.text(f"Parallel workers: {st.session_state.max_workers}")
                debug_container.text(f"Strict mode: {'Enabled' if st.session_state.strict_mode else 'Disabled'}")
                
                # Run matching with the specified number of parallel workers
                st.session_state.matches = find_matching_faces(
                    st.session_state.target_image_path, 
                    library_images,
                    threshold=st.session_state.threshold,
                    model_name=st.session_state.model_name,
                    progress_bar=progress_bar,
                    max_workers=st.session_state.max_workers,
                    library_path=LIBRARY_PATH
                )
                
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
                    debug_container.text("2. The threshold might be too strict (try lowering it)")
                    debug_container.text("3. Try a different model like ArcFace")
                    debug_container.text("4. Try disabling Strict Matching mode")
                    debug_container.text("\nTry uploading clearer images with faces clearly visible.")
        else:
            st.warning("No images in the library. Please upload some images to the library first.")

# Results Section
with tab3:
    st.header("Identification Results")
    
    # Check if we have any matches
    if 'matches' in st.session_state and st.session_state.matches:
        matches = st.session_state.matches
        
        # Categorize matches into direct and group photo matches
        direct_matches = [m for m in matches if not m.get('is_face_extract', False)]
        group_matches = [m for m in matches if m.get('is_face_extract', False)]
        
        st.success(f"Found {len(matches)} matching faces! ({len(direct_matches)} direct matches, {len(group_matches)} from group photos)")
        
        # Allow filtering matches further with a confidence slider
        min_confidence = st.slider(
            "Minimum match confidence (%)", 
            0, 100, 60,  # Default to higher confidence minimum
            help="Only show matches with confidence above this threshold"
        )
        
        # Filter matches based on confidence
        filtered_matches = [
            m for m in matches 
            if max(0, min(100, (1 - m['distance']) * 100)) >= min_confidence
        ]
        
        if not filtered_matches:
            st.info(f"No matches meet the minimum confidence threshold of {min_confidence}%. Try lowering the threshold.")
            st.stop()
            
        st.write(f"Showing {len(filtered_matches)} matches above {min_confidence}% confidence")
        
        # Sort the matches by match quality (lowest distance = best match)
        sorted_matches = sorted(filtered_matches, key=lambda x: x['distance'])
        
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
                            # Show original image with highlight if it's from a group photo
                            original_img_path = match['image_path']
                            display_img = Image.open(original_img_path)
                            
                            # If this is a face from a group photo, highlight the face
                            if match.get('is_face_extract', False) and match.get('facial_area'):
                                # Draw rectangle around matched face in original image
                                img_cv = cv2.imread(original_img_path)
                                facial_area = match['facial_area']
                                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                                
                                # Draw a more visible rectangle (thicker, bright green)
                                cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 5)
                                
                                # Convert back to PIL for display
                                display_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                                
                                # Show the caption with group photo designation
                                st.image(display_img, caption=f"Group photo: {os.path.basename(original_img_path)}", width=200)
                                
                                # Also display the extracted face
                                face_img = Image.open(match['face_path'])
                                st.image(face_img, caption="Matched Face", width=120)
                            else:
                                # Regular single face photo
                                st.image(display_img, caption=os.path.basename(original_img_path), width=200)
                                
                        except Exception as e:
                            st.error(f"Error loading image: {e}")
                            continue
                        
                        st.write(f"**Match #{match_idx+1}**")
                        # Convert distance to percentage (lower distance = better match)
                        match_percentage = max(0, min(100, (1 - match['distance']) * 100))
                        
                        # Use colors to indicate match quality
                        if match_percentage >= 90:
                            st.markdown(f"<h3 style='color:green'>Match: {match_percentage:.1f}%</h3>", unsafe_allow_html=True)
                        elif match_percentage >= 75:
                            st.markdown(f"<h3 style='color:orange'>Match: {match_percentage:.1f}%</h3>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<h3 style='color:red'>Match: {match_percentage:.1f}%</h3>", unsafe_allow_html=True)
    else:
        st.info("No matches found. Please run an identification in the Identification tab.")

# Footer
st.markdown("---")
st.markdown("Face Identification App using DeepFace and Streamlit")