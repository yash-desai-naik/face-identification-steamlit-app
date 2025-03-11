import os
import numpy as np
from PIL import Image
import shutil
import importlib.util
import concurrent.futures
import time
import gc
import cv2
import json
import tempfile

# Check if deepface is available without using Streamlit
deepface_available = importlib.util.find_spec("deepface") is not None

# Only import if available
if deepface_available:
    from deepface import DeepFace
else:
    # Create a dummy DeepFace class for minimal functionality
    class DummyDeepFace:
        @staticmethod
        def analyze(*args, **kwargs):
            return None
        
        @staticmethod
        def verify(*args, **kwargs):
            return {"verified": False, "distance": 1.0}
        
        @staticmethod
        def extract_faces(*args, **kwargs):
            return []
    
    DeepFace = DummyDeepFace

def create_library_directory(library_path='library'):
    """Create the library directory if it doesn't exist."""
    if not os.path.exists(library_path):
        os.makedirs(library_path)
    
    # Also create a directory to store face metadata
    metadata_path = os.path.join(library_path, 'metadata')
    if not os.path.exists(metadata_path):
        os.makedirs(metadata_path)
        
    return library_path

def save_uploaded_image(uploaded_image, library_path='library', filename=None):
    """Save an uploaded image to the library directory."""
    if filename is None:
        filename = uploaded_image.name
    
    filepath = os.path.join(library_path, filename)
    with open(filepath, 'wb') as f:
        f.write(uploaded_image.getbuffer())
    
    # Extract and save face information for the image
    extract_and_save_faces_metadata(filepath, library_path)
    
    return filepath

def extract_and_save_faces_metadata(image_path, library_path='library'):
    """Extract faces from an image and save metadata."""
    if not deepface_available or not os.path.exists(image_path):
        return []
    
    try:
        # Create a unique metadata filename based on the image name
        image_filename = os.path.basename(image_path)
        metadata_filename = f"{os.path.splitext(image_filename)[0]}_faces.json"
        metadata_path = os.path.join(library_path, 'metadata', metadata_filename)
        
        # Extract faces from the image
        faces = DeepFace.extract_faces(
            img_path=image_path,
            enforce_detection=False,  # Don't fail if face detection isn't clear
            align=True,
            detector_backend='opencv'
        )
        
        # Check if this is a group photo (has multiple faces)
        is_group_photo = len(faces) > 1
        
        # Save temporary face images for group photos
        face_files = []
        if is_group_photo:
            # Load the original image
            img = cv2.imread(image_path)
            if img is None:
                return []
                
            # Save each face as a temporary file
            for i, face_data in enumerate(faces):
                face_region = face_data['facial_area']
                x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
                
                # Add a margin around the face (20%)
                margin_x = int(w * 0.2)
                margin_y = int(h * 0.2)
                
                # Ensure coordinates are within image bounds
                x_start = max(0, x - margin_x)
                y_start = max(0, y - margin_y)
                x_end = min(img.shape[1], x + w + margin_x)
                y_end = min(img.shape[0], y + h + margin_y)
                
                # Extract the face region with margin
                face_img = img[y_start:y_end, x_start:x_end]
                
                # Skip if face extraction failed
                if face_img.size == 0:
                    continue
                    
                # Save the face to a temporary file in the metadata directory
                face_filename = f"{os.path.splitext(image_filename)[0]}_face_{i}.jpg"
                face_path = os.path.join(library_path, 'metadata', face_filename)
                cv2.imwrite(face_path, face_img)
                
                # Add file info to the list
                face_files.append({
                    'filename': face_filename,
                    'path': face_path,
                    'facial_area': face_region
                })
        
        # Save the metadata
        metadata = {
            'original_image': image_path,
            'is_group_photo': is_group_photo,
            'face_count': len(faces),
            'face_files': face_files,
            'faces': [
                {
                    'facial_area': face['facial_area'],
                    'confidence': face['confidence'] if 'confidence' in face else None
                } for face in faces
            ]
        }
        
        # Save metadata to JSON file
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return metadata
        
    except Exception as e:
        print(f"Error extracting faces from {image_path}: {e}")
        return []

def get_library_images(library_path='library'):
    """Get a list of all images in the library directory."""
    if not os.path.exists(library_path):
        return []
    
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    return [os.path.join(library_path, f) for f in os.listdir(library_path) 
            if os.path.isfile(os.path.join(library_path, f)) and 
            any(f.lower().endswith(ext) for ext in valid_extensions)]

def get_face_metadata(image_path, library_path='library'):
    """Get the face metadata for an image."""
    if not os.path.exists(image_path):
        return None
    
    image_filename = os.path.basename(image_path)
    metadata_filename = f"{os.path.splitext(image_filename)[0]}_faces.json"
    metadata_path = os.path.join(library_path, 'metadata', metadata_filename)
    
    if not os.path.exists(metadata_path):
        # If metadata doesn't exist, create it
        metadata = extract_and_save_faces_metadata(image_path, library_path)
        return metadata
    
    # Load metadata from file
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        print(f"Error loading face metadata for {image_path}: {e}")
        return None

def delete_image(filepath):
    """Delete an image from the library."""
    if os.path.exists(filepath):
        # Delete the image
        os.remove(filepath)
        
        # Also delete its metadata and face files
        image_filename = os.path.basename(filepath)
        base_name = os.path.splitext(image_filename)[0]
        library_path = os.path.dirname(filepath)
        metadata_dir = os.path.join(library_path, 'metadata')
        
        # Delete metadata file
        metadata_filename = f"{base_name}_faces.json"
        metadata_path = os.path.join(metadata_dir, metadata_filename)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            
        # Delete face files
        for f in os.listdir(metadata_dir):
            if f.startswith(f"{base_name}_face_") and os.path.isfile(os.path.join(metadata_dir, f)):
                os.remove(os.path.join(metadata_dir, f))
                
        return True
    return False

def analyze_face(image_path):
    """Analyze a face using DeepFace."""
    try:
        # Check if DeepFace is available
        if not deepface_available:
            return None
        
        # Make sure the image path exists
        if not os.path.isfile(image_path):
            print(f"Image not found for analysis: {image_path}")
            return None
            
        # Analyze the face for age and gender only
        analysis = DeepFace.analyze(
            img_path=image_path, 
            actions=['age', 'gender'],
            enforce_detection=False  # Important: Don't fail if face isn't clearly detected
        )
        return analysis
    except Exception as e:
        print(f"Error analyzing face: {e}")
        return None

def process_single_comparison(args):
    """Process a single face comparison - function for parallel processing."""
    target_image_path, comparison_data, threshold, model_name = args
    
    # Unpack comparison data
    if isinstance(comparison_data, dict):
        # This is a face from a group photo
        library_image = comparison_data['original_image']
        face_path = comparison_data['face_path']
        facial_area = comparison_data['facial_area']
        is_face_extract = True
    else:
        # This is a regular image path
        library_image = comparison_data
        face_path = comparison_data
        facial_area = None
        is_face_extract = False
    
    try:
        # Check if both images exist and are readable
        if not os.path.isfile(target_image_path) or not os.path.isfile(face_path):
            return None
        
        # Verify files are valid images
        try:
            img1 = Image.open(target_image_path)
            img1.verify()
            img1.close()
            
            img2 = Image.open(face_path)
            img2.verify()
            img2.close()
        except Exception as e:
            print(f"Image verification failed: {e}")
            return None
            
        # Compare faces using DeepFace
        result = DeepFace.verify(
            img1_path=target_image_path, 
            img2_path=face_path, 
            model_name=model_name,
            enforce_detection=False
        )
        
        # Check if match passes threshold
        passed_threshold = result['verified'] or result['distance'] < threshold
        
        if passed_threshold:
            # Try to get analysis but don't fail if it doesn't work
            try:
                analysis = analyze_face(face_path)
            except Exception:
                analysis = None
                
            return {
                'image_path': library_image,  # Original image path
                'face_path': face_path,       # Path to the face image
                'distance': result['distance'],
                'verified': result['verified'],
                'passed_threshold': passed_threshold,
                'analysis': analysis,
                'is_face_extract': is_face_extract,
                'facial_area': facial_area
            }
        else:
            # For non-matches, return minimal info to save memory
            return {
                'image_path': library_image,
                'face_path': face_path,
                'distance': result['distance'],
                'verified': result['verified'],
                'passed_threshold': False,
                'analysis': None,
                'is_face_extract': is_face_extract,
                'facial_area': facial_area
            }
            
    except Exception as e:
        print(f"Error processing comparison for {os.path.basename(face_path)}: {e}")
        return None

def find_matching_faces(target_image_path, library_images, threshold=0.6, model_name='VGG-Face', progress_bar=None, max_workers=2, library_path='library'):
    """Find matching faces in the library using parallel processing."""
    matches = []
    
    if not library_images or not deepface_available:
        return matches
    
    if not os.path.isfile(target_image_path):
        print(f"Target image not found: {target_image_path}")
        return matches
    
    try:
        # Prepare comparison items - includes both full images and extracted faces
        comparison_items = []
        
        # Process each library image
        for img_path in library_images:
            # Add the full image to comparison items
            comparison_items.append(img_path)
            
            # Check if this is a group photo and extract individual faces
            metadata = get_face_metadata(img_path, library_path)
            
            if metadata and metadata.get('is_group_photo', False) and metadata.get('face_files'):
                for face_file in metadata['face_files']:
                    comparison_items.append({
                        'original_image': img_path,
                        'face_path': face_file['path'],
                        'facial_area': face_file['facial_area']
                    })
        
        # Prepare arguments for parallel processing
        args_list = [(target_image_path, item, threshold, model_name) for item in comparison_items]
        total_items = len(comparison_items)
        
        # Limit number of workers based on system
        if max_workers > total_items:
            max_workers = total_items
            
        completed = 0
        all_results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_item = {executor.submit(process_single_comparison, args): args[1] 
                             for args in args_list}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_item):
                completed += 1
                if progress_bar:
                    progress_bar.progress(completed / total_items)
                
                item = future_to_item[future]
                try:
                    result = future.result()
                    if result is not None:
                        # Only keep full details for matches
                        if result['passed_threshold']:
                            matches.append(result)
                        # Basic info for debugging/tracking
                        if isinstance(item, dict):
                            item_name = os.path.basename(item['face_path'])
                        else:
                            item_name = os.path.basename(item)
                            
                        all_results.append({
                            'image': item_name,
                            'distance': result['distance'],
                            'verified': result['verified']
                        })
                except Exception as e:
                    if isinstance(item, dict):
                        item_name = os.path.basename(item['face_path'])
                    else:
                        item_name = os.path.basename(item)
                    print(f"Error getting result for {item_name}: {e}")
        
        # Manual garbage collection to help with memory
        gc.collect()
                
        # Print debug summary
        print(f"Debug Summary: Processed {len(all_results)} items, found {len(matches)} matches")
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        
    # Sort matches by distance (best matches first)
    matches.sort(key=lambda x: x['distance'])
    return matches

def sort_matches(matches, sort_by='distance'):
    """Sort matches based on the specified criteria."""
    if not matches:
        return []
        
    if sort_by == 'distance':
        return sorted(matches, key=lambda x: x['distance'])
    elif sort_by == 'gender':
        # Sort by gender
        def get_gender(match):
            if match['analysis'] is None:
                return ''
            return match['analysis'][0]['gender']
        return sorted(matches, key=get_gender)
    elif sort_by == 'age':
        # Sort by age
        def get_age(match):
            if match['analysis'] is None:
                return 0
            return match['analysis'][0]['age']
        return sorted(matches, key=get_age)
    else:
        return matches