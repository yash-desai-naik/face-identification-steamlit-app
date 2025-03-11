import os
import numpy as np
from PIL import Image
import shutil
import importlib.util
import os

# Force CPU only for TensorFlow operations
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import concurrent.futures
import time
import gc
import cv2
import json
import tempfile
from pathlib import Path

# Check if deepface is available without using Streamlit
deepface_available = importlib.util.find_spec("deepface") is not None

# Only import if available
if deepface_available:
    from deepface import DeepFace
    # Try to import face embeddings functionality
    try:
        from deepface.commons import functions as deepface_functions
        embeddings_available = True
    except:
        embeddings_available = False
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
    embeddings_available = False

def create_library_directory(library_path='library'):
    """Create the library directory if it doesn't exist."""
    if not os.path.exists(library_path):
        os.makedirs(library_path)
    
    # Also create directories to store face metadata and embeddings
    metadata_path = os.path.join(library_path, 'metadata')
    if not os.path.exists(metadata_path):
        os.makedirs(metadata_path)
        
    embeddings_path = os.path.join(library_path, 'embeddings')
    if not os.path.exists(embeddings_path):
        os.makedirs(embeddings_path)
        
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

def extract_faces_with_mtcnn(img_path):
    """Use MTCNN for more accurate face detection on Apple Silicon."""
    try:
        from mtcnn import MTCNN
        import cv2
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            return []
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Initialize MTCNN with lower confidence threshold
        detector = MTCNN(min_face_size=20)
        
        # Detect faces
        faces = detector.detect_faces(img_rgb)
        
        # Convert to DeepFace-like format
        result = []
        for face in faces:
            # Lower confidence threshold to 0.7 to catch more faces
            if face.get('confidence', 0) < 0.7:
                continue
                
            box = face['box']
            result.append({
                'facial_area': {
                    'x': box[0],
                    'y': box[1],
                    'w': box[2],
                    'h': box[3]
                },
                'confidence': face.get('confidence', 0.9)
            })
        
        # If MTCNN fails to detect faces, try cascade classifier as fallback
        if not result:
            # Try OpenCV's cascade classifier as fallback
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                result.append({
                    'facial_area': {
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h
                    },
                    'confidence': 0.8  # Default confidence for cascade detector
                })
        
        return result
    except Exception as e:
        print(f"MTCNN face detection failed: {e}")
        return []

def extract_and_save_faces_metadata(image_path, library_path='library'):
    """Extract faces from an image and save metadata."""
    if not os.path.exists(image_path):
        return []
    
    try:
        # Create metadata directory if it doesn't exist
        metadata_dir = os.path.join(library_path, 'metadata')
        if not os.path.exists(metadata_dir):
            os.makedirs(metadata_dir)
        
        # Create embeddings directory if it doesn't exist
        embeddings_dir = os.path.join(library_path, 'embeddings')
        if not os.path.exists(embeddings_dir):
            os.makedirs(embeddings_dir)
        
        # Create a unique metadata filename based on the image name
        image_filename = os.path.basename(image_path)
        metadata_filename = f"{os.path.splitext(image_filename)[0]}_faces.json"
        metadata_path = os.path.join(metadata_dir, metadata_filename)
        
        faces = []
        
        # Step 1: Try MTCNN for better accuracy (includes fallback to cascade)
        faces = extract_faces_with_mtcnn(image_path)
        
        # Step 2: If MTCNN fails, try DeepFace if available
        if not faces and deepface_available:
            try:
                detector_backends = ['opencv', 'ssd', 'mtcnn', 'retinaface']
                
                # Try different detector backends until one works
                for detector in detector_backends:
                    try:
                        extracted_faces = DeepFace.extract_faces(
                            img_path=image_path,
                            enforce_detection=False,
                            align=True,
                            detector_backend=detector
                        )
                        
                        if extracted_faces:
                            faces = extracted_faces
                            print(f"Found faces using {detector} backend")
                            break
                    except Exception as e:
                        print(f"Failed with {detector} backend: {e}")
                        continue
            except Exception as e:
                print(f"DeepFace extraction failed: {e}")
        
        # Step 3: Manual face detection as last resort - full image
        if not faces:
            print("All face detection methods failed, using whole image as face")
            # Get image dimensions and use the entire image as a face
            img = cv2.imread(image_path)
            if img is not None:
                h, w = img.shape[:2]
                # Create a "face" covering most of the image
                faces = [{
                    'facial_area': {
                        'x': int(w * 0.1),  # 10% margin
                        'y': int(h * 0.1),  # 10% margin
                        'w': int(w * 0.8),  # 80% of width
                        'h': int(h * 0.8)   # 80% of height
                    },
                    'confidence': 0.5  # Low confidence for this fallback method
                }]
        
        # Check if this is a group photo (has multiple faces)
        is_group_photo = len(faces) > 1
        
        # Save face images with larger margins
        face_files = []
        if faces:  # Process all images with detected faces
            # Load the original image
            img = cv2.imread(image_path)
            if img is None:
                return []
                
            # Save each face as a file
            for i, face_data in enumerate(faces):
                face_region = face_data['facial_area']
                x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
                
                # Add a larger margin around the face (50% for better recognition)
                margin_x = int(w * 0.5)
                margin_y = int(h * 0.5)
                
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
                    
                # Save the face to a file in the metadata directory
                face_filename = f"{os.path.splitext(image_filename)[0]}_face_{i}.jpg"
                face_path = os.path.join(metadata_dir, face_filename)
                cv2.imwrite(face_path, face_img)
                
                # Generate and save face embedding if available
                embedding_path = None
                if embeddings_available and deepface_available:
                    try:
                        # Compute embedding for the face image
                        models = ["VGG-Face", "Facenet", "Facenet512", "ArcFace"]
                        embeddings = {}
                        
                        for model_name in models:
                            try:
                                embedding = DeepFace.represent(
                                    img_path=face_path,
                                    model_name=model_name,
                                    enforce_detection=False
                                )
                                embeddings[model_name] = embedding
                            except:
                                pass
                        
                        if embeddings:
                            # Save embeddings to file
                            embedding_filename = f"{os.path.splitext(image_filename)[0]}_face_{i}_embedding.json"
                            embedding_path = os.path.join(embeddings_dir, embedding_filename)
                            with open(embedding_path, 'w') as f:
                                json.dump(embeddings, f)
                    except Exception as e:
                        print(f"Failed to generate face embedding: {e}")
                
                # Add file info to the list
                face_files.append({
                    'filename': face_filename,
                    'path': face_path,
                    'embedding_path': embedding_path,
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
                    'confidence': face.get('confidence', 0.5)
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
        embeddings_dir = os.path.join(library_path, 'embeddings')
        
        # Delete metadata file
        metadata_filename = f"{base_name}_faces.json"
        metadata_path = os.path.join(metadata_dir, metadata_filename)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            
        # Delete face files
        for f in os.listdir(metadata_dir):
            if f.startswith(f"{base_name}_face_") and os.path.isfile(os.path.join(metadata_dir, f)):
                os.remove(os.path.join(metadata_dir, f))
                
        # Delete embedding files
        if os.path.exists(embeddings_dir):
            for f in os.listdir(embeddings_dir):
                if f.startswith(f"{base_name}_face_") and os.path.isfile(os.path.join(embeddings_dir, f)):
                    os.remove(os.path.join(embeddings_dir, f))
                
        return True
    return False

def evaluate_face_match(result, target_path, face_path):
    """Additional verification for face matches to reduce false positives."""
    try:
        # Primary verification from DeepFace.verify result
        distance = result['distance']
        verified = result['verified']
        
        # Attempt secondary verification using alternative models
        if deepface_available:
            # Try a different model for verification
            alternative_models = ["Facenet512", "ArcFace", "Facenet"]
            current_model = result.get('model', "VGG-Face")
            
            # Find a model we didn't use yet
            verification_model = next((m for m in alternative_models if m != current_model), None)
            
            if verification_model:
                try:
                    second_verification = DeepFace.verify(
                        img1_path=target_path,
                        img2_path=face_path,
                        model_name=verification_model,
                        enforce_detection=False
                    )
                    
                    # Require at least one alternative model to agree
                    if not second_verification['verified'] and verified:
                        print(f"Secondary verification failed using {verification_model}")
                        # Increase distance to reduce match confidence
                        distance = (distance + 1.0) / 2  # Average with 1.0 (maximum distance)
                        verified = False
                except:
                    # If secondary verification fails, be more conservative
                    if distance > 0.3:  # If it's not a very strong match already
                        distance += 0.1  # Penalize the confidence
                
            # Check face size ratio for additional verification
            try:
                img1 = cv2.imread(target_path)
                img2 = cv2.imread(face_path)
                
                if img1 is not None and img2 is not None:
                    # Get face sizes
                    h1, w1 = img1.shape[:2]
                    h2, w2 = img2.shape[:2]
                    
                    # Compare face aspect ratios - they should be similar for the same person
                    ratio1 = w1 / h1
                    ratio2 = w2 / h2
                    
                    ratio_diff = abs(ratio1 - ratio2)
                    if ratio_diff > 0.5:  # If aspect ratios are very different
                        print(f"Face aspect ratio differs significantly: {ratio_diff}")
                        distance += 0.15  # Penalize the match
                        if distance >= 0.8:
                            verified = False
            except Exception as e:
                print(f"Face size verification error: {e}")
        
        # Update the result with our new verification
        return {
            'distance': distance,
            'verified': verified,
            'adjusted': True  # Flag that we've applied additional verification
        }
        
    except Exception as e:
        print(f"Error in evaluate_face_match: {e}")
        # Return original result if evaluation fails
        return result

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
            
        # Compare faces using DeepFace with optimized settings for performance
        result = DeepFace.verify(
            img1_path=target_image_path, 
            img2_path=face_path, 
            model_name=model_name,
            detector_backend='opencv',  # Fastest detector
            enforce_detection=False,    # Don't fail on detection errors
            align=True                 # Align faces for better accuracy
        )
        
        # Apply additional verification to reduce false positives
        result = evaluate_face_match(result, target_image_path, face_path)
        
        # Check if match passes threshold
        passed_threshold = result['verified'] or result['distance'] < threshold
        
        if passed_threshold:
            return {
                'image_path': library_image,  # Original image path
                'face_path': face_path,       # Path to the face image
                'distance': result['distance'],
                'verified': result['verified'],
                'passed_threshold': passed_threshold,
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
            # Check if this is a group photo and extract individual faces
            metadata = get_face_metadata(img_path, library_path)
            
            if metadata and metadata.get('face_files'):
                # Add individual faces from the image
                for face_file in metadata['face_files']:
                    comparison_items.append({
                        'original_image': img_path,
                        'face_path': face_file['path'],
                        'facial_area': face_file['facial_area']
                    })
            else:
                # Add the full image if no faces were extracted
                comparison_items.append(img_path)
        
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
                            
                        # Basic info for debugging
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