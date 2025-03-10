import os
import numpy as np
from PIL import Image
import shutil
import importlib.util
import concurrent.futures
import time
import gc

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
    
    DeepFace = DummyDeepFace

def create_library_directory(library_path='library'):
    """Create the library directory if it doesn't exist."""
    if not os.path.exists(library_path):
        os.makedirs(library_path)
    return library_path

def save_uploaded_image(uploaded_image, library_path='library', filename=None):
    """Save an uploaded image to the library directory."""
    if filename is None:
        filename = uploaded_image.name
    
    filepath = os.path.join(library_path, filename)
    with open(filepath, 'wb') as f:
        f.write(uploaded_image.getbuffer())
    
    return filepath

def get_library_images(library_path='library'):
    """Get a list of all images in the library directory."""
    if not os.path.exists(library_path):
        return []
    
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    return [os.path.join(library_path, f) for f in os.listdir(library_path) 
            if os.path.isfile(os.path.join(library_path, f)) and 
            any(f.lower().endswith(ext) for ext in valid_extensions)]

def delete_image(filepath):
    """Delete an image from the library."""
    if os.path.exists(filepath):
        os.remove(filepath)
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
    target_image_path, library_image, threshold, model_name = args
    
    try:
        # Check if both images exist and are readable
        if not os.path.isfile(target_image_path) or not os.path.isfile(library_image):
            return None
        
        # Verify files are valid images
        try:
            img1 = Image.open(target_image_path)
            img1.verify()
            img1.close()
            
            img2 = Image.open(library_image)
            img2.verify()
            img2.close()
        except Exception as e:
            print(f"Image verification failed: {e}")
            return None
            
        # Compare faces using DeepFace
        result = DeepFace.verify(
            img1_path=target_image_path, 
            img2_path=library_image, 
            model_name=model_name,
            enforce_detection=False
        )
        
        # Check if match passes threshold
        passed_threshold = result['verified'] or result['distance'] < threshold
        
        if passed_threshold:
            # Try to get analysis but don't fail if it doesn't work
            try:
                analysis = analyze_face(library_image)
            except Exception:
                analysis = None
                
            return {
                'image_path': library_image,
                'distance': result['distance'],
                'verified': result['verified'],
                'passed_threshold': passed_threshold,
                'analysis': analysis
            }
        else:
            # For non-matches, return minimal info to save memory
            return {
                'image_path': library_image,
                'distance': result['distance'],
                'verified': result['verified'],
                'passed_threshold': False,
                'analysis': None
            }
            
    except Exception as e:
        print(f"Error processing comparison for {os.path.basename(library_image)}: {e}")
        return None

def find_matching_faces(target_image_path, library_images, threshold=0.6, model_name='VGG-Face', progress_bar=None, max_workers=2):
    """Find matching faces in the library using parallel processing."""
    matches = []
    
    if not library_images or not deepface_available:
        return matches
    
    if not os.path.isfile(target_image_path):
        print(f"Target image not found: {target_image_path}")
        return matches
    
    try:
        # Prepare arguments for parallel processing
        args_list = [(target_image_path, img, threshold, model_name) for img in library_images]
        total_images = len(library_images)
        
        # Limit number of workers based on system
        if max_workers > total_images:
            max_workers = total_images
            
        completed = 0
        all_results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_image = {executor.submit(process_single_comparison, args): args[1] 
                              for args in args_list}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_image):
                completed += 1
                if progress_bar:
                    progress_bar.progress(completed / total_images)
                
                image_path = future_to_image[future]
                try:
                    result = future.result()
                    if result is not None:
                        # Only keep full details for matches
                        if result['passed_threshold']:
                            matches.append(result)
                        # Basic info for debugging/tracking
                        all_results.append({
                            'image': os.path.basename(image_path),
                            'distance': result['distance'],
                            'verified': result['verified']
                        })
                except Exception as e:
                    print(f"Error getting result for {os.path.basename(image_path)}: {e}")
        
        # Manual garbage collection to help with memory
        gc.collect()
                
        # Print debug summary
        print(f"Debug Summary: Processed {len(all_results)} images, found {len(matches)} matches")
        
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