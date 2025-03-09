import os
import numpy as np
from PIL import Image
import shutil
import importlib.util

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
            
        # Analyze the face for age, gender, emotion, and race
        analysis = DeepFace.analyze(
            img_path=image_path, 
            actions=['age', 'gender', 'emotion', 'race'],
            enforce_detection=False  # Important: Don't fail if face isn't clearly detected
        )
        return analysis
    except Exception as e:
        print(f"Error analyzing face: {e}")
        import traceback
        traceback.print_exc()
        return None

def find_matching_faces(target_image_path, library_images, threshold=0.6, model_name='VGG-Face', progress_bar=None):
    """Find matching faces in the library."""
    matches = []
    debug_info = []
    
    if not library_images:
        return matches
    
    # Check if DeepFace is available
    if not deepface_available:
        print("DeepFace is not available")
        return matches
    
    # Ensure target_image_path exists and is readable
    if not os.path.isfile(target_image_path):
        print(f"Target image not found or not readable: {target_image_path}")
        return matches
    
    try:
        # Try to open the target image to confirm it's valid
        try:
            img = Image.open(target_image_path)
            img.verify()  # Verify it's a valid image
            print(f"Target image verified: {target_image_path}")
        except Exception as e:
            print(f"Target image verification failed: {e}")
            return matches
        
        total_images = len(library_images)
        print(f"Processing {total_images} library images")
        
        for i, library_image in enumerate(library_images):
            if progress_bar:
                progress_bar.progress((i + 1) / total_images)
            
            # Ensure library_image exists and is readable
            if not os.path.isfile(library_image):
                print(f"Library image not found or not readable: {library_image}")
                continue
            
            try:
                # Try to open the library image to confirm it's valid
                try:
                    img = Image.open(library_image)
                    img.verify()  # Verify it's a valid image
                except Exception as e:
                    print(f"Library image verification failed for {library_image}: {e}")
                    continue
                
                print(f"Comparing target with: {os.path.basename(library_image)}")
                
                # We'll directly use the DeepFace.verify function
                try:
                    result = DeepFace.verify(
                        img1_path=target_image_path, 
                        img2_path=library_image, 
                        model_name=model_name,
                        enforce_detection=False  # Important: Don't fail if face isn't clearly detected
                    )
                    
                    # Log the result for debugging
                    print(f"Result for {os.path.basename(library_image)}: Distance={result['distance']}, Verified={result['verified']}")
                    debug_info.append({
                        'image': os.path.basename(library_image),
                        'distance': result['distance'],
                        'verified': result['verified']
                    })
                    
                    # Always add to matches for debugging, but mark whether it passed threshold
                    passed_threshold = result['verified'] or result['distance'] < threshold
                    
                    if passed_threshold:
                        # Get additional analysis for the face
                        try:
                            # Try to get analysis but don't fail if it doesn't work
                            try:
                                analysis = analyze_face(library_image)
                            except Exception as e:
                                print(f"Analysis failed for {os.path.basename(library_image)}: {e}")
                                analysis = None
                                
                            matches.append({
                                'image_path': library_image,
                                'distance': result['distance'],
                                'verified': result['verified'],
                                'passed_threshold': passed_threshold,
                                'analysis': analysis
                            })
                        except Exception as e:
                            print(f"Could not analyze face in {os.path.basename(library_image)}: {e}")
                            # Still add the match but without analysis
                            matches.append({
                                'image_path': library_image,
                                'distance': result['distance'],
                                'verified': result['verified'],
                                'passed_threshold': passed_threshold,
                                'analysis': None
                            })
                except Exception as e:
                    print(f"Verification failed for {os.path.basename(library_image)}: {e}")
            except Exception as e:
                print(f"Error processing library image {library_image}: {e}")
                continue
    except Exception as e:
        print(f"Error finding matching faces: {e}")
        import traceback
        traceback.print_exc()
    
    # Print debug summary
    print(f"Debug Summary: Found {len(matches)} total images")
    for match in matches:
        print(f"{os.path.basename(match['image_path'])}: Distance={match['distance']}, Verified={match['verified']}")
    
    return matches

def sort_matches(matches, sort_by='distance'):
    """Sort matches based on the specified criteria."""
    if not matches:
        return []
        
    if sort_by == 'distance':
        return sorted(matches, key=lambda x: x['distance'])
    elif sort_by == 'emotion':
        # Sort by the highest emotion value
        def get_top_emotion(match):
            if match['analysis'] is None:
                return 0
            emotions = match['analysis'][0]['emotion']
            return max(emotions.values()) if emotions else 0
        return sorted(matches, key=get_top_emotion, reverse=True)
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