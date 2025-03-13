# Face Identification App

A powerful face recognition application built with Streamlit and DeepFace, optimized for performance on Apple Silicon Macs and other platforms.

![Face Identification App](https://i.imgur.com/placeholder/400/300)

## Features

- **Photo Library Management**: Upload, view, and manage your photo collection
- **Face Identification**: Match faces between uploaded photos and the library
- **Group Photo Support**: Automatically detects and extracts individual faces from group photos
- **Multiple Recognition Models**: Choose from Facenet512, ArcFace, VGG-Face, and more
- **Adjustable Threshold**: Fine-tune the matching confidence level
- **Performance Optimization**: Parallel processing with automatic hardware detection
- **Strict Matching Mode**: Advanced verification to reduce false positives
- **Camera Integration**: Use your webcam for real-time face identification

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/face-identification-app.git
cd face-identification-app
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# For macOS/Linux
python -m venv .venv
source .venv/bin/activate

# For Windows
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Note for Apple Silicon (M1/M2/M3) Users

This app is optimized for Apple Silicon. The default configuration should work well, but if you encounter any issues:

1. Ensure you're using a Python version built for ARM architecture
2. The app automatically uses CPU-only mode for better compatibility

#### Note for Windows & Linux Users

If you encounter issues with the `dlib-bin` package, you might need to install it separately:

```bash
# For Windows users who encounter dlib installation issues
pip install dlib-bin
# OR if that fails, try: 
pip install https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.99-cp310-cp310-win_amd64.whl

# For Linux users
# You might need to install additional dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake libopenblas-dev liblapack-dev
pip install dlib
```

## Usage

### Step 1: Start the App

```bash
streamlit run app.py
```

The app will launch in your default web browser, typically at `http://localhost:8501`.

### Step 2: Add Photos to Your Library

1. Navigate to the **Library** tab
2. Click the **Browse files** button to upload images
3. You can upload multiple images at once
4. For group photos, use the **Extract Faces** button to identify individual faces

### Step 3: Identify Faces

1. Switch to the **Identification** tab
2. Choose to upload an image or use your camera
3. Click **Identify Face** to start the matching process
4. View results in the **Results** tab

### Step 4: View and Filter Results

1. In the **Results** tab, you'll see all matches sorted by confidence
2. Use the confidence slider to filter results
3. For matches from group photos, the face will be highlighted in the original image

## Advanced Settings

### Model Selection

Choose from different face recognition models in the sidebar:

- **Facenet512**: Best overall accuracy (default)
- **ArcFace**: High accuracy but slower
- **VGG-Face**: Faster but less accurate
- **DeepFace**: Balance between speed and accuracy

### Matching Threshold

- Lower values (0.1-0.3): Very strict matching, fewer false positives
- Medium values (0.4-0.6): Balanced approach (default: 0.5)
- Higher values (0.7-0.9): More permissive matching, may include false positives

### Performance Settings

- **Parallel Workers**: Adjust the number of concurrent processes based on your hardware
- **Strict Matching**: Enable additional verification to reduce false positives

## Directory Structure

```
face-identification-app/
├── app.py              # Main application file
├── force_cpu.py        # CPU configuration for TensorFlow
├── requirements.txt    # Python dependencies
├── utils.py            # Utility functions
├── library/            # Photo library directory (created on first run)
│   ├── metadata/       # Face metadata and extracted faces
│   └── embeddings/     # Face embeddings for faster matching
```

## Troubleshooting

### Common Issues

1. **Face Detection Problems**
   - Ensure the face is clearly visible in the image
   - Try a different recognition model from the sidebar
   - Disable strict matching mode for difficult images

2. **Performance Issues**
   - Reduce the number of parallel workers in the sidebar
   - Close other memory-intensive applications
   - For slow systems, use smaller images in your library

3. **Installation Errors**
   - If you encounter issues with TensorFlow installation, try:
     ```bash
     pip install tensorflow-cpu==2.12.0
     ```
   - For dlib installation problems, see the platform-specific notes above

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [DeepFace](https://github.com/serengil/deepface) for the face recognition engine
- [Streamlit](https://streamlit.io/) for the web application framework
- [OpenCV](https://opencv.org/) for image processing capabilities
- [MTCNN](https://github.com/ipazc/mtcnn) for enhanced face detection

---

For questions or support, please open an issue on the GitHub repository.
