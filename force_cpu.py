# This file should be imported before any other imports in app.py
import os
import warnings

# Force CPU-only operation
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging (0=all, 1=INFO, 2=WARNING, 3=ERROR)

# Disable warnings (optional)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)