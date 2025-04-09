'''
Description: The following code will use a GPU to load, validate, and train a MobileNetV2 or a ResNet50 base model
to train and save a new model to perform image recognization and classify an image as either a vinyl record or
a book.

Purpose: The purpose of the model is to more efficiently and expeditiously create a list of images with vinyl records
and books to use is an AI pipeline to extract the text from the image, search the web for the publisher details, determine 
estimated value. The list will be used in a pipeline action that will create an inventory of the vinyl records and records
in a MySQL database. This program is the first step in the pipeline process.

Author: Mark Wireman

Pre-requisites:
    1) A Linux environment (will also work with WSL2).
    2) A CUDA recognized GPU.
    3) Follow the commands to execute in the miniconda_setup.sh file.
    4) Images in the data directory (see download_images.py for details).
    5) Test images in the test_images directory.
    
User Inputs:
    1) Base model to use: MobileNetV2 or ResNet50
    2) Number of layers
    3) Number of epocs
'''

import warnings
import argparse
import traceback
import time
import socketio
from PIL import Image, UnidentifiedImageError, ImageFile
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.layers import Layer
from sklearn.model_selection import KFold
from datetime import datetime
import shutil
import math
import random
import tensorflow as tf
import json
import os


# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['MLIR_CRASH_REPRODUCER_DIRECTORY'] = 'enable' # Enable MLIR crash directory

# Enable XLA for GPU with shared memory optimization and auto-clustering
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'

#print("Before getting the socketio.Client()")
# SocketIO client instance
socket_io = socketio.Client()
#print("After getting the socketio.Client()")

@socket_io.on('connect')
def on_connect():
    print("Connected to the server.")

@socket_io.on('disconnect')
def on_disconnect():
    print("Disconnected from the server.")
    
# Custom preprocessing layer that becomes part of the model
@register_keras_serializable(package="Custom")
class PreprocessingLayer(tf.keras.layers.Layer):
    """Custom preprocessing layer that can be part of the model."""
    
    def __init__(self, model_type="mobilenet", **kwargs):
        super(PreprocessingLayer, self).__init__(**kwargs)
        self.model_type = model_type
    
    def call(self, inputs):
        if self.model_type == 'resnet':
            # ResNet preprocessing
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3])
            return tf.subtract(inputs, mean)
        else:
            # MobileNet preprocessing
            scaled_inputs = tf.subtract(inputs, 127.5)
            return tf.divide(scaled_inputs, 127.5)
    
    def get_config(self):
        config = super(PreprocessingLayer, self).get_config()
        config.update({"model_type": self.model_type})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def send_status_message(msg,reciever='status_message'):
    if socket_io and socket_io.connected:
        socket_io.emit(reciever, {'message': msg})
        time.sleep(1)
    else:
        print("SocketIO client is not connected. Cannot send message.")

# Fix for libdevice not found error and setup GPU with shared memory
def setup_gpu():
    """Configure environment for TensorFlow GPU usage with shared memory optimization."""
    # Path to CUDA installation - adjust this based on output from fix_tensorflow_gpu.py
    cuda_paths = [
        '/usr/local/cuda',
        '/usr/lib/cuda',
        '/home/mwireman/miniconda3/pkgs/cuda-nvcc-11.8.89-0',
        '/usr/local/cuda-11.8',
    ]

    # Find valid CUDA path
    cuda_path = None
    for path in cuda_paths:
        if os.path.exists(path):
            cuda_path = path
            break

    if cuda_path:
        # Set XLA_FLAGS to point to CUDA directory
        os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir={cuda_path}'
        msg = f"Set XLA_FLAGS to use CUDA at {cuda_path}"
        print(msg)
        
        send_status_message(msg)

        # Set environment variables for GPU memory optimization
        # Use private thread pool
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        # Use async malloc for better memory management
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

        # Control the GPU memory fraction allocated to TensorFlow (slightly below max to allow for shared memory)
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

        # Enable cudnn autotuner which selects optimal algorithms with shared memory usage
        os.environ['TF_CUDNN_USE_AUTOTUNE'] = '1'
    else:
        # Disable XLA JIT compilation as fallback
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
        msg = "CUDA not found. Disabled XLA devices."
        print(msg)

    # Set library duplicates OK
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Reduce TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    return cuda_path is not None


# Run GPU setup
gpu_available = setup_gpu()

try:
    # Configure GPU memory growth (prevents TensorFlow from taking all GPU memory)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        msg = f"Found {len(gpus)} GPU(s)"
        print(msg)
        #send_status_message(msg)
        for gpu in gpus:
            msg = f"  Name: {gpu.name}, Type: {gpu.device_type}"
            print(msg)
            #send_status_message(msg)
            try:
                # Enable memory growth for dynamic allocation
                tf.config.experimental.set_memory_growth(gpu, True)
                msg = f"  Memory growth enabled for {gpu.name}"
                print(msg)
                #send_status_message(msg)

                # Set virtual device configuration with limited memory per GPU
                # This allows better management of shared memory
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=4096)]
                )
                msg = f"  Memory limit set for {gpu.name}"
                print(msg)
                #send_status_message(msg)
            except RuntimeError as e:
                msg = f"  Error configuring GPU: {e}"
                print(msg)
                #send_status_message(msg)
    else:
        msg = "No GPUs found, using CPU"
        print(msg)
        #send_status_message(msg)
except Exception as e:
    msg = f"Error configuring GPUs: {e}"
    print(msg)
    #send_status_message(msg)


# Set PIL to be more tolerant of image files
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Filter out specific PIL warnings if desired
warnings.filterwarnings("ignore", category=UserWarning,
                        message="Palette images with Transparency.*")

# Create compiled TF functions that utilize GPU shared memory through XLA
@tf.function(jit_compile=True)  # Enable XLA optimization (uses shared memory)
def preprocess_batch(images):
    """
    GPU-optimized preprocessing using XLA and shared memory
    
    This function uses XLA compilation to optimize the preprocessing pipeline.
    XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear
    algebra that optimizes TensorFlow computations.
    
    Benefits:
    1. Faster execution through fusion of operations
    2. Improved memory usage through shared memory on GPU
    3. Reduced overhead by minimizing host-device communication
    4. Optimized kernel selection for the specific hardware
    
    Args:
        images: A batch of images as a tensor with shape [batch_size, height, width, channels]
        
    Returns:
        Preprocessed images tensor optimized for the model
    """
    # Convert to float32 if needed
    if images.dtype != tf.float32:
        images = tf.cast(images, tf.float32)
    
    # MobileNetV2 preprocessing:
    # 1. Scales values to [-1, 1]
    # 2. Does not include normalization by imagenet mean/std as it's built into the model
    scaled_images = tf.subtract(images, 127.5)
    return tf.divide(scaled_images, 127.5)

@tf.function(jit_compile=True)
def preprocess_batch_resnet(images):
    """
    GPU-optimized preprocessing for ResNet using XLA and shared memory
    
    ResNet uses a different preprocessing approach than MobileNet:
    - Subtracts ImageNet mean values
    - Scales by ImageNet standard deviation
    
    Args:
        images: A batch of images as a tensor with shape [batch_size, height, width, channels]
        
    Returns:
        Preprocessed images tensor optimized for ResNet
    """
    # Convert to float32 if needed
    if images.dtype != tf.float32:
        images = tf.cast(images, tf.float32)
    
    # RGB ImageNet mean values
    mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3])
    
    # Subtract mean (ResNet preprocessing doesn't divide by std in the preprocessing step)
    return tf.subtract(images, mean)

def get_preprocessing_function(model_type):
    """
    Returns the appropriate preprocessing function based on model type
    
    Args:
        model_type: String indicating the model type ('mobilenet' or 'resnet')
        
    Returns:
        The appropriate preprocessing function
    """
    if model_type.lower() == 'resnet':
        return preprocess_batch_resnet
    else:  # default to mobilenet
        return preprocess_batch


@tf.function(jit_compile=True)  # Enable XLA optimization (uses shared memory)
def resize_images(images, target_size=(224, 224)):
    """GPU-optimized image resizing using XLA and shared memory"""
    return tf.image.resize(images, target_size)

# Enhanced model creation with shared memory optimizations
def create_model(num_classes=2, base_model_name="mobilenet", no_of_layers=2, use_xla=True):
    base_model = None
    inputShape = (224, 224, 3)
    
    # Choose base model
    if base_model_name.lower() == 'resnet':
        base_model = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=inputShape
        )
    else:  # mobilenet
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=inputShape
        )
    
    
    # Fine-tune the top layers of the base model
    for layer in base_model.layers[:-20]:  # Freeze only early layers
        layer.trainable = False
    
    # Create model
    inputs = tf.keras.Input(shape=inputShape)
    
    # Apply custom preprocessing layer
    x = PreprocessingLayer(model_type=base_model_name.lower())(inputs)
    
    # Base model
    x = base_model(x, training=False)
    
    # Pooling layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    counter = 1
    
    # Add custom layers
    for i in range(no_of_layers):
        if counter < 4:
            units = 2048 // (2**min(counter, 3))  # Decrease units as layers increase, for 3 layers, then repeat
            counter = counter + 1
        else:
            counter = 1
            
        x = tf.keras.layers.Dense(units, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        msg = f"Created Layer {i} with units of {units}."
        print(msg)
        send_status_message(msg)

        # Final classification layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    msg = f"{base_model_name}_xla_{use_xla}"
    print(msg)
    send_status_message(msg)
    # Assemble model
    model = tf.keras.Model(inputs, outputs, name=f"{base_model_name}_xla_{use_xla}")
    
    # Compile model with metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    return model

# Function to check if an image file is valid
def is_valid_image(img_path):
    try:
        with Image.open(img_path) as img:
            img.verify()  # Verify image integrity
        return True
    except Exception as e:
        print(f"Invalid image file {img_path}: {str(e)} will be removed.")
        os.remove(img_path)
        print(f"{img_path} has been deleted successfully.")
        return False

# Function to properly load and convert image with transparency handling
def load_and_convert_image(img_path):
    """Load image and properly handle palette images with transparency"""
    try:
        # Open the image
        img = Image.open(img_path)

        # Check if image is a palette image with transparency
        if img.mode == 'P':
            # Convert to RGBA first to properly handle transparency
            # This addresses the "Palette images with Transparency" warning
            img = img.convert('RGBA')

        # Then convert to RGB (removing alpha channel) for model input
        if img.mode != 'RGB':
            img = img.convert('RGB')

        return img
    except Exception as e:
        print(f"Error loading image {img_path}: {str(e)}")
        raise e

# Optimized batch image loading for GPU
def load_images_batch(img_paths, batch_size=16):
    """Load and prepare a batch of images for processing with GPU shared memory"""
    images = []
    valid_paths = []

    for path in img_paths[:batch_size]:
        try:
            if is_valid_image(path):
                img = load_and_convert_image(path)
                img = img.resize((224, 224), Image.LANCZOS)
                img_array = np.array(img, dtype=np.float32)

                # Handle grayscale images
                if len(img_array.shape) == 2:
                    img_array = np.stack((img_array,) * 3, axis=-1)
                # Handle RGBA by removing alpha channel
                elif img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]

                images.append(img_array)
                valid_paths.append(path)
        except Exception as e:
            print(f"Error loading image {path}: {e}")

    if not images:
        return None, []

    # Convert to tensor
    image_batch = tf.convert_to_tensor(np.array(images), dtype=tf.float32)

    # Use GPU-optimized preprocessing with shared memory
    preprocessed_batch = preprocess_batch(image_batch)

    return preprocessed_batch, valid_paths

# Optimized function to preprocess a single image using GPU
def preprocess_image(img_path):
    """Process a single image with GPU optimization and shared memory"""
    try:
        # First check if the file exists
        if not os.path.exists(img_path):
            print(f"Error: Image file does not exist: {img_path}")
            # Return a blank image array as fallback
            return np.zeros((1, 224, 224, 3))

        # Try to open and verify the image
        try:
            with Image.open(img_path) as img_check:
                img_check.verify()
        except Exception as e:
            print(f"Error verifying image {img_path}: {e}")
            print(f"Exception type: {type(e).__name__}")
            # Return a blank image array as fallback
            return np.zeros((1, 224, 224, 3))

        # If verification passed, load and process the image
        # Use our custom function to handle palette images with transparency
        img = load_and_convert_image(img_path)

        # Resize the image
        img = img.resize((224, 224), Image.LANCZOS)

        # Convert to array
        img_array = np.array(img)

        # Make sure image has 3 channels (RGB)
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            # Convert grayscale to RGB if needed
            if len(img_array.shape) == 2:
                img_array = np.stack((img_array,) * 3, axis=-1)
            # Handle RGBA by removing alpha channel
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]

        # Convert to tensor
        img_tensor = tf.convert_to_tensor(
            np.expand_dims(img_array, axis=0), dtype=tf.float32)

        # Use the GPU-optimized preprocessing function
        preprocessed = preprocess_batch(img_tensor)

        return preprocessed
    except UnidentifiedImageError as e:
        print(f"Cannot identify image file: {img_path}")
        print(f"Error details: {str(e)}")
        # Return a blank image array as fallback
        return np.zeros((1, 224, 224, 3))
    except Exception as e:
        print(f"Error processing image {img_path}: {str(e)}")
        print(f"Exception type: {type(e).__name__}")
        traceback.print_exc()
        # Return a blank image array as fallback
        return np.zeros((1, 224, 224, 3))

# Function to validate all images in a directory
def validate_images(directory, batch_size):
    """Check all images in the directory for corruption or invalid formats"""
    print(f"Validating images in {directory}...")

    valid_extensions = ['.jpg', '.jpeg', '.png',
                        '.bmp', '.tiff', '.webp', '.gif', '.JPG']

    # Get all files with valid extensions
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                image_files.append(os.path.join(root, file))

    print(f"Found {len(image_files)} image files")

    # Process images in batches for efficiency
    
    # batch_size = 32  # Larger batch size for validation only
    invalid_files = []
    palette_transparency_files = []

    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]

        # Process each image in the batch
        for img_path in batch_files:
            try:
                # Try to open and verify the image
                with Image.open(img_path) as img:
                    # Check if the image can be verified
                    img.verify()

                    # Reopen to check dimensions and mode (verify closes the file)
                    with Image.open(img_path) as img:
                        if img.width == 0 or img.height == 0:
                            print(
                                f"Warning: Image {img_path} has invalid dimensions: {img.width}x{img.height}")
                            invalid_files.append(
                                (img_path, "Invalid dimensions"))

                        # Check for palette image with transparency
                        if img.mode == 'P':
                            try:
                                # Try to see if the palette has transparency
                                transparency = img.info.get('transparency')
                                if transparency is not None:
                                    print(
                                        f"Note: Palette image with transparency: {img_path}")
                                    palette_transparency_files.append(img_path)
                                    # This is not considered invalid, just noted
                            except Exception:
                                pass
            except Exception as e:
                print(f"Warning: Could not open or verify {img_path}: {e}")
                invalid_files.append((img_path, str(e)))

    # Report results
    if invalid_files:
        print(f"\nFound {len(invalid_files)} problematic files:")
        for path, error in invalid_files:
            print(f"  - {path}: {error}")

        # Write report to file
        report_path = os.path.join(os.path.dirname(directory) if os.path.dirname(
            directory) else '.', 'invalid_images_report.txt')
        with open(report_path, 'w') as f:
            f.write(
                f"Invalid images report - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Directory: {directory}\n")
            f.write(f"Total images checked: {len(image_files)}\n")
            f.write(f"Invalid images found: {len(invalid_files)}\n\n")
            f.write("Details:\n")
            for path, error in invalid_files:
                f.write(f"  - {path}: {error}\n")

            if palette_transparency_files:
                f.write(
                    f"\nPalette images with transparency (valid but need special handling): {len(palette_transparency_files)}\n")
                for path in palette_transparency_files:
                    f.write(f"  - {path}\n")

        print(f"Report saved to {report_path}")

        return False, invalid_files
    else:
        if palette_transparency_files:
            print(
                f"\nFound {len(palette_transparency_files)} palette images with transparency (valid but need special handling).")
            # Write report just for palette images
            report_path = os.path.join(os.path.dirname(directory) if os.path.dirname(
                directory) else '.', 'palette_transparency_images.txt')
            with open(report_path, 'w') as f:
                f.write(
                    f"Palette images report - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Directory: {directory}\n")
                f.write(
                    f"Palette images with transparency: {len(palette_transparency_files)}\n\n")
                for path in palette_transparency_files:
                    f.write(f"  - {path}\n")
            print(f"Palette images report saved to {report_path}")

        print("All images are valid!")
        return True, []

# Optimize the model prediction function with XLA compilation
@tf.function(jit_compile=True)  # Enable XLA optimization for shared memory
def predict_optimized(model, img_array):
    """GPU-optimized prediction using XLA and shared memory"""
    return model(img_array, training=False)


# Define training step as a compiled function for better GPU usage
@tf.function(jit_compile=True)
def train_step(inputs, labels):
    # Compute class weights if needed
    class_weights = None
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
        # Apply class weights if available
        if class_weights:
            # Convert class indices to weights
            sample_weights = tf.ones_like(labels[:, 0])
            for class_idx, weight in class_weights.items():
                mask = tf.cast(tf.argmax(labels, axis=1)
                               == class_idx, tf.float32)
                sample_weights = sample_weights * (1 - mask) + mask * weight
            loss = loss * sample_weights
        loss = tf.reduce_mean(loss)

    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, predictions

# Define preprocessing function


def preprocess_img(image, label):
    """Preprocesses an image for MobileNetV2"""
    # Resize and preprocess
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image, label

# Define augmentation function
def augment(image, label):
    """Apply data augmentation to the image"""
    # Random flips, rotations, etc.
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Random brightness, contrast, etc.
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)

    # Ensure pixel values are still in valid range after augmentations
    image = tf.clip_by_value(image, -1.0, 1.0)
    return image, label

# Progress tracking callback
class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = f"Epoc {epoch+1}: loss={logs.get('loss', 0):.4f}, acc={logs.get('accuracy', 0):.4f}"
        print(msg)
        send_status_message(msg)


def train_model(model, data_dir, epochs=15, batch_size=16, k_folds=5, use_xla=True, save_model=True, layers=4):
    start_time = time.time()
    
    send_status_message("*** Start: Training Model ***")
    
    # Step 1: Enable XLA optimization if requested
    send_status_message("Step 1: Enable XLA optimization if requested")
    if use_xla:
        tf.config.optimizer.set_jit(True)
        msg = "XLA optimization enabled"
        print(msg)
        send_status_message(msg)
    
    # Step 2: Create a timestamp for this run
    send_status_message("Step 2: Create a timestamp for this run")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Step 3: Create directory structure for the K-Fold datasets
    send_status_message("Step 3: Create directory structure for the K-Fold datasets")
    base_fold_dir = f"tmp/kfold_{timestamp}"
    os.makedirs(base_fold_dir, exist_ok=True)
    
    # Directory for the test set
    test_dir = os.path.join(base_fold_dir, "test")
    os.makedirs(test_dir, exist_ok=True)
    
    # Step 4: Parse the dataset structure and count files
    send_status_message("Step 4: Parse the dataset structure and count files")
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    class_names = sorted(class_dirs)
    msg = f"Found {len(class_names)} classes: {class_names}"
    print(msg)
    send_status_message(msg)
    
    # Count files in each class
    class_files = {}
    for cls in class_names:
        class_path = os.path.join(data_dir, cls)
        files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f)) and
                 f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        
        # Limit samples per class if specified
        #if max_samples_per_class and len(files) > max_samples_per_class:
        #    files = random.sample(files, max_samples_per_class)
        #    print(f"Limiting class '{cls}' to {max_samples_per_class} samples")
            
        class_files[cls] = files
        msg = f"Class '{cls}': {len(files)} files"
        print(msg)
        send_status_message(msg)
    
    # Step 5: Create train-test split at the file level
    send_status_message("Step 5: Create train-test split at the file level")
    train_files = {}
    test_files = {}
    
    for cls, files in class_files.items():
        # Shuffle files
        random.shuffle(files)
        
        # Calculate split index
        test_count = math.ceil(len(files) * 0.15)
        
        # Split files
        test_files[cls] = files[:test_count]
        train_files[cls] = files[test_count:]
        
        msg = f"Class '{cls}': {len(train_files[cls])} training files, {len(test_files[cls])} test files"
        print(msg)
        send_status_message(msg)
    
    # Step 6: Copy test files to test directory
    send_status_message("Step 6: Copy test files to test directory")
    mnsg = "Creating test dataset..." 
    print(msg)
    send_status_message(msg)
    for cls, files in test_files.items():
        cls_dir = os.path.join(test_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        
        for file in files:
            src = os.path.join(data_dir, cls, file)
            dst = os.path.join(cls_dir, file)
            shutil.copy2(src, dst)
    
    # Step 7: Setup K-Fold Cross-Validation at the file level
    send_status_message("Step 7: Setup K-Fold Cross-Validation at the file level")
    fold_dirs = []
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Create a list of all training files with their class labels
    all_files = []
    for cls, files in train_files.items():
        for file in files:
            all_files.append((cls, file))
    
    # Shuffle all files
    random.shuffle(all_files)
    
    # Split files into folds
    fold_indices = list(kf.split(all_files))
    
    # Step 8: Create directories for each fold
    send_status_message("Step 8: Create directories for each fold")
    msg = f"Creating {k_folds} fold directories..."
    print(msg)
    send_status_message(msg)
    for fold in range(k_folds):
        fold_dir = os.path.join(base_fold_dir, f"fold_{fold+1}")
        fold_dirs.append(fold_dir)
        
        # Create train and validation directories for this fold
        train_dir = os.path.join(fold_dir, "train")
        val_dir = os.path.join(fold_dir, "val")
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # Create class subdirectories
        for cls in class_names:
            os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
            os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        
        # Get train and validation indices for this fold
        train_idx, val_idx = fold_indices[fold]
        
        # Copy files to train directory
        for idx in train_idx:
            cls, file = all_files[idx]
            src = os.path.join(data_dir, cls, file)
            dst = os.path.join(train_dir, cls, file)
            shutil.copy2(src, dst)
        
        # Copy files to validation directory
        for idx in val_idx:
            cls, file = all_files[idx]
            src = os.path.join(data_dir, cls, file)
            dst = os.path.join(val_dir, cls, file)
            shutil.copy2(src, dst)
        
        msg = f"Fold {fold+1}: {len(train_idx)} training files, {len(val_idx)} validation files"
        print(msg)
        send_status_message(msg)
        
        # Step 9: Create data generators with augmentation
        send_status_message("Step 9: Create data generators with augmentation")
    def create_data_generators(augment=True):
        if augment:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True
            )
        else:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
            
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
        return train_datagen, val_datagen
    
    # Step 10: Perform K-Fold Cross-Validation
    send_status_message("Step 10: Perform K-Fold Cross-Validation")
    msg = f"Starting {k_folds}-fold cross-validation..."
    print(msg)
    send_status_message(msg)
    
    # Storage for fold results
    fold_histories = []
    fold_models = []
    fold_scores = []
    fold_val_accuracies = []
    fold_val_losses = []
    input_shape=(224, 224, 3)
    
    for fold in range(k_folds):
        msg = f"\nTraining fold {fold+1}/{k_folds}"
        print(msg)
        send_status_message(msg)
        fold_dir = fold_dirs[fold]
        
        # Create data generators
        train_datagen, val_datagen = create_data_generators(augment=True)
        
        # Create flow_from_directory generators
        train_generator = train_datagen.flow_from_directory(
            os.path.join(fold_dir, "train"),
            target_size=input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            os.path.join(fold_dir, "val"),
            target_size=input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Create a new model for this fold
        fold_model = create_model(base_model_name=model,no_of_layers=layers,use_xla=use_xla) #create_model_architecture()
        
        # Setup callbacks
        fold_callbacks = []
        
        # Add early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        fold_callbacks.append(early_stopping)
        
        # Add learning rate scheduler
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        fold_callbacks.append(reduce_lr)
        
        # Add model checkpoint for this fold
        checkpoint_path = os.path.join(fold_dir, "checkpoint")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_path, "best_model.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        fold_callbacks.append(checkpoint)
        
        callbacks = [TrainingProgressCallback()]
        verbose = 1
        
        # Add any additional callbacks
        if callbacks:
            fold_callbacks.extend(callbacks)
        
        # Train the model
        fold_history = fold_model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=fold_callbacks,
            verbose=verbose
        )
        
        # Evaluate on this fold's validation data
        val_scores = fold_model.evaluate(val_generator, verbose=0)
        val_loss = val_scores[0]
        val_acc = val_scores[1]
        
        msg = f"Fold {fold+1} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}"
        print(msg)
        send_status_message(msg)
        
        # Store fold results
        fold_histories.append(fold_history.history)
        fold_models.append(fold_model)
        fold_scores.append(val_scores)
        fold_val_accuracies.append(val_acc)
        fold_val_losses.append(val_loss)
    
    # Step 11: Calculate average cross-validation metrics
    send_status_message("Step 11: Calculate average cross-validation metrics")
    avg_val_accuracy = np.mean(fold_val_accuracies)
    avg_val_loss = np.mean(fold_val_losses)
    
    msg = f"\nCross-Validation Results:"
    print(msg)
    send_status_message(msg)
    msg = f"Average Validation Loss: {avg_val_loss:.4f}"
    print(msg)
    send_status_message(msg)
    msg = f"Average Validation Accuracy: {avg_val_accuracy:.4f}"
    print(msg)
    send_status_message(msg)
    
    # Step 12: Find the best fold model
    send_status_message("Step 12: Find the best fold model")
    best_fold_idx = np.argmin(fold_val_losses)
    best_fold_model = fold_models[best_fold_idx]
    
    msg = f"Best model from fold {best_fold_idx+1} with validation loss: {fold_val_losses[best_fold_idx]:.4f}"
    print(msg)
    send_status_message(msg)
    
    # Step 13: Optional - Train a final model on all training data
    send_status_message("Step 13: Optional - Train a final model on all training data")
    msg = "\nTraining final model on all training data..."
    print(msg)
    send_status_message(msg)
    
    # Create a directory for the full training dataset (all train data, no validation split)
    full_train_dir = os.path.join(base_fold_dir, "full_train")
    os.makedirs(full_train_dir, exist_ok=True)
    
    # Create class subdirectories
    for cls in class_names:
        os.makedirs(os.path.join(full_train_dir, cls), exist_ok=True)
    
    # Copy all training files to this directory
    for cls, files in train_files.items():
        for file in files:
            src = os.path.join(data_dir, cls, file)
            dst = os.path.join(full_train_dir, cls, file)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
    
    # Create data generators for final training
    final_train_datagen, _ = create_data_generators(augment=True)
    
    final_train_generator = final_train_datagen.flow_from_directory(
        full_train_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    final_model = create_model(base_model_name = model,no_of_layers=layers,use_xla=use_xla) # create_model_architecture()
    
    # Setup callbacks for final training
    final_callbacks = []
    
    # Add early stopping based on training loss
    final_early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        restore_best_weights=True
    )
    final_callbacks.append(final_early_stopping)
    
    # Add model checkpoint
    final_checkpoint_path = os.path.join(base_fold_dir, "final_checkpoint")
    os.makedirs(final_checkpoint_path, exist_ok=True)
    
    final_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(final_checkpoint_path, "best_model.keras"),
        monitor='loss',
        save_best_only=True,
        verbose=1
    )
    final_callbacks.append(final_checkpoint)
    
    final_callbacks.extend(callbacks)
    
    # Train the final model
    final_history = final_model.fit(
        final_train_generator,
        epochs=epochs,
        callbacks=final_callbacks,
        verbose=verbose
    )
    
    # Step 14: Evaluate on the held-out test set
    send_status_message("Step 14: Evaluate on the held-out test set")
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Evaluate best fold model
    msg = "\nEvaluating best fold model on test set..."
    print(msg)
    send_status_message(msg)
    best_model_test_scores = best_fold_model.evaluate(test_generator, verbose=1)
    best_model_test_loss = best_model_test_scores[0]
    best_model_test_acc = best_model_test_scores[1]
    
    msg = f"Best Fold Model - Test Loss: {best_model_test_loss:.4f}, Test Accuracy: {best_model_test_acc:.4f}"
    print(msg)
    send_status_message(msg)
    
    # Reset generator for final model evaluation
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Evaluate final model
    msg = "\nEvaluating final model on test set..."
    print(msg)
    send_status_message(msg)
    final_test_scores = final_model.evaluate(test_generator, verbose=1)
    final_test_loss = final_test_scores[0]
    final_test_acc = final_test_scores[1]
    
    msg = f"Final Model - Test Loss: {final_test_loss:.4f}, Test Accuracy: {final_test_acc:.4f}"
    print(msg)
    send_status_message(msg)
    
    # Step 15: Save models if requested
    send_status_message("Step 15: Save models if requested")
    if save_model:
        # Create models directory
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Save the best cross-validation model
        best_model_path = os.path.join(models_dir, f"best_cv_{model}_{layers}_{timestamp}.keras")
        best_fold_model.save(best_model_path)
        best_model_path = os.path.join(models_dir, f"best_cv_{model}_{layers}_{timestamp}.h5")
        best_fold_model.save(best_model_path)
        msg = f"Best CV model saved to {best_model_path}"
        print(msg)
        send_status_message(msg)
        
        # Save the final model trained on all data
        final_model_path = os.path.join(models_dir, f"final_{model}_{layers}_{timestamp}.keras")
        final_model.save(final_model_path)
        final_model_path = os.path.join(models_dir, f"final_{model}_{layers}_{timestamp}.h5")
        final_model.save(final_model_path)
        msg = f"Final model saved to {final_model_path}"
        print(msg)
        send_status_message(msg)
        
        # Save class names for inference
        class_names_path = os.path.join(models_dir, f"class_names_{timestamp}.txt")
        with open(class_names_path, 'w') as f:
            f.write('\n'.join(class_names))
    
    # Step 16: Visualize cross-validation results
    send_status_message("Step 16: Visualize cross-validation results")
    plt.figure(figsize=(15, 10))
    
    # Plot validation loss for each fold
    plt.subplot(2, 2, 1)
    for i, history in enumerate(fold_histories):
        plt.plot(history['val_loss'], label=f'Fold {i+1}')
    plt.title(f"Validation Loss by Fold for best_cv_{model}_{layers}_{timestamp}.keras")
    plt.xlabel('Epoc')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation accuracy for each fold
    plt.subplot(2, 2, 2)
    for i, history in enumerate(fold_histories):
        plt.plot(history['val_accuracy'], label=f'Fold {i+1}')
    plt.title(f"Validation Accuracy by Fold for final_{model}_{layers}_{timestamp}.keras")
    plt.xlabel('Epoc')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot average metrics
    plt.subplot(2, 2, 3)
    # Get the minimum history length across all folds
    min_length = min([len(history['loss']) for history in fold_histories])
    
    # Truncate histories to the same length
    truncated_train_loss = [history['loss'][:min_length] for history in fold_histories]
    truncated_val_loss = [history['val_loss'][:min_length] for history in fold_histories]
    
    # Calculate average metrics
    avg_train_loss = np.mean(truncated_train_loss, axis=0)
    avg_val_loss = np.mean(truncated_val_loss, axis=0)
    
    plt.plot(avg_train_loss, label='Avg Train Loss')
    plt.plot(avg_val_loss, label='Avg Validation Loss')
    plt.title(f"Average Training and Validation Loss for final_{model}_{layers}_{timestamp}.keras")
    plt.xlabel('Epoc')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot final model metrics
    plt.subplot(2, 2, 4)
    plt.plot(final_history.history['loss'], label='Training Loss')
    plt.axhline(y=final_test_loss, color='r', linestyle='-', label=f'Test Loss: {final_test_loss:.4f}')
    plt.title(f'Final Model Training and Test Loss for final_{model}_{layers}_{timestamp}.keras')
    plt.xlabel('Epoc')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save or show the figure
    plt.tight_layout()
    
    save_model = True
    
    if save_model:
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        plots_path = os.path.join(plots_dir, f"training_plots_{timestamp}.png")
        plt.savefig(plots_path)
        msg = f"Training plots saved to {plots_path}"
        print(msg)
        send_status_message(msg)
    
    #plt.show()
    plt.close()
    # Step 17: Clean up temporary directories (optional)
    send_status_message("Step 17: Clean up temporary directories (optional)")
    # Uncomment to enable cleanup:
    f"Cleaning up temporary directories in {base_fold_dir}..."
    print(msg)
    send_status_message(msg)
    shutil.rmtree(base_fold_dir)
    
    # Calculate total training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    msg = f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s"
    print(msg)
    send_status_message(msg)
    
    # Return comprehensive results
    return {
        'fold_histories': fold_histories,
        'fold_models': fold_models,
        'fold_scores': fold_scores,
        'best_fold_index': best_fold_idx,
        'best_fold_model': best_fold_model,
        'best_fold_val_loss': fold_val_losses[best_fold_idx],
        'best_fold_val_accuracy': fold_val_accuracies[best_fold_idx],
        'final_model': final_model,
        'final_history': final_history.history,
        'best_model_test_loss': best_model_test_loss,
        'best_model_test_accuracy': best_model_test_acc,
        'final_test_loss': final_test_loss,
        'final_test_accuracy': final_test_acc,
        'cross_val_avg_loss': avg_val_loss[-1],  # Last epoch average loss
        'cross_val_avg_accuracy': avg_val_accuracy,
        'class_names': class_names,
        'training_time': total_time,
        'timestamp': timestamp,
        'model_paths': {
            'best_cv_model': best_model_path if save_model else None,
            'final_model': final_model_path if save_model else None,
            'class_names': class_names_path if save_model else None
        } if save_model else None
    }

@register_keras_serializable()
class PreprocessingLayer(Layer):
    def __init__(self, model_type='mobilenet', **kwargs):
        super(PreprocessingLayer, self).__init__(**kwargs)
        self.model_type = model_type

    def call(self, inputs):
        # Add your preprocessing logic here
        return inputs

def predict_image(model, image_path, class_names = ['book','vinyl'], use_xla=True, confidence_threshold=0.7):
    """
    Predicts the class of an image using the trained model.

    Args:
        model: The trained TensorFlow model
        image_path: Path to the image file
        class_names: List of class names (default: ['book', 'vinyl'])

    Returns:
        Predicted class and confidence
    """
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    try:
    
        # Make prediction with XLA optimization
        if use_xla:
            predictions = predict_optimized(model, img_array).numpy()
        else:
            predictions = model.predict(img_array)
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Map to class name
        if predicted_class_idx < len(class_names):
            predicted_class = class_names[predicted_class_idx]
        else:
            predicted_class = f"Class {predicted_class_idx}"
        
        # Apply confidence threshold
        if confidence < confidence_threshold * 100:
            predicted_class = "uncertain"
        
        return predicted_class, confidence

    except Exception as e:
        import traceback
        print(f"Error in predict_image function for {image_path}: {str(e)}")
        traceback.print_exc()
        return None, 0.0


# Function to perform batch prediction using shared memory
def predict_batch(model, images, confidence_threshold=0.70, batch_size=16):
    """Predict a batch of images using GPU shared memory optimization"""
    results = []
    current_time = time.time()
    
    for i in range(0, len(images), batch_size):
        batch_paths = images[i:i+batch_size]

        # Load and preprocess batch
        preprocessed_batch, valid_paths = load_images_batch(
            batch_paths, batch_size)

        if preprocessed_batch is None or len(valid_paths) == 0:
            # Skip this batch if no valid images
            continue

        # Run optimized prediction on the batch
        predictions = predict_optimized(model, preprocessed_batch)
        predictions_np = predictions.numpy()

        # Process each prediction
        for j, path in enumerate(valid_paths):
            class_idx = np.argmax(predictions_np[j])
            confidence = predictions_np[j][class_idx] * 100

            # Map index to class name - MODIFIED FOR TWO CLASSES
            class_names = ['book', 'vinyl record']
            predicted_class = class_names[class_idx]

            # Apply confidence threshold
            if confidence < confidence_threshold * 100:
                predicted_class = "uncertain"

            # Store result
            results.append({
                'path': path,
                'predicted_class': predicted_class,
                'confidence': confidence
            })

    print(f"\nPrediction results:\n\t{results}\n\n")
    send_status_message(results)
    with open(f'prediction_results_{current_time}.txt', 'w') as file:
        file.write(f"\nPrediction results:\n\t{results}\n\n")
    
    return results

# Function to display an image for visualization with proper transparency handling
def load_image_for_display(img_path):
    """Load an image for display purposes, handling transparency properly"""
    try:
        # First check if the file exists
        if not os.path.exists(img_path):
            print(f"Error: Display image file does not exist: {img_path}")
            return None

        # Load the image handling palette transparency
        img = load_and_convert_image(img_path)
        return img
    except Exception as e:
        print(f"Error loading display image {img_path}: {str(e)}")
        return None

# Benchmark function to measure performance improvements with shared memory
def benchmark_gpu_performance(model, img_paths, runs=5, batch_sizes=[1, 4, 8, 16, 32, 64]):
    """Benchmark GPU performance with different batch sizes to show shared memory benefits"""
    if not img_paths:
        print("No images provided for benchmarking")
        return
    
    current_time = time.time()
    filetxt = ""

    # Limit to first 100 images for benchmarking
    test_paths = img_paths[:100] if len(img_paths) > 100 else img_paths

    print("\n=== GPU SHARED MEMORY PERFORMANCE BENCHMARK ===")
    filetxt = f"{filetxt}\n=== GPU SHARED MEMORY PERFORMANCE BENCHMARK ==="
    print(
        f"Testing with {len(test_paths)} images, {runs} runs per configuration")
    filetxt = f"{filetxt}Testing with {len(test_paths)} images, {runs} runs per configuration"
    results = {}

    # Test individual image processing (no shared memory benefit)
    print("\nBenchmarking individual image processing (baseline)...")
    filetxt = f"{filetxt}\nBenchmarking individual image processing (baseline)..."
    start_time = time.time()
    for _ in range(runs):
        for path in test_paths[:10]:  # Limit to 10 images for individual processing
            _ = predict_image(model, path)
    individual_time = (time.time() - start_time) / (10 * runs)
    print(
        f"Average time per image (individual): {individual_time*1000:.2f} ms")
    filetxt = f"{filetxt}Average time per image (individual): {individual_time*1000:.2f} ms"
    results['individual'] = individual_time * 1000

    # Test batch processing with different batch sizes
    for batch_size in batch_sizes:
        print(f"\nBenchmarking batch size {batch_size}...")
        filetxt = f"{filetxt}\nBenchmarking batch size {batch_size}..."
        start_time = time.time()
        for _ in range(runs):
            _ = predict_batch(model, test_paths, batch_size=batch_size)
        batch_time = (time.time() - start_time) / (len(test_paths) * runs)
        print(
            f"Average time per image (batch size {batch_size}): {batch_time*1000:.2f} ms")
        filetxt = f"{filetxt}Average time per image (batch size {batch_size}): {batch_time*1000:.2f} ms"
        results[f'batch_{batch_size}'] = batch_time * 1000

        # Calculate speedup
        speedup = individual_time / batch_time
        print(f"Speedup with batch size {batch_size}: {speedup:.2f}x")
        filetxt = f"{filetxt}Speedup with batch size {batch_size}: {speedup:.2f}x"
        results[f'speedup_{batch_size}'] = speedup

    # Find the best batch size
    best_batch_size = batch_sizes[0]
    best_speedup = results[f'speedup_{batch_sizes[0]}']

    for batch_size in batch_sizes[1:]:
        if results[f'speedup_{batch_size}'] > best_speedup:
            best_speedup = results[f'speedup_{batch_size}']
            best_batch_size = batch_size

    print(
        f"\nBest performance with batch size {best_batch_size}: {best_speedup:.2f}x speedup")
    filetxt = f"{filetxt}\nBest performance with batch size {best_batch_size}: {best_speedup:.2f}x speedup"
    print("Larger batch sizes benefit more from GPU shared memory optimization")
    filetxt = f"{filetxt}Larger batch sizes benefit more from GPU shared memory optimization"
    
    with open(f'prediction_results_{current_time}.txt', 'w') as file:
        file.write(filetxt)
        file.write("\nRaw Data")
        file.write("\n***************************\n")
        file.write(f"\nResults:\n\t{results}")
        file.write(f"\n\nBest Batch Size:\n\t{best_batch_size}")
        file.write("\n***************************\n")

    return results, best_batch_size

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='Predict the images.')
    parser.add_argument('--dir_data', required=True, help='Directory of the images to use in training.')
    parser.add_argument('--model', required=True, help='Name the trained model to use: resnet or mobilenet.')
    parser.add_argument('--confidence', default='0.7',
                        help='Confidence level to use for the prediction. Default is 0.7 (70%).')
    parser.add_argument('--folds', default='3',
                        help='Number of folds to use in k-Folds.')
    parser.add_argument('--layers', default='5', help='Number of layers to use per epoc.')
    parser.add_argument('--epocs', default='25', help='Number of epocs.')
    parser.add_argument('--use_saved_model', default=False, help='Use the saved model to predict images.')
    parser.add_argument('--model_path', default='vinyl_book_classifier_v2.keras', help="Path to the saved model.")
    parser.add_argument('--socketio', default='http://127.0.0.1:5000', help="Socket to send responses to.")
    
    
    args = parser.parse_args()
    
    # Directories for training and validation data
    train_dir = args.dir_data #'data/train'
    
    model_name = args.model
    
    # Connect to the SocketIO server
    try:
        socket_io.connect('http://127.0.0.1:5000')  # Replace with your server's address
    except socketio.exceptions.ConnectionError as e:
        print(f"Failed to connect to the server in mobilenet_resnet_keras: {e}")
        socket_io = None  # Ensure socket_io is None if connection fails
    
    # Check if model exists, if not, train a new one
    model_path = args.model_path  # Updated to .keras format

    confidence = float(args.confidence)
    folds = int(args.folds)
    layers = int(args.layers)
    epocs = int(args.epocs)
    use_saved_model = bool(args.use_saved_model)
    
     # Adjust batch size for better shared memory utilization
    optimal_batch_size = 8  # Default value, will be adjusted based on GPU

    # If we have a decent GPU, use a larger batch size
    if gpus and len(gpus) > 0:
        try:
            results, optimal_batch_size = benchmark_gpu_performance(train_dir) # 16
        except:
            # If we get any error, fall back to smaller batch size
            optimal_batch_size = 8

    # Force retraining with 2 classes
    if os.path.exists(model_path) and bool(args.use_saved_model):  # Set to True to use saved model
        print("Loading existing model...")
        model = tf.keras.models.load_model(model_path,
                                           custom_objects={'PreprocessingLayer': PreprocessingLayer},
                                           compile=False)
        
        # Recompile the model to ensure XLA optimization is enabled for shared memory usage
        try:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy'],
                jit_compile=True  # Enable XLA optimization for shared memory
            )
        except:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy'],
                jit_compile=False  # Disable XLA optimization for shared memory
            )
            msg = f"Error with XLA - disabling jit compile."
            print(msg)
        
        files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f)) and
                 f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        
        print(f"Files list: {files}")
        
        prediction_results = predict_batch(model, files, confidence_threshold=confidence,batch_size=optimal_batch_size)
        send_status_message("*** PREDICTION RESULTS ***",reciever='results_status')
        send_status_message(json.dumps(prediction_results),reciever='results_status')
    else:
        print("Training new model with 2 classes (book, vinyl)...")
        # Check if we have data in all required folders
        required_dirs = [
            os.path.join(train_dir, 'book'),
            os.path.join(train_dir, 'vinyl')
        ]

        all_dirs_have_data = True
        
        for d in required_dirs:
            if not os.path.exists(d):
                # Create directories if they don't exist
                try:
                    os.makedirs(d, exist_ok=True)
                    print(f"Created directory: {d}")
                except Exception as e:
                    print(f"Error creating directory {d}: {e}")

            # Check if directory has image files
            if not os.path.exists(d) or len([f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) == 0:
                print(
                    f"Warning: Directory {d} is missing or empty of image files.")
                all_dirs_have_data = False

        if all_dirs_have_data:
            # Create and train the model
            training_results = train_model(
                model_name, train_dir, epochs=epocs, batch_size=optimal_batch_size, layers=layers,k_folds=folds)

            # Save the model
            training_results['final_model'].save(model_path)  # Using .keras format
            send_status_message("*** TRAINING RESULTS ***",reciever='results_status')
            send_status_message(json.dumps(training_results['fold_histories']),reciever='results_status')
            send_status_message(json.dumps(training_results['fold_scores']),reciever='results_status')
    
    if socket_io:     
        socket_io.disconnect()