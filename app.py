# app.py
import os
import random
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable
import tensorflow as tf
from flask_socketio import emit
import numpy as np
from orchestrator import Orchestrator
from socket_io_singleton import get_socketio

app = Flask(__name__)
socketio = get_socketio(app,cors_allowed_origins="*")

# Directory containing images
IMAGE_DIR = "plots"
MODEL_DIR = "models"

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

@app.route('/',methods=["GET"])
def index():
    return render_template('index.html')

@app.route('/api/images', methods=['GET'])
def get_images():
    """
    Get a list of all image files in the IMAGE_DIR directory.
    """
    try:
        # List all files in the directory
        image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        return jsonify({'images': image_files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/images/<filename>')
def serve_image(filename):
    """
    Serve an image file from the IMAGE_DIR directory.
    """
    return send_from_directory(IMAGE_DIR, filename)

@app.route('/api/models', methods=['GET'])
def list_models():
    """
    Get a list of all Keras model files in the MODEL_DIR directory.
    """
    try:
        # List all model files in the directory
        model_files_keras = [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras')]
        model_files_h5 = [f for f in os.listdir(MODEL_DIR) if f.endswith('.h5')]
        model_files = [item for sublist in [model_files_keras, model_files_h5] for item in sublist]
        return jsonify({'models': model_files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['POST'])
def get_model_info():
    """
    Load the selected model and return its summary and layers.
    """
    try:
        # Get the model filename from the request
        data = request.json
        print(data)
        model_filename = data.get('model_filename')
        if not model_filename:
            return jsonify({'error': 'Model filename is required.'}), 400

        # Ensure the model file exists
        model_path = os.path.join(MODEL_DIR, model_filename)
        print(model_path)
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model file not found.'}), 404

        # Load the model
        model = load_model(model_path, custom_objects={"PreprocessingLayer": PreprocessingLayer})

        # Get the model summary as a string
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        model_summary = "\n".join(summary_lines)
        
        # Access shapes
        print("Model input shape:", model.input_shape)  # Input shape of the model
        print("Model output shape:", model.output_shape)  # Output shape of the model

        # Get information about the layers
        layers = [{'name': layer.name, 'type': layer.__class__.__name__, 'output_shape': None}
                  for layer in model.layers]

        return jsonify({'summary': model_summary, 'layers': layers})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/submit', methods=['POST'])
def submit():
    try:
        data = request.json
        print(data)
        orchestrator_config = {
            'search_query': data.get('search_query'),
            'download_dir': data.get('download_dir'),
            #'validated_dir': data.get('validated_dir'),
            'train_dir': data.get('train_dir'),
            #'validation_dir': data.get('validation_dir'),
            'model_dir': data.get('model_dir'),
            'test_dir': data.get('test_dir'),
            'prediction_dir': data.get('prediction_dir'),
            'model_name': data.get('model_name'),
            #'inventory_dir': data.get('inventory_dir'),
            'layers': int(data.get('layers')),
            'confidence': float(data.get('confidence')),
            'folds': int(data.get('folds')),
            'epocs': int(data.get('epocs')),
            'use_saved_model': str(data.get('use_saved_model')) == 'True',
            'predict_dir': data.get('predict_dir'),
            'download_images': str(data.get('download_images')) == 'True',
            'validate_images': str(data.get('validate_images')) == 'True',
            'train_model': str(data.get('train_model')) == 'True',
            'predict_images': str(data.get('predict_images')) == 'True',
            #'create_inventory': str(data.get('create_inventory')) == 'True'
        }
        orchestrator = Orchestrator(orchestrator_config)
        orchestrator.orchestrate_pipeline()
        return "DONE"
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

@socketio.on('status_message')
def handle_status_message(data):
    emit('status_message',data,broadcast=True)
    #print(f"Received status message: {data}")
    
@socketio.on('agent_status')
def handle_status_message(data):
    emit('agent_status',data,broadcast=True)
    
@socketio.on('results_status')
def handle_status_message(data):
    emit('results_status',data,broadcast=True)
    
    #print(f"Received status message: {data}")


if __name__ == '__main__':
    socketio.run(app, debug=True)