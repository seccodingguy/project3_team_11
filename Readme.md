# Image Classification with ResNet and MobileNet

This repository provides a simple and efficient way to train and deploy image classification models using ResNet and MobileNet architectures.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Agents](#agents)
- [Functions](#functions)
- [Orchestrator](#orchestrator)
- [Server](#server)
- [Socket.IO Singleton](#socketio-singleton)
- [Download Images](#download-images)
- [Fix Image Issues](#fix-image-issues)
- [index.html](#indexhtml)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This repository provides a basic implementation of image classification using ResNet and MobileNet architectures. The code is designed to be easy to use and modify, making it a great starting point for your own image classification projects.

## Requirements

- **Python** 3.8+
- **TensorFlow** 2.4+
- **Keras** 2.4+
- **NumPy** 1.20+
- **Matplotlib** 3.4+
- **SocketIO**
- **Flask**
- **DuckDuckGo Search**

You can install the required packages using pip (after you clone the repository):

Copy
pip install -r requirements.txt
bash

## Usage
Clone the repository:
bash

Copy
git clone https://github.com/your-username/image-classification.git
Install the required packages:
bash

Copy
pip install -r requirements.txt
Prepare your dataset: Create a directory with two subdirectories, train and validation, each containing images for the respective classes.
Run the training script:
bash

Copy
python train.py --model resnet --dir_data /path/to/dataset
Model Architecture
The code provides two model architectures:

ResNet: A pre-trained ResNet model with a custom classification head.
MobileNet: A pre-trained MobileNet model with a custom classification head.
Preprocessing
The code applies the following preprocessing techniques:

ResNet: Subtracts the ImageNet mean values [123.68, 116.779, 103.939].
MobileNet: Subtracts 127.5 and divides by 127.5.
Training
The code trains the models using the Adam optimizer and categorical cross-entropy loss.

Evaluation
The code evaluates the models using accuracy, precision, and recall metrics.

Deployment
The code provides a simple way to deploy the trained models using TensorFlow Serving.

Deploying the Model
To deploy the model, follow these steps:

Save the model: Save the trained model using orchestrator.save_model().
Create a TensorFlow Serving instance: Create a TensorFlow Serving instance using the tensorflow_serving package.
Configure the model: Configure the model to use the saved model and specify the input and output tensors.
Start the TensorFlow Serving instance: Start the TensorFlow Serving instance using tensorflow_serving.serve().
Test the deployment: Test the deployment by sending a request to the TensorFlow Serving instance using a tool like curl.
Example Deployment Code
python

Run

Copy
import tensorflow_serving

# Save the model
orchestrator.save_model()

# Create a TensorFlow Serving instance
serving_config = tensorflow_serving.ServingConfig(
    model_config_list=[
        tensorflow_serving.ModelConfig(
            name='image-classification',
            base_path='/path/to/model'
        )
    ]
)

# Start the TensorFlow Serving instance
tensorflow_serving.serve(serving_config)
Agents
The code provides six agents:

Agent 1: Download images
Agent 2: Preprocess images
Agent 3: Train model
Agent 4: Predict images
Agent 5: Create inventory
Agent 6: Job search
Functions
Agent 1: Download Images
python

Run

Copy
download_images(self, search_query, output_dir)
Downloads images for a given search query into a specified output directory.

Agent 2: Preprocess Images
python

Run

Copy
preprocess_images(self, directory)
Validates images in a given directory.

Agent 3: Train Model
python

Run

Copy
train_model(self, train_dir, model_name, socketio='http://127.0.0.1:5000', folds=3, layers=5, epocs=25, confidence=0.75)
Trains a model with data from a given directory.

Agent 4: Predict Images
python

Run

Copy
predict_images(self, image_dir, model_name, confidence, socketio='http://127.0.0.1:5000')
Predicts images in a given directory using a trained model.

Agent 5: Create Inventory
python

Run

Copy
create_inventory(self, image_dir, output_inventory_file)
Creates an inventory from images in a given directory.

Agent 6: Job Search
python

Run

Copy
job_search(self, query, location, num_of_results=20)
Searches for open jobs using a given query in a specified location.

Orchestrator
The Orchestrator class is responsible for orchestrating the pipeline of agents.

Orchestrator Functions
orchestrate_pipeline(self): Orchestrates the pipeline of agents.
get_training_results(self): Loads the training results from a file.
get_prediction_results(self): Loads the prediction results from a file.
Server
The server is a Flask application that provides APIs for submitting jobs and handling socket.io events.

Server Functions
/submit: Submits a job with the given configuration.
/api/images: Returns a list of image files in the plots directory.
/images/<filename>: Serves an image file from the plots directory.
Socket.IO Singleton
The socket_io_singleton.py script provides a singleton Socket.IO instance.

Socket.IO Singleton Function
python

Run

Copy
get_socketio(app=None, cors_allowed_origins="*")
Returns the singleton Socket.IO instance.

Download Images
The download_images.py script downloads images from DuckDuckGo.

Download Images Function
python

Run

Copy
search_and_download_images_duckduckgo(query, num_images, save_folder)
Downloads images for a given search query into a specified output directory.

Fix Image Issues
The fix_image_issues.py script fixes image format issues using TensorFlow and PIL.

Fix Image Issues Function
python

Run

Copy
clean(dir, remove=True, convert=True, move=None)
Fixes image format issues in a given directory.

index.html
The index.html file provides a user interface for configuring and running the orchestrator pipeline.

Contributing
Contributions are welcome! If you'd like to contribute to this repository, please:

Fork the repository.
Make your changes.
Create a pull request.
License
This repository is licensed under the MIT License. See LICENSE for details.