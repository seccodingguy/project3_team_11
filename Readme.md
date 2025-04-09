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


pip install -r requirements.txt

## Usage

### Clone the repository:

git clone https://github.com/your-username/image-classification.git
Install the required packages:

 
pip install -r requirements.txt

### Prepare your dataset: 

### Directory setup

1. If not using the provided data directory, create a directory that has 2 subfolders: book and vinyl
2. Depending on the Usage path, modify the referenced source file(s) to point to the directory where the book and vinyl subfolders are located

### Code execution:

#### OPTION 1:

#### Run the Test script

1. Open the test_orchestrator.py source file
2. Modify the configuration values 
3. Save and close the source file
4. python test_orchestrator.py

#### OPTION 2:

#### Run the scripts individually:

1. python download_images.py (if you want to create your own image set)
2. python fix_images_issues.py
3. python train.py --model resnet --dir_data data [if using the provided images] (or path/to/parentfolder/of/book/and/vinyl if you downloaded images)

#### OPTION 3: (Recommended)

#### Use the Flask Web App

1. python app.py
2. Open the link to the URL display
3. Modify any settings (or keep the default values)
4. Make sure all checkboxes are checked

## Model Architecture

The code provides two model architectures:

ResNet: A pre-trained ResNet model with a custom classification head.
MobileNet: A pre-trained MobileNet model with a custom classification head.

## Preprocessing

The code applies the following preprocessing techniques:

ResNet: Subtracts the ImageNet mean values [123.68, 116.779, 103.939].

MobileNet: Subtracts 127.5 and divides by 127.5.

## Training

The code trains the models using the Adam optimizer and categorical cross-entropy loss.

## Evaluation

The code evaluates the models using accuracy, precision, and recall metrics.

## Deployment

The code provides a simple way to deploy the trained models using TensorFlow Serving.

## Deploying the Model

To deploy the model, follow these steps:

1. Save the model: Save the trained model using orchestrator.save_model().
2. Create a TensorFlow Serving instance: Create a TensorFlow Serving instance using the tensorflow_serving package.
3. Configure the model: Configure the model to use the saved model and specify the input and output tensors.
4. Start the TensorFlow Serving instance: Start the TensorFlow Serving instance using tensorflow_serving.serve().
5. Test the deployment: Test the deployment by sending a request to the TensorFlow Serving instance using a tool like curl.

### Example Deployment Code

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

## Agents

The code provides six agents:

1. Agent 1: Download images
2. Agent 2: Preprocess images
3. Agent 3: Train model
4. Agent 4: Predict images
5. Agent 5: Create inventory

### Functions

Agent 1: Download Images
 
download_images(self, search_query, output_dir)
Downloads images for a given search query into a specified output directory.

### Agent 2: Preprocess Images
 
preprocess_images(self, directory)
Validates images in a given directory.

### Agent 3: Train Model
 
train_model(self, train_dir, model_name, socketio='http://127.0.0.1:5000', folds=3, layers=5, epocs=25, confidence=0.75)
Trains a model with data from a given directory.

### Agent 4: Predict Images
 
predict_images(self, image_dir, model_name, confidence, socketio='http://127.0.0.1:5000')
Predicts images in a given directory using a trained model.

### Agent 5: Create Inventory

create_inventory(self, image_dir, output_inventory_file)
Creates an inventory from images in a given directory.

## Orchestrator

The Orchestrator class is responsible for orchestrating the pipeline of agents.

### Orchestrator Functions

1. orchestrate_pipeline(self): Orchestrates the pipeline of agents.
2. get_training_results(self): Loads the training results from a file.
3. get_prediction_results(self): Loads the prediction results from a file.

## Server

The server is a Flask application that provides APIs for submitting jobs and handling socket.io events.

### Server Functions
1. /submit: Submits a job with the given configuration.
2. /api/images: Returns a list of image files in the plots directory.
3. /images/<filename>: Serves an image file from the plots directory.
4. /api/models: Returns a list of the best models trained during each run.
5. /api/model-info: Returns a model summary and details of each layer in the model.

## Socket.IO Singleton

The socket_io_singleton.py script provides a singleton Socket.IO instance.

## Contributing
Contributions are welcome! If you'd like to contribute to this repository, please:

* Fork the repository.
* Make your changes.
* Create a pull request.

## License

This repository is licensed under the MIT License. See LICENSE for details.
