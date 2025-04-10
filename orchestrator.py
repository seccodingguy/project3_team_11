import os
import json
import time
import socketio
from agents import Agents as agents
import pickle

class Orchestrator:
    def __init__(self, orchestrator_config_settings):
        config_settings = orchestrator_config_settings
        self._search_query = config_settings['search_query']
        self._download_dir = config_settings['download_dir']
        #self._validated_dir = config_settings['validated_dir']
        self._train_dir = config_settings['train_dir']
        #self._validation_dir = config_settings['validation_dir']
        self._model_output_path = config_settings['model_dir']
        self._test_images_dir = config_settings['test_dir']
        self._prediction_output_file = config_settings['prediction_dir']
        #self._inventory_output_file = config_settings['inventory_dir']
        self._model_to_use = config_settings['model_name']
        self._number_of_layers = int(config_settings['layers'])
        self._download_images = bool(config_settings['download_images'])
        self._validate_images = bool(config_settings['validate_images'])
        self._train_model = bool(config_settings['train_model'])
        self._predict_images = bool(config_settings['predict_images'])
        #self._create_inventory = bool(config_settings['create_inventory'])
        self._confidence = config_settings['confidence']
        self._predict_dir = config_settings['predict_dir']
        self._folds = config_settings['folds']
        self._epocs = config_settings['epocs']
        self._use_xla = bool(config_settings['use_xla'])
        self._socket_io = config_settings['socket_url']
        
           
    def orchestrate_pipeline(self):
        # Ensure directories exist
        os.makedirs(self._download_dir, exist_ok=True)
        #os.makedirs(self._validated_dir, exist_ok=True)
        os.makedirs(self._train_dir, exist_ok=True)
        #os.makedirs(self._validation_dir, exist_ok=True)
        os.makedirs(self._test_images_dir, exist_ok=True)
        socket_io = socketio.Client()
        
        try:
            socket_io.connect(self._socket_io)
            socket_io.emit('agent_status', {'message': "Pipeline execution started."})
            time.sleep(1)
        except Exception as e:
        #    print(f"Failed to connect to the server: {e}")
            socket_io = None  # Ensure socket_io is None if connection fails

        agents_obj = agents()
        
        print(f"Download images value = {self._download_images}")
        
        if self._download_images:
            if socket_io:
                socket_io.emit('agent_status', {'message': "Step: Download images."})
                time.sleep(1)
            # Step: Download images
            agents_obj.download_images(self._search_query, self._download_dir)

        if self._validate_images:
            # Step: Validate images
            if socket_io:
                socket_io.emit('agent_status', {'message': "Step: Validate images."})
                time.sleep(1)
            agents_obj.preprocess_images(self._train_dir)

        if self._train_model:
            # Step: Train the model
            if socket_io:
                socket_io.emit('agent_status', {'message': "Step: Train the model."})
                time.sleep(1)
            agents_obj.train_model(self._train_dir, self._model_to_use, 
                               folds=self._folds, layers=self._number_of_layers, epocs=self._epocs, 
                               confidence=self._confidence,use_xla=self._use_xla,socketio=self._socket_io)

        if self._predict_images:
            # Step: Predict image classes
            if socket_io:
                socket_io.emit('agent_status', {'message': "Step: Predict the images."})
                time.sleep(1)
            agents_obj.predict_images(self._predict_dir, self._model_to_use, confidence=self._confidence,socketio=self._socket_io)
'''
FUTURE RELEASE

        if self._create_inventory:
            # Step: Create inventory
            agents_obj.create_inventory(self._test_images_dir, self._inventory_output_file)
        
        try:
            socket_io.emit('agent_status', {'message': "Pipeline execution completed successfully."})
            time.sleep(1)
        except Exception as e:
            print(f"Failed to connect to the server: {e}")
            socket_io = None  # Ensure socket_io is None if connection fails
        #finally:
        #    socket_io.disconnect()
'''
