import subprocess
import socketio
import time

class Agents:
    
    
    def __init__(self):
        print("Initialized Agents.")
    
    def download_images(self,search_query, output_dir):
        msg = f"Agent 1: Downloading images for '{search_query}' into '{output_dir}'..."
        print(msg)
        
        try:
            download_process = subprocess.run(
                ["python", "download_images.py"]
            )
            # Wait for the process to complete
            msg="Agent 1: Images downloaded successfully."
            print(msg)
            
            
        except subprocess.CalledProcessError as e:
            msg=f"Agent 1: Error downloading images: {e}"
            print(msg)
            
            
    def preprocess_images(self,directory):
        msg=f"Agent 2: Validating images in directory '{directory}'..."
        print(msg)
        
        try:
            preprocess_process = subprocess.run(
                ["python", "fix_image_issues.py"]
            )
            msg="Agent 2: Images validated successfully."
            print(msg)
            
        except subprocess.CalledProcessError as e:
            msg=f"Agent 2: Error validating images: {e}"
            print(msg)
            
            
    def train_model(self,train_dir, model_name, socketio='http://127.0.0.1:5000', folds=3, layers=5, epocs=25, confidence=0.75, use_xla=True):
        msg=f"Agent 3: Training model with data from '{train_dir}'..."
        print(msg)
               
        try:
            training_process = subprocess.run(
                ["python", "mobilenet_resnet_keras.py", '--dir_data', train_dir, '--model', model_name, 
                 '--folds', str(folds),'--layers',str(layers),'--epocs',str(epocs),
                 '--confidence',str(confidence), '--use_xla',str(use_xla), '--socketio',socketio]
            )
            # Wait for the process to complete
            #stdout, stderr = training_process.communicate()
            msg="Agent 3: Model training completed successfully."
            print(msg)
            
        except subprocess.CalledProcessError as e:
            msg=f"Agent 3: Error training model: {e}"
            print(msg)
            
    def predict_images(self,image_dir, model_name, confidence, socketio='http://127.0.0.1:5000',use_xla=True):
        msg=f"Agent 4: Predicting images in directory '{image_dir}' using model '{model_name}'..."
        print(msg)
        
        try:
            predict_process = subprocess.run(
                ["python", "mobilenet_resnet_keras.py", '--dir_data', image_dir, '--model', model_name, 
                 '--use_saved_model', str(True), '--confidence', str(confidence), '--use_xla',str(use_xla), 
                 '--socketio',socketio]
            )
            # Wait for the process to complete
            # stdout, stderr = predict_process.communicate()
            msg="Agent 4: Predictions completed successfully."
            print(msg)
            
        except subprocess.CalledProcessError as e:
            msg=f"Agent 4: Error predicting images: {e}"
            print(msg)
            
            
    def create_inventory(self,image_dir, output_inventory_file):
        msg=f"Agent 5: Creating inventory from images in '{image_dir}'..."
        print(msg)
        
        try:
            subprocess.run(
                ["python", "image_processing.py", image_dir, output_inventory_file],
                check=True
            )
            # Wait for the process to complete
            #stdout, stderr = inventory_process.communicate()
            msg="Agent 5: Inventory created successfully."
            print(msg)
            
        except subprocess.CalledProcessError as e:
            msg=f"Agent 5: Error creating inventory: {e}"
            print(msg)
            
            
           
