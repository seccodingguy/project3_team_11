import os
import requests
from duckduckgo_search import DDGS
import argparse
import socketio
import time

socket_io = socketio.Client()

def send_status_message(msg,reciever='status_message'):
    if socket_io and socket_io.connected:
        socket_io.emit(reciever, {'message': msg})
        time.sleep(1)
    else:
        print("SocketIO client is not connected. Cannot send message.")


def search_and_download_images_duckduckgo(query, num_images, save_folder):
        
    # Create the folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    send_status_message(f"Searching for '{query}' images on DuckDuckGo...")
    # Use DuckDuckGo to search for image URLs
    # print(f"Searching for '{query}' images on DuckDuckGo...")

    with DDGS() as ddgs:
        results = ddgs.images(query, max_results=num_images)
        count = 0
        #ddgs_images_list = []
        # Download images
        for index, result in enumerate(results):
            try:
                image_url = result["image"]
                response = requests.get(image_url, stream=True)
                response.raise_for_status()

                # Save the image to the folder
                file_path = os.path.join(
                    save_folder, f"{query}_{index + 1}.jpg")
                with open(file_path, "wb") as f:
                    f.write(response.content)

                print(f"Image {index + 1} saved successfully: {file_path}")
                count = count + 1
            except Exception as e:
                print(f"Failed to download image {index + 1}: {e}")

    # print(f'Total images saved for {query} = {count}')
    send_status_message(f'Total images saved for {query} = {count}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Download vinyl records and books images to use to train the model.')
    parser.add_argument('--vinyl_download_dir', default="data/train/vinyl", 
                        help='Directory to save the downloaded vinyl images to. Default is /data/train/vinyl')
    parser.add_argument('--book_download_dir', default="data/train/book", 
                        help='Directory to save the downloaded book images to. Default is /data/train/book')
    parser.add_argument('--number_to_download', default='150',
                        help='Number of images to download. Default is 150.')
    parser.add_argument('--socketio', default='http://127.0.0.1:5000', help="Socket to send responses to.")
    
    args = parser.parse_args()
    
    # Connect to the SocketIO server
    try:
        socket_io.connect('http://127.0.0.1:5000')  # Replace with your server's address
    except socketio.exceptions.ConnectionError as e:
        print(f"Failed to connect to the server in mobilenet_resnet_keras: {e}")
        socket_io = None  # Ensure socket_io is None if connection fails
    
    # Get random images of Vinyl records and Books
    search_query = "vinyl records"
    num_images_to_download = int(args.number_to_download)  # Number of images to download
    output_folder = args.vinyl_download_dir  # Folder to save the images
    os.makedirs(output_folder, exist_ok=True)
    search_and_download_images_duckduckgo(
        search_query, num_images_to_download, output_folder)

    search_query = "books"
    output_folder = args.book_download_dir  # Folder to save the images
    os.makedirs(output_folder, exist_ok=True)
    search_and_download_images_duckduckgo(
        search_query, num_images_to_download, output_folder)
