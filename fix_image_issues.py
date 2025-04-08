import os
import tensorflow as tf
import argparse
from PIL import Image
import imghdr
import shutil
import socketio
import time

socket_io = socketio.Client()

def send_status_message(msg,reciever='status_message'):
    if socket_io and socket_io.connected:
        socket_io.emit(reciever, {'message': msg})
        time.sleep(1)
    else:
        print("SocketIO client is not connected. Cannot send message.")


def clean(dir="data", remove=True, convert=True, move=None):

    # Find problematic files
    problematic_files = []

    for root, _, files in os.walk(dir):
        for filename in files:
            if filename.startswith('.'):  # Skip hidden files
                continue

            file_path = os.path.join(root, filename)

            try:
                # Check actual file type
                img_type = imghdr.what(file_path)

                # If not a recognized image type
                if img_type not in ['jpeg', 'png', 'gif', 'bmp']:
                    # Try with PIL
                    try:
                        with Image.open(file_path) as img:
                            img_format = img.format
                            if img_format not in ['JPEG', 'PNG', 'GIF', 'BMP']:
                                problematic_files.append(
                                    (file_path,
                                     f"Unsupported format: {img_format}"))
                    except Exception as e:
                        problematic_files.append(
                            (file_path, f"Not a valid image: {str(e)}"))

                # Try loading with TensorFlow
                try:
                    img_data = tf.io.read_file(file_path)
                    if img_type == 'jpeg':
                        tf.image.decode_jpeg(img_data)
                    elif img_type == 'png':
                        tf.image.decode_png(img_data)
                    elif img_type == 'gif':
                        tf.image.decode_gif(img_data)
                    elif img_type == 'bmp':
                        tf.image.decode_bmp(img_data)
                    else:
                        problematic_files.append(
                            (file_path, "Fails TensorFlow decoding"))
                except Exception as e:
                    problematic_files.append(
                        (file_path, f"TensorFlow error: {str(e)}"))

            except Exception as e:
                problematic_files.append((file_path, f"Error: {str(e)}"))

    # Report findings
    send_status_message(f"\nFound {len(problematic_files)} problematic files")
    #print(f"\nFound {len(problematic_files)} problematic files")

    if problematic_files:
        #print("\nProblematic files:")
        send_status_message("\nProblematic files:")
        for path, reason in problematic_files:
            #print(f"- {path}: {reason}")
            send_status_message(f"- {path}: {reason}")

        # Handle problematic files
        for path, reason in problematic_files:
            if convert:
                try:
                    # Try to convert to PNG
                    with Image.open(path) as img:
                        new_path = f"{os.path.splitext(path)[0]}.png"
                        img.save(new_path, 'PNG')
                        #print(f"Converted: {path} → {new_path}")
                        send_status_message(f"Converted: {path} → {new_path}")

                        # Remove original only if conversion succeeded
                        if os.path.exists(new_path):
                            os.remove(path)
                except Exception as e:
                    #print(f"Failed to convert {path}: {str(e)}")
                    send_status_message(f"Failed to convert {path}: {str(e)}")

                    # Handle conversion failures
                    if remove:
                        try:
                            os.remove(path)
                            #print(f"Removed: {path}")
                            send_status_message(f"Removed: {path}")
                        except Exception as e:
                            #print(f"Failed to remove {path}: {str(e)}")
                            send_status_message(f"Failed to remove {path}: {str(e)}")
                    elif move:
                        try:
                            dest = os.path.join(
                                move, os.path.basename(path))
                            shutil.move(path, dest)
                            #print(f"Moved: {path} → {dest}")
                            send_status_message(f"Moved: {path} → {dest}")
                        except Exception as e:
                            #print(f"Failed to move {path}: {str(e)}")
                            send_status_message(f"Failed to move {path}: {str(e)}")
            elif remove:
                try:
                    os.remove(path)
                    #print(f"Removed: {path}")
                    send_status_message(f"Removed: {path}")
                except Exception as e:
                    print(f"Failed to remove {path}: {str(e)}")
            elif move:
                try:
                    dest = os.path.join(
                        move, os.path.basename(path))
                    shutil.move(path, dest)
                    #print(f"Moved: {path} → {dest}")
                    send_status_message(f"Moved: {path} → {dest}")
                except Exception as e:
                    print(f"Failed to move {path}: {str(e)}")

    #print("\nScan and fix complete.")
    send_status_message("\nScan and fix complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fix image format issues with TensorFlow')
    parser.add_argument('--dir', default="data", help='Directory to scan')
    parser.add_argument('--remove', default=True,
                        help='Remove problematic files')
    parser.add_argument('--convert', default=True,
                        help='Convert to compatible formats')
    parser.add_argument(
        '--move', help='Move problematic files to this directory')

    args = parser.parse_args()
    
    # Connect to the SocketIO server
    try:
        socket_io.connect('http://127.0.0.1:5000')  # Replace with your server's address
    except socketio.exceptions.ConnectionError as e:
        print(f"Failed to connect to the server in mobilenet_resnet_keras: {e}")
        socket_io = None  # Ensure socket_io is None if connection fails

    if args.move and not os.path.exists(args.move):
        os.makedirs(args.move, exist_ok=True)

    if not os.path.exists(args.dir):
        print(f"Directory {args.dir} does not exist")
    else:
        print(f"Scanning directory: {args.dir}")
        send_status_message(f"Scanning directory: {args.dir}")
        clean()
