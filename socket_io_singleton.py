from flask_socketio import SocketIO

# A global variable to hold the singleton instance
_socketio_instance = None

def get_socketio(app=None, cors_allowed_origins="*"):
    """
    Returns the singleton SocketIO instance. If the instance doesn't exist,
    it creates one using the provided Flask app and sets the cors_allowed_origins.

    Args:
        app (Flask, optional): The Flask application instance to initialize SocketIO.
        cors_allowed_origins (str or list, optional): Origins allowed to make cross-origin requests.
            Defaults to "*", which allows all origins.

    Returns:
        SocketIO: The singleton SocketIO instance.
    """
    global _socketio_instance
    if _socketio_instance is None:
        if app is None:
            raise ValueError("SocketIO instance does not exist. Pass a Flask app to initialize it.")
        # Initialize the global SocketIO instance with CORS settings
        _socketio_instance = SocketIO(app, cors_allowed_origins=cors_allowed_origins)
    return _socketio_instance