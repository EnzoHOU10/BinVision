from extensions import socketio
from app import create_app

app = create_app()

@socketio.on('connect')
def handle_connect():
    print("Client connecté")

if __name__ == "__main__":
    socketio.run(app, debug=True)
