from extensions import socketio
from app import create_app
import os
app = create_app()

@socketio.on('connect')
def handle_connect():
    print("Client connecté")

if __name__ == "__main__":
    #port = int(os.environ.get("PORT", 10000))
    #socketio.run(app, host="0.0.0.0", port=port)
    socketio.run(app, debug=True)