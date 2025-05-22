from app.app import app, socketio
from app.app import app, socketio
from app.routes.criticalEvents import read_gps_data_from_serial, SERIAL_PORT_NAME, BAUDRATE
import threading
import os
import logging

# Within the application context, clear exports and create database tables
# with app.app_context():
#     clear_exports_directory()
#     db.create_all()

__all__ = ['app', 'socketio', 'read_gps_data_from_serial']

if __name__ == '__main__':
    # Start background GPS thread
    gps_thread = threading.Thread(target=read_gps_data_from_serial, args=(SERIAL_PORT_NAME, BAUDRATE))
    gps_thread.daemon = True
    gps_thread.start()

    # Run the Flask-SocketIO app
    socketio.run(app, debug=True)
