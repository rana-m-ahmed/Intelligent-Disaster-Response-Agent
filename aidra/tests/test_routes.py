import os
import sys
import time

import pytest

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

from app import app, socketio


def test_index_route():
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200


def test_socket_init_scenario():
    # Use the SocketIO test client to exercise init_scenario and state update
    test_client = socketio.test_client(app)
    assert test_client.is_connected()

    test_client.emit("init_scenario", {"scenario": "A"})
    # allow server processing
    time.sleep(0.1)
    received = test_client.get_received()
    # expect at least one state_update or ml_report event
    event_names = {msg["name"] for msg in received}
    assert "state_update" in event_names or "ml_report" in event_names

    test_client.disconnect()
