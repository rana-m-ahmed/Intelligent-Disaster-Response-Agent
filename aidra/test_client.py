import socketio
import time

sio = socketio.Client()

state = {}

@sio.event
def connect():
    print('connected')

@sio.on('state_update')
def on_state(data):
    global state
    state = data
    print('state_update received: victims=', [v['id'] for v in data.get('victims', [])])

@sio.on('route_planned')
def on_route(data):
    print('route_planned:', data)

@sio.on('event_triggered')
def on_event(data):
    print('event_triggered:', data)

@sio.on('victim_added')
def on_victim_added(data):
    print('victim_added:', data)

@sio.event
def disconnect():
    print('disconnected')

if __name__ == '__main__':
    sio.connect('http://localhost:5000')
    time.sleep(0.5)
    sio.emit('init_scenario', {'scenario': 'A'})
    time.sleep(0.5)

    # wait for state
    for _ in range(10):
        if state.get('victims'):
            break
        time.sleep(0.2)

    victims = state.get('victims', [])
    if not victims:
        print('No victims in state; aborting')
    else:
        vid = victims[0]['id']
        print('planning route to', vid)
        sio.emit('plan_route', {'victim_id': vid, 'algorithm': 'astar', 'alpha': 1.0})

    time.sleep(1)

    # step simulation a few times
    for i in range(8):
        sio.emit('step_simulation')
        time.sleep(0.3)

    # trigger a road block at (3,3)
    sio.emit('trigger_event', {'type': 'block', 'coords': [3,3]})
    time.sleep(0.5)

    # add a new victim
    sio.emit('trigger_event', {'type': 'new_victim', 'coords': [7,7]})
    time.sleep(0.5)

    # step more
    for i in range(8):
        sio.emit('step_simulation')
        time.sleep(0.3)

    # get final state
    sio.emit('get_state')
    time.sleep(0.5)

    sio.disconnect()
