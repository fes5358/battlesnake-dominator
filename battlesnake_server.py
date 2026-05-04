import logging
import time

from flask import Flask
from flask import request

from battlesnake_types import GameState, BaseAgent


@staticmethod
def start_server(agent: BaseAgent, port):
    if port is None:
        raise ValueError('please select your port')

    app = Flask("Battlesnake")

    @app.get("/")
    def on_info():
        # TIP: If you open your Battlesnake URL in browser you should see this data
        data = {
            "author": agent.get_author(),
            "color": agent.get_color(),
            "head": agent.get_head(),
            "tail": agent.get_tail(),
        }

        # filter None values
        data = {k: v for k, v in data.items() if v is not None}

        if 'kilab' in request.args:
            name = agent.get_name()
            data['name'] = name

        return data

    @app.post("/start")
    def on_start():
        """start is called when your Battlesnake begins a game"""
        data = request.get_json()
        game_state = GameState(**data)
        agent.start(game_state)
        print("START")
        return "ok"

    @app.post("/move")
    def on_move():
        """move is called on every turn and returns your next move"""
        start = time.time()
        data = request.get_json()
        game_state = GameState(**data)
        move = agent.move(game_state)

        print(f"MOVE: {move}, {time.time() - start}")
        return move.model_dump()

    @app.post("/end")
    def on_end():
        """end is called when your Battlesnake finishes a game"""
        data = request.get_json()
        game_state = GameState(**data)
        agent.end(game_state)
        print("END")
        return "ok"

    host = "0.0.0.0"

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    print(f"\nRunning Battlesnake at http://{host}:{port}")
    app.run(host=host, port=port)
