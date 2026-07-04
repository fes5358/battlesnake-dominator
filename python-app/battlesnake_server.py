import logging
import os
import sys
import time

from flask import Flask, jsonify
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

    @app.get("/analyze")
    def on_analyze():
        """
        Analyze recent tournament games and return a self-improvement report.

        Query params:
          games=N     — number of recent games to analyze (default 30)
          apply=true  — apply recommended constant adjustments (default false)

        Example:
          GET /analyze
          GET /analyze?games=50
          GET /analyze?apply=true
        """
        sys.path.insert(0, os.path.dirname(__file__))
        from self_improve import TournamentAnalyzer

        n_games      = request.args.get("games", 30, type=int)
        apply_changes = request.args.get("apply", "false").lower() == "true"

        try:
            analyzer = TournamentAnalyzer()
            report   = analyzer.run_analysis(games=n_games)

            if apply_changes:
                adjs    = report.get("recommended_adjustments", [])
                applied = analyzer.apply_adjustments(adjs)
                report["applied_adjustments"] = [
                    {"constant": a["constant"], "old": a["current"], "new": a["new"]}
                    for a in applied
                ]
                if applied:
                    report["note"] = (
                        "Constants updated. Restart the Python App workflow "
                        "to activate changes."
                    )
            return jsonify(report)

        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    host = "0.0.0.0"

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    print(f"\nRunning Battlesnake at http://{host}:{port}")
    app.run(host=host, port=port)
