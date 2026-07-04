import os

from dominator_agent import DominatorAgent
from battlesnake_server import start_server

agent = DominatorAgent()
port = int(os.environ.get("PORT", 8000))

start_server(agent=agent, port=port)
