import os
import threading
import urllib.request
import time

from dominator_agent import DominatorAgent
from battlesnake_server import start_server

def warm_up():
    time.sleep(5)
    try:
        url = os.environ.get("RENDER_EXTERNAL_URL", "https://battlesnake-dominator.onrender.com/")
        urllib.request.urlopen(url)
        print("Warm-up ping successful!")
    except Exception as e:
        print(f"Warm-up ping failed: {e}")

threading.Thread(target=warm_up, daemon=True).start()

agent = DominatorAgent()
port = int(os.environ.get("PORT", 8000))
start_server(agent=agent, port=port)
