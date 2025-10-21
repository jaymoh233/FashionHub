# app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import logging
from routes.register_routes import register_routes
from services.chat_api import chat_api
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Init app
# app = Flask(__name__)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
templates_path = os.path.join(base_dir, "templates")

app = Flask(__name__, template_folder=templates_path)
CORS(app)
load_dotenv()

# Register routes
register_routes(app)

# Setup directory
documents_dir = 'documents'
os.makedirs(documents_dir, exist_ok=True)

# Initialize Chat API
chat_api.start_time = time.time()
success = chat_api.setup(documents_dir)


if success:
    logger.info("Chat API setup complete. Server starting...")

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
else:
    logger.error("Failed to initialize Chat API")
