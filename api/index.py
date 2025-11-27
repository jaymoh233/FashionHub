# app.py
import sys
import os
from dotenv import load_dotenv

# --- FIX: Explicitly find .env in the project root ---
# Get the directory where index.py is located (e.g., /project/api)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root (e.g., /project)
project_root = os.path.dirname(current_dir)
# Construct the full path to .env
env_path = os.path.join(project_root, '.env')

# Load the specific .env file
load_dotenv(dotenv_path=env_path)

# 2. Then set up paths
sys.path.append(project_root)

from flask import Flask
from flask_cors import CORS
import logging
from routes.register_routes import register_routes
# Importing this runs its code immediately, so env vars must be loaded already
from services.chat_api import chat_api
import time

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Init app
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
templates_path = os.path.join(base_dir, "templates")

app = Flask(__name__, template_folder=templates_path)
CORS(app)

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