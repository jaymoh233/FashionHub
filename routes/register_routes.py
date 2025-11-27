#
#
# # routes.py
#
# from flask import request, jsonify, render_template
# from services.chat_api import chat_api
# import os
# import tempfile
# import logging
# from werkzeug.utils import secure_filename
#
# logger = logging.getLogger(__name__)
#
#
# def register_routes(app):
#     @app.route('/')
#     def home():
#         # Pass frontend keys securely
#         return render_template('index.html',
#                                supabase_url=os.environ.get("SUPABASE_URL"),
#                                supabase_key=os.environ.get("SUPABASE_ANON_KEY")
#                                )
#
#     @app.route('/api/query', methods=['POST'])
#     def query():
#         try:
#             data = request.json
#             question = data.get('question')
#             session_id = data.get('session_id', 'default')
#
#             # --- NEW: Get the chat_id from frontend ---
#             chat_id = data.get('chat_id')
#
#             if not question:
#                 return jsonify({"error": "No question provided"}), 400
#
#             # Pass chat_id to the service layer so it can save to DB
#             response = chat_api.get_intelligent_response(question, session_id, chat_id)
#             return jsonify(response)
#
#         except Exception as e:
#             logger.error(f"Query error: {str(e)}")
#             return jsonify({"error": str(e)}), 500
#
#     @app.route('/api/feedback', methods=['POST'])
#     def feedback():
#         try:
#             data = request.json
#             question = data.get('question')
#             answer = data.get('answer')
#             rating = data.get('rating')
#             session_id = data.get('session_id')
#
#             if not all([question, answer, rating]):
#                 return jsonify({"error": "Missing required fields"}), 400
#
#             response = chat_api.add_feedback(question, answer, rating, session_id)
#             return jsonify(response)
#
#         except Exception as e:
#             return jsonify({"error": str(e)}), 500
#
#     @app.route('/api/search', methods=['POST'])
#     def advanced_search():
#         try:
#             data = request.json
#             query = data.get('query')
#             filters = data.get('filters', {})
#
#             if not query:
#                 return jsonify({"error": "No query provided"}), 400
#
#             response = chat_api.search_with_filters(query, filters)
#             return jsonify(response)
#
#         except Exception as e:
#             return jsonify({"error": str(e)}), 500
#
#     @app.route('/api/analytics', methods=['GET'])
#     def analytics():
#         try:
#             response = chat_api.get_analytics()
#             return jsonify(response)
#         except Exception as e:
#             return jsonify({"error": str(e)}), 500
#
#     @app.route('/api/stats', methods=['GET'])
#     def stats():
#         try:
#             if not chat_api.processor:
#                 return jsonify({"error": "Chat API not initialized"})
#
#             basic_stats = {
#                 "total_chunks": len(chat_api.processor.text_chunks),
#                 "chat_history_length": len(chat_api.processor.chat_history),
#                 "vector_store_initialized": chat_api.processor.vector_store is not None,
#                 "qa_chain_initialized": chat_api.processor.qa_chain is not None
#             }
#             basic_stats.update(chat_api.processor.performance_metrics)
#             return jsonify(basic_stats)
#
#         except Exception as e:
#             return jsonify({"error": str(e)}), 500
#
#     @app.route('/api/upload', methods=['POST'])
#     def upload():
#         try:
#             if 'file' not in request.files:
#                 return jsonify({"error": "No file part in request"}), 400
#
#             file = request.files['file']
#
#             if file.filename == '':
#                 return jsonify({"error": "No selected file"}), 400
#
#             # Preserve original extension
#             filename = secure_filename(file.filename)
#             ext = os.path.splitext(filename)[1]  # e.g., .pdf, .docx
#             with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
#                 file.save(tmp.name)
#                 tmp_path = tmp.name
#                 logger.info(f"Saved uploaded file as: {tmp_path}")
#
#             success = chat_api.processor.process_uploaded_file(tmp_path)
#
#             os.unlink(tmp_path)  # Clean up temp file
#
#             if success:
#                 return jsonify({"status": "File processed and added to vector store"})
#             else:
#                 return jsonify({"error": "Failed to process file"}), 500
#
#         except Exception as e:
#             logger.error(f"Upload error: {str(e)}")
#             return jsonify({"error": str(e)}), 500


# routes.py

from flask import request, jsonify, render_template
from services.chat_api import chat_api
import os
import tempfile
import logging
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)


def register_routes(app):
    @app.route('/')
    def home():
        # Pass frontend keys securely
        return render_template('index.html',
                               supabase_url=os.environ.get("SUPABASE_URL"),
                               supabase_key=os.environ.get("SUPABASE_ANON_KEY")
                               )

    @app.route('/api/query', methods=['POST'])
    def query():
        try:
            data = request.json
            question = data.get('question')
            session_id = data.get('session_id', 'default')

            # --- NEW: Get the chat_id from frontend ---
            chat_id = data.get('chat_id')

            if not question:
                return jsonify({"error": "No question provided"}), 400

            # Pass chat_id to the service layer so it can save to DB
            response = chat_api.get_intelligent_response(question, session_id, chat_id)
            return jsonify(response)

        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/feedback', methods=['POST'])
    def feedback():
        try:
            data = request.json
            question = data.get('question')
            answer = data.get('answer')
            rating = data.get('rating')
            session_id = data.get('session_id')

            if not all([question, answer, rating]):
                return jsonify({"error": "Missing required fields"}), 400

            response = chat_api.add_feedback(question, answer, rating, session_id)
            return jsonify(response)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/search', methods=['POST'])
    def advanced_search():
        try:
            data = request.json
            query = data.get('query')
            filters = data.get('filters', {})

            if not query:
                return jsonify({"error": "No query provided"}), 400

            response = chat_api.search_with_filters(query, filters)
            return jsonify(response)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/analytics', methods=['GET'])
    def analytics():
        try:
            response = chat_api.get_analytics()
            return jsonify(response)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/stats', methods=['GET'])
    def stats():
        try:
            if not chat_api.processor:
                return jsonify({"error": "Chat API not initialized"})

            basic_stats = {
                "total_chunks": len(chat_api.processor.text_chunks),
                "chat_history_length": len(chat_api.processor.chat_history),
                "vector_store_initialized": chat_api.processor.vector_store is not None,
                "qa_chain_initialized": chat_api.processor.qa_chain is not None
            }
            basic_stats.update(chat_api.processor.performance_metrics)
            return jsonify(basic_stats)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/upload', methods=['POST'])
    def upload():
        try:
            if 'file' not in request.files:
                return jsonify({"error": "No file part in request"}), 400

            file = request.files['file']

            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400

            # Preserve original extension
            filename = secure_filename(file.filename)
            ext = os.path.splitext(filename)[1]  # e.g., .pdf, .docx
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name
                logger.info(f"Saved uploaded file as: {tmp_path}")

            success = chat_api.processor.process_uploaded_file(tmp_path)

            os.unlink(tmp_path)  # Clean up temp file

            if success:
                return jsonify({"status": "File processed and added to vector store"})
            else:
                return jsonify({"error": "Failed to process file"}), 500

        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            return jsonify({"error": str(e)}), 500