#
# # services/chat_api.py
#
# import time
# import hashlib
# import logging
# import os
# from datetime import datetime
# # NEW: Import Supabase
# from supabase import create_client, Client
#
# from services.document_processor import AdvancedDocumentProcessor
#
# logger = logging.getLogger(__name__)
#
#
# class EnhancedChatAPI:
#     def __init__(self):
#         self.processor = None
#         self.session_data = {}
#
#         # --- NEW: Initialize Supabase Client ---
#         # We use the SERVICE_KEY to bypass RLS permissions on the backend
#         # Ensure these are in your .env file!
#         self.supabase_url = os.environ.get("SUPABASE_URL")
#         self.supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
#         self.db: Client = None
#
#         if self.supabase_url and self.supabase_key:
#             try:
#                 self.db = create_client(self.supabase_url, self.supabase_key)
#                 logger.info("Supabase client initialized successfully.")
#             except Exception as e:
#                 logger.error(f"Failed to init Supabase: {e}")
#         else:
#             logger.warning("Supabase credentials missing in environment variables.")
#
#     def setup(self, document_directory):
#         logger.info("Starting enhanced document processing...")
#         self.processor = AdvancedDocumentProcessor(document_directory)
#
#         documents = self.processor.read_documents_with_metadata()
#         if not documents:
#             logger.error("No documents found")
#             return False
#
#         self.processor.smart_chunk_text(documents)
#         self.processor.create_or_load_vector_store()
#         self.processor.initialize_advanced_conversation()
#
#         logger.info("Enhanced setup complete!")
#         return True
#
#     # --- NEW: Helper function to save to DB ---
#     def save_bot_response(self, chat_id, answer):
#         if not self.db or not chat_id:
#             return
#
#         try:
#             current_time = datetime.now().isoformat()
#
#             # Update the row that the frontend created
#             data = self.db.table("chats").update({
#                 "bot_response": answer,
#                 "bot_timestamp": current_time
#             }).eq("id", chat_id).execute()
#
#             logger.info(f"Successfully saved response to chat_id: {chat_id}")
#         except Exception as e:
#             logger.error(f"Failed to save to Supabase: {e}")
#
#     # --- UPDATED: Accepts chat_id ---
#     def get_intelligent_response(self, question, session_id=None, chat_id=None):
#         start_time = time.time()
#
#         if not self.processor or not self.processor.qa_chain:
#             return {"error": "System not initialized"}
#
#         try:
#             question_hash = hashlib.md5(question.encode()).hexdigest()
#
#             # Check Cache
#             cached = self.processor.get_cached_response(question_hash)
#             if cached:
#                 self.processor.performance_metrics["cache_hits"] += 1
#                 response_time = time.time() - start_time
#                 answer = cached["response"]["answer"]
#
#                 # --- NEW: Save Cached Response to DB ---
#                 if chat_id:
#                     self.save_bot_response(chat_id, answer)
#
#                 return {
#                     "answer": answer,
#                     "cached": True,
#                     "response_time": response_time,
#                     "sources": cached["response"].get("sources", [])
#                 }
#
#             # Generate New Response
#             enhanced_question = self._enhance_with_session_context(question, session_id)
#             response = self.processor.qa_chain.invoke({"query": enhanced_question})
#             answer = response["result"]
#             source_docs = response.get("source_documents", [])
#
#             answer = self._make_response_intelligent(answer, question)
#             response_time = time.time() - start_time
#
#             self._update_performance_metrics(response_time)
#
#             # --- NEW: Save Generated Response to DB ---
#             if chat_id:
#                 self.save_bot_response(chat_id, answer)
#
#             result = {
#                 "answer": answer,
#                 "cached": False,
#                 "response_time": response_time,
#                 "sources": self._format_sources_intelligently(source_docs),
#                 "confidence": self._calculate_confidence(source_docs, question),
#                 "suggestions": self._generate_follow_up_suggestions(answer, source_docs)
#             }
#
#             self.processor.add_to_cache(question_hash, result)
#
#             if session_id:
#                 self._update_session_context(session_id, question, answer)
#
#             return result
#
#         except Exception as e:
#             logger.error(f"Error processing question: {str(e)}")
#
#             # --- NEW: Save Error to DB ---
#             # If generation fails, tell the user via the DB so they aren't stuck "Thinking"
#             if chat_id:
#                 self.save_bot_response(chat_id, "I encountered a server error processing your request.")
#
#             return {"error": f"Processing failed: {str(e)}"}
#
#     def _enhance_with_session_context(self, question, session_id):
#         if not session_id or session_id not in self.session_data:
#             return question
#         session = self.session_data[session_id]
#         recent_context = session.get("recent_topics", [])
#         if recent_context:
#             context_str = " | ".join(recent_context[-3:])
#             return f"Recent topics discussed: {context_str}\n\nCurrent question: {question}"
#         return question
#
#     def _make_response_intelligent(self, answer, question):
#         robotic_phrases = [
#             "Based on the context provided,", "According to the context,",
#             "From the information provided,", "Based on the available information,",
#             "According to the documents,", "The context indicates that",
#             "Based on the context,", "From the context,"
#         ]
#         for phrase in robotic_phrases:
#             if answer.strip().startswith(phrase):
#                 answer = answer[len(phrase):].strip()
#                 if answer:
#                     answer = answer[0].upper() + answer[1:]
#                 break
#
#         if "?" in question and "how" in question.lower():
#             if not answer.startswith(("Here's how", "You can", "To ")):
#                 answer = answer
#         elif "what" in question.lower():
#             if not answer.startswith(("What", "This", "It")):
#                 answer = answer
#         return answer
#
#     def _calculate_confidence(self, source_docs, question):
#         if not source_docs:
#             return 0.1
#         confidence = min(len(source_docs) * 0.15, 0.9)
#         question_words = set(question.lower().split())
#         for doc in source_docs:
#             doc_words = set(doc.page_content.lower().split())
#             overlap = len(question_words.intersection(doc_words))
#             confidence += overlap * 0.05
#         return min(confidence, 1.0)
#
#     def _generate_follow_up_suggestions(self, answer, source_docs):
#         suggestions = []
#         topics = []
#         for doc in source_docs[:3]:
#             content_words = doc.page_content.split()[:50]
#             potential_topics = [word for word in content_words if len(word) > 5 and word.isalpha()]
#             topics.extend(potential_topics[:2])
#         if topics:
#             suggestions.append(f"Would you like to know more about {topics[0]}?")
#             if len(topics) > 1:
#                 suggestions.append(f"How does this relate to {topics[1]}?")
#         suggestions.append("Is there a specific aspect you'd like me to elaborate on?")
#         return suggestions[:3]
#
#     def _format_sources_intelligently(self, source_docs):
#         sources = []
#         for i, doc in enumerate(source_docs):
#             relevance_score = max(0.1, 1.0 - (i * 0.1))
#             source_info = {
#                 "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
#                 "metadata": doc.metadata,
#                 "relevance_score": round(relevance_score, 2),
#                 "source": doc.metadata.get("source", "Unknown"),
#                 "page": doc.metadata.get("page", "Unknown"),
#                 "word_count": doc.metadata.get("word_count", 0)
#             }
#             sources.append(source_info)
#         return sources
#
#     def _update_session_context(self, session_id, question, answer):
#         if session_id not in self.session_data:
#             self.session_data[session_id] = {
#                 "start_time": time.time(),
#                 "recent_topics": [],
#                 "question_count": 0
#             }
#         session = self.session_data[session_id]
#         session["question_count"] += 1
#         combined_text = f"{question} {answer}"
#         words = combined_text.split()
#         topics = [word for word in words if len(word) > 6 and word.isalpha()]
#         if topics:
#             session["recent_topics"].extend(topics[:3])
#             session["recent_topics"] = session["recent_topics"][-10:]
#
#     def _update_performance_metrics(self, response_time):
#         metrics = self.processor.performance_metrics
#         metrics["total_queries"] += 1
#         current_avg = metrics["average_response_time"]
#         total_queries = metrics["total_queries"]
#         metrics["average_response_time"] = ((current_avg * (total_queries - 1)) + response_time) / total_queries
#
#     def add_feedback(self, question, answer, rating, session_id=None):
#         feedback = {
#             "timestamp": time.time(),
#             "question": question,
#             "answer": answer,
#             "rating": rating,
#             "session_id": session_id
#         }
#         self.processor.feedback_data.append(feedback)
#         ratings = [f["rating"] for f in self.processor.feedback_data]
#         self.processor.performance_metrics["user_satisfaction"] = sum(ratings) / len(ratings)
#         return {"status": "feedback_recorded"}
#
#     def get_analytics(self):
#         if not self.processor:
#             return {"error": "System not initialized"}
#         return {
#             "performance_metrics": self.processor.performance_metrics,
#             "cache_size": len(self.processor.cache),
#             "total_chunks": len(self.processor.text_chunks),
#             "active_sessions": len(self.session_data),
#             "system_uptime": time.time() - getattr(self, 'start_time', time.time()),
#             "recent_feedback_count": len(self.processor.feedback_data),
#             "average_confidence": 0.75
#         }
#
#     def search_with_filters(self, query, filters=None):
#         if not self.processor or not self.processor.vector_store:
#             return {"error": "System not initialized"}
#         try:
#             docs = self.processor.vector_store.similarity_search(query, k=10)
#             if filters:
#                 filtered_docs = []
#                 for doc in docs:
#                     metadata = doc.metadata
#                     if filters.get("source") and metadata.get("source") != filters["source"]:
#                         continue
#                     if filters.get("page_range"):
#                         page = metadata.get("page", 0)
#                         if not (filters["page_range"][0] <= page <= filters["page_range"][1]):
#                             continue
#                     filtered_docs.append(doc)
#                 docs = filtered_docs
#             return {
#                 "query": query,
#                 "total_results": len(docs),
#                 "results": [{
#                     "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
#                     "metadata": doc.metadata,
#                     "relevance_score": 1.0 - (i * 0.1)
#                 } for i, doc in enumerate(docs)]
#             }
#         except Exception as e:
#             return {"error": str(e)}
#
#
# # Exportable singleton
# chat_api = EnhancedChatAPI()


# services/chat_api.py

import time
import hashlib
import logging
import os
from datetime import datetime
# NEW: Import Supabase
from supabase import create_client, Client

from services.document_processor import AdvancedDocumentProcessor

logger = logging.getLogger(__name__)


class EnhancedChatAPI:
    def __init__(self):
        self.processor = None
        self.session_data = {}

        # --- NEW: Initialize Supabase Client ---
        # We use the SERVICE_KEY to bypass RLS permissions on the backend
        # Ensure these are in your .env file!
        self.supabase_url = os.environ.get("SUPABASE_URL")
        self.supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
        self.db: Client = None

        if self.supabase_url and self.supabase_key:
            try:
                self.db = create_client(self.supabase_url, self.supabase_key)
                logger.info("Supabase client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to init Supabase: {e}")
        else:
            logger.warning("Supabase credentials missing in environment variables.")

    def setup(self, document_directory):
        logger.info("Starting enhanced document processing...")
        self.processor = AdvancedDocumentProcessor(document_directory)

        documents = self.processor.read_documents_with_metadata()
        if not documents:
            logger.error("No documents found")
            return False

        self.processor.smart_chunk_text(documents)
        self.processor.create_or_load_vector_store()
        self.processor.initialize_advanced_conversation()

        logger.info("Enhanced setup complete!")
        return True

    # --- NEW: Helper function to save to DB ---
    # def save_bot_response(self, chat_id, answer):
    #     if not self.db or not chat_id:
    #         return
    #
    #     try:
    #         current_time = datetime.now().isoformat()
    #
    #         # Update the row that the frontend created
    #         data = self.db.table("chats").update({
    #             "bot_response": answer,
    #             "bot_timestamp": current_time
    #         }).eq("id", chat_id).execute()
    #
    #         logger.info(f"Successfully saved response to chat_id: {chat_id}")
    #     except Exception as e:
    #         logger.error(f"Failed to save to Supabase: {e}")

    def save_bot_response(self, chat_id, answer):
        # --- DEBUG LOGGING ---
        if not self.db:
            logger.error("❌ Save failed: Supabase client (self.db) is NOT initialized. Check SUPABASE_SERVICE_KEY.")
            return
        if not chat_id:
            logger.error("❌ Save failed: No chat_id provided by frontend.")
            return
        # ---------------------

        try:
            current_time = datetime.now().isoformat()

            data = self.db.table("chats").update({
                "bot_response": answer,
                "bot_timestamp": current_time
            }).eq("id", chat_id).execute()

            # logger.info(f"✅ Successfully saved response to chat_id: {chat_id}")
        except Exception as e:
            logger.error(f"❌ Failed to save to Supabase: {e}")

    # --- UPDATED: Accepts chat_id ---
    def get_intelligent_response(self, question, session_id=None, chat_id=None):
        start_time = time.time()

        if not self.processor or not self.processor.qa_chain:
            return {"error": "System not initialized"}

        try:
            question_hash = hashlib.md5(question.encode()).hexdigest()

            # Check Cache
            cached = self.processor.get_cached_response(question_hash)
            if cached:
                self.processor.performance_metrics["cache_hits"] += 1
                response_time = time.time() - start_time
                answer = cached["response"]["answer"]

                # --- NEW: Save Cached Response to DB ---
                if chat_id:
                    self.save_bot_response(chat_id, answer)

                return {
                    "answer": answer,
                    "cached": True,
                    "response_time": response_time,
                    "sources": cached["response"].get("sources", [])
                }

            # Generate New Response
            enhanced_question = self._enhance_with_session_context(question, session_id)
            response = self.processor.qa_chain.invoke({"query": enhanced_question})
            answer = response["result"]
            source_docs = response.get("source_documents", [])

            answer = self._make_response_intelligent(answer, question)
            response_time = time.time() - start_time

            self._update_performance_metrics(response_time)

            # --- NEW: Save Generated Response to DB ---
            if chat_id:
                self.save_bot_response(chat_id, answer)

            result = {
                "answer": answer,
                "cached": False,
                "response_time": response_time,
                "sources": self._format_sources_intelligently(source_docs),
                "confidence": self._calculate_confidence(source_docs, question),
                "suggestions": self._generate_follow_up_suggestions(answer, source_docs)
            }

            self.processor.add_to_cache(question_hash, result)

            if session_id:
                self._update_session_context(session_id, question, answer)

            return result

        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")

            # --- NEW: Save Error to DB ---
            # If generation fails, tell the user via the DB so they aren't stuck "Thinking"
            if chat_id:
                self.save_bot_response(chat_id, "I encountered a server error processing your request.")

            return {"error": f"Processing failed: {str(e)}"}

    def _enhance_with_session_context(self, question, session_id):
        if not session_id or session_id not in self.session_data:
            return question
        session = self.session_data[session_id]
        recent_context = session.get("recent_topics", [])
        if recent_context:
            context_str = " | ".join(recent_context[-3:])
            return f"Recent topics discussed: {context_str}\n\nCurrent question: {question}"
        return question

    def _make_response_intelligent(self, answer, question):
        robotic_phrases = [
            "Based on the context provided,", "According to the context,",
            "From the information provided,", "Based on the available information,",
            "According to the documents,", "The context indicates that",
            "Based on the context,", "From the context,"
        ]
        for phrase in robotic_phrases:
            if answer.strip().startswith(phrase):
                answer = answer[len(phrase):].strip()
                if answer:
                    answer = answer[0].upper() + answer[1:]
                break

        if "?" in question and "how" in question.lower():
            if not answer.startswith(("Here's how", "You can", "To ")):
                answer = answer
        elif "what" in question.lower():
            if not answer.startswith(("What", "This", "It")):
                answer = answer
        return answer

    def _calculate_confidence(self, source_docs, question):
        if not source_docs:
            return 0.1
        confidence = min(len(source_docs) * 0.15, 0.9)
        question_words = set(question.lower().split())
        for doc in source_docs:
            doc_words = set(doc.page_content.lower().split())
            overlap = len(question_words.intersection(doc_words))
            confidence += overlap * 0.05
        return min(confidence, 1.0)

    def _generate_follow_up_suggestions(self, answer, source_docs):
        suggestions = []
        topics = []
        for doc in source_docs[:3]:
            content_words = doc.page_content.split()[:50]
            potential_topics = [word for word in content_words if len(word) > 5 and word.isalpha()]
            topics.extend(potential_topics[:2])
        if topics:
            suggestions.append(f"Would you like to know more about {topics[0]}?")
            if len(topics) > 1:
                suggestions.append(f"How does this relate to {topics[1]}?")
        suggestions.append("Is there a specific aspect you'd like me to elaborate on?")
        return suggestions[:3]

    def _format_sources_intelligently(self, source_docs):
        sources = []
        for i, doc in enumerate(source_docs):
            relevance_score = max(0.1, 1.0 - (i * 0.1))
            source_info = {
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": round(relevance_score, 2),
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "word_count": doc.metadata.get("word_count", 0)
            }
            sources.append(source_info)
        return sources

    def _update_session_context(self, session_id, question, answer):
        if session_id not in self.session_data:
            self.session_data[session_id] = {
                "start_time": time.time(),
                "recent_topics": [],
                "question_count": 0
            }
        session = self.session_data[session_id]
        session["question_count"] += 1
        combined_text = f"{question} {answer}"
        words = combined_text.split()
        topics = [word for word in words if len(word) > 6 and word.isalpha()]
        if topics:
            session["recent_topics"].extend(topics[:3])
            session["recent_topics"] = session["recent_topics"][-10:]

    def _update_performance_metrics(self, response_time):
        metrics = self.processor.performance_metrics
        metrics["total_queries"] += 1
        current_avg = metrics["average_response_time"]
        total_queries = metrics["total_queries"]
        metrics["average_response_time"] = ((current_avg * (total_queries - 1)) + response_time) / total_queries

    def add_feedback(self, question, answer, rating, session_id=None):
        feedback = {
            "timestamp": time.time(),
            "question": question,
            "answer": answer,
            "rating": rating,
            "session_id": session_id
        }
        self.processor.feedback_data.append(feedback)
        ratings = [f["rating"] for f in self.processor.feedback_data]
        self.processor.performance_metrics["user_satisfaction"] = sum(ratings) / len(ratings)
        return {"status": "feedback_recorded"}

    def get_analytics(self):
        if not self.processor:
            return {"error": "System not initialized"}
        return {
            "performance_metrics": self.processor.performance_metrics,
            "cache_size": len(self.processor.cache),
            "total_chunks": len(self.processor.text_chunks),
            "active_sessions": len(self.session_data),
            "system_uptime": time.time() - getattr(self, 'start_time', time.time()),
            "recent_feedback_count": len(self.processor.feedback_data),
            "average_confidence": 0.75
        }

    def search_with_filters(self, query, filters=None):
        if not self.processor or not self.processor.vector_store:
            return {"error": "System not initialized"}
        try:
            docs = self.processor.vector_store.similarity_search(query, k=10)
            if filters:
                filtered_docs = []
                for doc in docs:
                    metadata = doc.metadata
                    if filters.get("source") and metadata.get("source") != filters["source"]:
                        continue
                    if filters.get("page_range"):
                        page = metadata.get("page", 0)
                        if not (filters["page_range"][0] <= page <= filters["page_range"][1]):
                            continue
                    filtered_docs.append(doc)
                docs = filtered_docs
            return {
                "query": query,
                "total_results": len(docs),
                "results": [{
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": 1.0 - (i * 0.1)
                } for i, doc in enumerate(docs)]
            }
        except Exception as e:
            return {"error": str(e)}


# Exportable singleton
chat_api = EnhancedChatAPI()