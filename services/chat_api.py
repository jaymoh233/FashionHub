
# services/chat_api.py

import time
import hashlib
import logging
import os
from datetime import datetime
from supabase import create_client, Client
from services.document_processor import AdvancedDocumentProcessor
from langchain.schema import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


class EnhancedChatAPI:
    def __init__(self):
        self.processor = None
        # Metric storage (In-memory for simplicity, resets on restart)
        self.feedback_data = []

        self.supabase_url = os.environ.get("SUPABASE_URL")
        self.supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
        self.db: Client = None

        if self.supabase_url and self.supabase_key:
            try:
                self.db = create_client(self.supabase_url, self.supabase_key)
                logger.info("Supabase client initialized.")
            except Exception as e:
                logger.error(f"Failed to init Supabase: {e}")
        else:
            logger.warning("Supabase credentials missing.")

    def setup(self, document_directory):
        logger.info("Starting setup...")
        self.processor = AdvancedDocumentProcessor(document_directory)

        # 1. Connect to DB
        success = self.processor.create_or_load_vector_store()
        if not success: return False

        # 2. Sync Files
        self.processor.sync_all_documents()

        logger.info("Setup complete!")
        return True

    def save_bot_response(self, chat_id, answer):
        """Updates the specific chat UI row with the answer."""
        if not self.db or not chat_id: return
        try:
            current_time = datetime.now().isoformat()
            self.db.table("chats").update({
                "bot_response": answer,
                "bot_timestamp": current_time
            }).eq("id", chat_id).execute()
            logger.info(f"Saved response for chat_id: {chat_id}")
        except Exception as e:
            logger.error(f"Failed to save to Supabase: {e}")

    # --- DB CACHE METHODS ---
    def _check_db_cache(self, question_hash):
        if not self.db: return None
        try:
            response = self.db.table("query_cache").select("answer").eq("question_hash", question_hash).limit(
                1).execute()
            if response.data: return response.data[0]['answer']
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")
        return None

    def _save_to_db_cache(self, question_hash, question, answer):
        if not self.db: return
        try:
            self.db.table("query_cache").insert({
                "question_hash": question_hash,
                "question": question,
                "answer": answer
            }).execute()
        except Exception:
            pass  # Ignore duplicates

    # --- MAIN GENERATION LOGIC ---
    def get_intelligent_response(self, question, session_id=None, chat_id=None):
        start_time = time.time()

        if not self.processor or not self.processor.llm:
            return {"error": "System not initialized"}

        # --- UPDATED: CACHE ELIGIBILITY CHECK ---
        # We now cache EVERYTHING except empty strings.
        # We rely on the System Prompt to keep generic answers "safe" (neutral) for reuse.
        cleaned_q = question.strip().lower()
        is_cacheable = len(cleaned_q) > 0  # Cache everything valid

        try:
            # 1. Cache Check (Only if eligible)
            question_hash = hashlib.md5(cleaned_q.encode()).hexdigest()

            if is_cacheable:
                cached_answer = self._check_db_cache(question_hash)
                if cached_answer:
                    if chat_id: self.save_bot_response(chat_id, cached_answer)
                    return {
                        "answer": cached_answer,
                        "cached": True,
                        "sources": [],
                        "suggestions": ["Tell me more", "How do I style this?"]
                    }

            # 2. Manual Search
            raw_docs = self.processor.manual_similarity_search(question, k=5)

            context_text = ""
            if raw_docs:
                for doc in raw_docs:
                    content = doc.get('content', '')
                    context_text += f"\n---\n{content}\n"
            else:
                context_text = "No specific documents found."

            # 3. Add History
            history_text = self._get_chat_history(session_id)

            # 4. Construct Prompt (Standardized & Clean)
            system_prompt = """
            ### ROLE & IDENTITY
            You are "Aura", the expert AI fashion stylist for FashionHub, developed by Mohammed Hussein.
            Your goal is to help users explore outfits, discover trends, and feel confident in their style.


            ### TONE & VOICE
            - **Friendly & Confident:** Speak like a knowledgeable personal stylist, not a robot.
            - **Body-Inclusive:** Always be positive, encouraging, and supportive of all body types.
            - **Stylish Vocabulary:** Use terms like "silhouette," "palette," "texture," "statement piece," and "timeless."
            - **Emojis:** Use relevant emojis sparingly to add warmth (e.g., 👗, ✨, 👠), but do not overuse them.


            ### SAFETY OVERRIDE FOR GENERIC INPUTS (CRITICAL)
            If the user's input is a generic greeting (e.g., "Hi", "Hello") or closing (e.g., "Thanks", "Bye", "Ok"):
            1. **IGNORE** the Context Information and Conversation History.
            2. **RESPOND NEUTRALLY:** Give a response that would make sense to *anyone*, regardless of what they previously talked about.
                - *Bad (Contextual):* "You're welcome! I hope those **red heels** work out!" (Do NOT do this).
                - *Good (Neutral):* "You're welcome! Let me know if you need more fashion advice." (Do this).
                - *Good (Greeting):* "Hello! I'm Aura. Ready to find your next look?"
            
            ### OPERATIONAL RULES
            1. **No Robotic Fillers:** NEVER say "As an AI." "Based on the context provided," or "According to the documents." Just give the answer naturally.
            2. **Context Priority:** For specific questions, use the Context Information.
            3. **Conciseness:** Keep paragraphs short.


            ### ENGAGEMENT (CRITICAL)
            - **Always end with a Follow-up Question:** Never leave the conversation dead. After answering, ask a relevant question to guide the user to the next step.
            - *Bad:* "Blue jeans look great with white shirts."
            - *Good:* "Blue jeans look great with white shirts! Are you dressing for a casual day out or something more semi-formal?"


            ### FORMATTING GUIDELINES
            - Use **Bold** for key items or tips to make the text scannable.
            - Use Bullet points for lists of recommendations.
            - Keep paragraphs short (2-3 sentences max).
                       """

            # The User Prompt injects the dynamic data
            user_prompt = f"""
                       ### CONTEXT INFORMATION (Knowledge Base)
                       {context_text}


                       ### CONVERSATION HISTORY
                       {history_text}


                       ### USER QUESTION
                       {question}


                       Answer the user's question now, adhering to the persona and guidelines above.
                       """

            # 5. Call OpenAI
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            ai_response = self.processor.llm.invoke(messages)
            answer = ai_response.content

            # 6. Save & Return
            if chat_id: self.save_bot_response(chat_id, answer)

            # Only save to cache if it passed the eligibility check earlier
            if is_cacheable:
                self._save_to_db_cache(question_hash, question, answer)

            self._update_performance_metrics(time.time() - start_time)

            return {
                "answer": answer,
                "cached": False,
                "response_time": time.time() - start_time,
                "sources": self._format_sources_intelligently(raw_docs),
                "confidence": self._calculate_confidence(raw_docs, question),
                "suggestions": self._generate_follow_up_suggestions(answer, raw_docs)
            }

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            if chat_id: self.save_bot_response(chat_id, "I encountered a system error.")
            return {"error": str(e)}

    # --- RESTORED HELPER FUNCTIONS ---

    def _get_chat_history(self, session_id):
        """Fetches history from DB to provide context."""
        if not session_id or session_id == 'default' or not self.db:
            return ""
        try:
            response = self.db.table("chats") \
                .select("user_message, bot_response") \
                .eq("conversation_id", session_id) \
                .order("created_at", desc=True) \
                .limit(10) \
                .execute()

            if response.data:
                history = response.data[::-1]
                text = ""
                for chat in history:
                    u = chat.get('user_message')
                    b = chat.get('bot_response')
                    if u and b: text += f"User: {u}\nAura: {b}\n"
                return text
        except Exception:
            pass
        return ""

    def _format_sources_intelligently(self, raw_docs):
        """Restored Source Formatting for UI."""
        sources = []
        for i, doc in enumerate(raw_docs):
            content = doc.get('content', '')
            meta = doc.get('metadata', {}) or {}

            # Since manual search doesn't always return score in the same way, we estimate
            relevance_score = max(0.1, 1.0 - (i * 0.1))

            source_info = {
                "content_preview": content[:200] + "..." if len(content) > 200 else content,
                "metadata": meta,
                "relevance_score": round(relevance_score, 2),
                "source": meta.get("source", "Unknown"),
                "page": meta.get("page", "N/A"),
                "word_count": meta.get("word_count", 0)
            }
            sources.append(source_info)
        return sources

    def _calculate_confidence(self, raw_docs, question):
        """Restored Confidence Calculation."""
        if not raw_docs:
            return 0.1
        confidence = min(len(raw_docs) * 0.15, 0.9)
        question_words = set(question.lower().split())
        for doc in raw_docs:
            content = doc.get('content', '').lower()
            doc_words = set(content.split())
            overlap = len(question_words.intersection(doc_words))
            confidence += overlap * 0.05
        return min(confidence, 1.0)

    def _generate_follow_up_suggestions(self, answer, raw_docs):
        """Restored Dynamic Suggestions."""
        suggestions = []
        topics = []

        # Try to find interesting topics in the source text
        for doc in raw_docs[:3]:
            content = doc.get('content', '')
            content_words = content.split()[:50]
            # Find capitalized words (Proper Nouns) as topics
            potential_topics = [word for word in content_words if
                                len(word) > 5 and word.isalpha() and word[0].isupper()]
            topics.extend(potential_topics[:2])

        if not topics:
            topics = ["Trends", "Styles", "Fabrics"]

        if topics:
            suggestions.append(f"Tell me more about {topics[0]}")
            if len(topics) > 1:
                suggestions.append(f"How do I style {topics[1]}?")

        suggestions.append("Can you give me more examples?")
        return suggestions[:3]

    def _update_performance_metrics(self, response_time):
        if self.processor:
            self.processor.performance_metrics["total_queries"] += 1
            # Simple average update
            current_avg = self.processor.performance_metrics["average_response_time"]
            count = self.processor.performance_metrics["total_queries"]
            self.processor.performance_metrics["average_response_time"] = ((current_avg * (
                        count - 1)) + response_time) / count

    def add_feedback(self, question, answer, rating, session_id=None):
        """Restored Feedback Logging."""
        feedback = {
            "timestamp": time.time(),
            "question": question,
            "answer": answer,
            "rating": rating,
            "session_id": session_id
        }
        self.feedback_data.append(feedback)

        # Update satisfaction score in processor metrics
        if self.processor:
            ratings = [f["rating"] for f in self.feedback_data]
            if ratings:
                self.processor.performance_metrics["user_satisfaction"] = sum(ratings) / len(ratings)

        return {"status": "feedback_recorded"}

    def get_analytics(self):
        """Restored Analytics."""
        if not self.processor: return {"status": "not_initialized"}

        # Merge processor metrics with chat_api metrics
        metrics = self.processor.performance_metrics.copy()
        metrics["feedback_count"] = len(self.feedback_data)
        return metrics

    def search_with_filters(self, query, filters=None):
        """Restored Search Endpoint using Manual Search."""
        if not self.processor: return {"results": []}

        # Use manual search (safe)
        raw_docs = self.processor.manual_similarity_search(query, k=10)

        filtered_results = []
        for doc in raw_docs:
            metadata = doc.get('metadata', {}) or {}
            content = doc.get('content', '')

            # Manual Filter Logic
            if filters:
                if filters.get("source") and metadata.get("source") != filters["source"]:
                    continue
                if filters.get("page_range"):
                    page = metadata.get("page", 0)
                    if not (filters["page_range"][0] <= page <= filters["page_range"][1]):
                        continue

            filtered_results.append({
                "content": content[:300] + "..." if len(content) > 300 else content,
                "metadata": metadata,
                "relevance_score": doc.get('similarity', 0.0)
            })

        return {
            "query": query,
            "total_results": len(filtered_results),
            "results": filtered_results
        }


# Exportable singleton
chat_api = EnhancedChatAPI()