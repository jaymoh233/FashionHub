
# services/document_processor.py

import os
import logging
from datetime import datetime
from functools import lru_cache

# Loaders
from PyPDF2 import PdfReader
from docx import Document
from bs4 import BeautifulSoup
from pptx import Presentation
import pandas as pd
import json

# LangChain & Supabase
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from supabase import create_client, Client

# We don't need RetrievalQA anymore as we will do it manually
from langchain.schema import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


class AdvancedDocumentProcessor:
    def __init__(self, directory):
        self.directory = directory
        self.vector_store = None
        # We keep the LLM here to reuse it
        self.llm = ChatOpenAI(temperature=0.3, model="gpt-4-turbo-preview", max_tokens=500)

        self.performance_metrics = {
            "total_queries": 0,
            "average_response_time": 0,
            "user_satisfaction": 0,
            "cache_hits": 0
        }

        self.supabase_url = os.environ.get("SUPABASE_URL")
        self.supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

        if self.supabase_url and self.supabase_key:
            self.supabase_client: Client = create_client(self.supabase_url, self.supabase_key)
        else:
            self.supabase_client = None
            logger.error("❌ Supabase Credentials Missing!")

        self.embeddings = OpenAIEmbeddings()

    def create_or_load_vector_store(self):
        """Connects to the existing Supabase Vector Store."""
        try:
            if not self.supabase_client: return False
            logger.info("Connecting to Supabase Vector Store...")

            # We still initialize this for the Upload logic (which works fine)
            self.vector_store = SupabaseVectorStore(
                client=self.supabase_client,
                embedding=self.embeddings,
                table_name="documents",
                query_name="match_documents"
            )
            logger.info("✅ Connected to Supabase Vector Store.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            return False

    # --- FIX: MANUAL SEARCH FUNCTION (Bypasses LangChain Bug) ---
    def manual_similarity_search(self, query, k=5):
        """
        Manually embeds the query and calls the Supabase RPC function.
        This avoids the 'SyncRPCFilterRequestBuilder' error.
        """
        try:
            # 1. Generate Embedding for the question
            query_embedding = self.embeddings.embed_query(query)

            # 2. Call the Database RPC directly
            params = {
                "query_embedding": query_embedding,
                "match_threshold": 0.5,  # Lower this if results are missing
                "match_count": k
            }

            response = self.supabase_client.rpc("match_documents", params).execute()

            # 3. Return pure data
            return response.data  # List of dicts: [{'content': '...', 'metadata': {...}}, ...]

        except Exception as e:
            logger.error(f"Manual search failed: {e}")
            return []

    # --- Standard Upload Logic (Unchanged) ---
    def sync_all_documents(self):
        if not os.path.exists(self.directory): return

        logger.info(f"🔄 Starting Robust Sync for directory: {self.directory}")
        local_files = set([f for f in os.listdir(self.directory) if not f.startswith('.')])

        try:
            response = self.supabase_client.table("documents").select("metadata").execute()
            db_files = set()
            if response.data:
                for row in response.data:
                    meta = row.get('metadata', {})
                    if meta and 'source' in meta:
                        db_files.add(meta['source'])
        except Exception as e:
            logger.error(f"Failed to fetch DB files: {e}")
            return

        files_to_upload = local_files - db_files
        files_to_delete = db_files - local_files

        if files_to_delete:
            for filename in files_to_delete:
                try:
                    self.supabase_client.table("documents").delete().eq("metadata->>source", filename).execute()
                    logger.info(f"❌ Deleted from DB: {filename}")
                except Exception as e:
                    logger.error(f"Error deleting {filename}: {e}")

        if files_to_upload:
            logger.info(f"📤 Uploading {len(files_to_upload)} new files...")
            for filename in files_to_upload:
                file_path = os.path.join(self.directory, filename)
                self.process_uploaded_file(file_path)

        logger.info("✅ Sync Complete.")

    def process_uploaded_file(self, file_path):
        if not os.path.exists(file_path): return False
        try:
            documents = self._read_single_file(file_path)
            if not documents: return False

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )

            chunks = []
            for doc in documents:
                split_docs = text_splitter.create_documents([doc["content"]], metadatas=[doc["metadata"]])
                for i, split_doc in enumerate(split_docs):
                    meta = split_doc.metadata.copy()
                    meta.update({"chunk_index": i})
                    chunks.append(split_doc)

            if chunks and self.vector_store:
                self.vector_store.add_documents(chunks)
                logger.info(f"✅ Indexed {os.path.basename(file_path)}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return False

    def _read_single_file(self, file_path):
        filename = os.path.basename(file_path)
        ext = filename.lower().split('.')[-1]
        documents = []
        try:
            metadata = {
                "source": filename,
                "file_size": os.path.getsize(file_path),
                "upload_date": datetime.now().isoformat()
            }
            content = ""
            if ext == 'pdf':
                with open(file_path, 'rb') as file:
                    pdf_reader = PdfReader(file)
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text: content += text + "\n"
            elif ext in ['txt', 'md', 'csv', 'json']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif ext == 'docx':
                doc = Document(file_path)
                content = '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])

            if content.strip():
                documents.append({"content": content, "metadata": metadata})
        except Exception as e:
            logger.error(f"Error reading {ext}: {e}")
        return documents

    # --- REMOVED: initialize_advanced_conversation ---
    # We no longer use the fragile RetrievalQA chain.
    # Instead, we define the prompt in chat_api.py manually.