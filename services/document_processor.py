# services/document_processor.py

import os
import json
import time
import hashlib
import logging
from datetime import datetime
from functools import lru_cache

import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from bs4 import BeautifulSoup
from pptx import Presentation

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class AdvancedDocumentProcessor:
    def __init__(self, directory):
        self.directory = directory
        self.text_chunks = []
        self.vector_store = None
        self.qa_chain = None
        self.chat_history = []
        self.cache = {}
        self.embeddings_cache_file = "embeddings_cache.pkl"
        self.feedback_data = []
        self.performance_metrics = {
            "total_queries": 0,
            "average_response_time": 0,
            "user_satisfaction": 0,
            "cache_hits": 0
        }

    def read_documents_with_metadata(self):
        documents = []
        if not os.path.exists(self.directory):
            logger.error(f"Directory {self.directory} does not exist")
            return documents

        for filename in os.listdir(self.directory):
            file_path = os.path.join(self.directory, filename)
            ext = filename.lower().split('.')[-1]

            try:
                metadata = {
                    "source": filename,
                    "file_size": os.path.getsize(file_path),
                    "creation_date": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                }

                if ext == 'pdf':
                    with open(file_path, 'rb') as file:
                        pdf_reader = PdfReader(file)
                        for page_num, page in enumerate(pdf_reader.pages):
                            page_text = page.extract_text()
                            if page_text and page_text.strip():
                                page_metadata = metadata.copy()
                                page_metadata.update({
                                    "page": page_num + 1,
                                    "word_count": len(page_text.split())
                                })
                                documents.append({
                                    "content": page_text,
                                    "metadata": page_metadata
                                })
                elif ext in ['txt', 'csv']:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():
                            documents.append({"content": content, "metadata": metadata})
                elif ext in ['xlsx', 'xls']:
                    df = pd.read_excel(file_path)
                    content = df.to_string(index=False)
                    documents.append({"content": content, "metadata": metadata})
                elif ext == 'docx':
                    doc = Document(file_path)
                    content = '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
                    if content:
                        documents.append({"content": content, "metadata": metadata})
                elif ext == 'html':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        soup = BeautifulSoup(f, 'lxml')
                        text = soup.get_text(separator='\n')
                        if text.strip():
                            documents.append({"content": text, "metadata": metadata})
                elif ext == 'json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        content = json.dumps(data, indent=2)
                        documents.append({"content": content, "metadata": metadata})
                elif ext == 'pptx':
                    prs = Presentation(file_path)
                    content = ""
                    for slide in prs.slides:
                        for shape in slide.shapes:
                            if hasattr(shape, "text"):
                                content += shape.text + "\n"
                    if content.strip():
                        documents.append({"content": content, "metadata": metadata})
                else:
                    logger.warning(f"Unsupported file format: {filename}")

            except Exception as e:
                logger.error(f"Error reading {filename}: {str(e)}")

        logger.info(f"Loaded {len(documents)} documents from {self.directory}")
        return documents

    def smart_chunk_text(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]
        )

        chunks = []
        for doc in documents:
            text_chunks = text_splitter.split_text(doc["content"])
            for i, chunk in enumerate(text_chunks):
                if chunk.strip():
                    chunk_metadata = doc["metadata"].copy()
                    chunk_metadata.update({
                        "chunk_id": len(chunks),
                        "chunk_index": i,
                        "chunk_length": len(chunk)
                    })
                    chunks.append({"content": chunk, "metadata": chunk_metadata})

        self.text_chunks = chunks
        logger.info(f"Created {len(self.text_chunks)} intelligent text chunks")

    def create_or_load_vector_store(self):
        cache_file = "vector_store_cache"
        if os.path.exists(f"{cache_file}.faiss") and os.path.exists(f"{cache_file}.pkl"):
            try:
                cache_time = os.path.getmtime(f"{cache_file}.faiss")
                if time.time() - cache_time < 7 * 24 * 3600:
                    logger.info("Loading cached vector store...")
                    embeddings = OpenAIEmbeddings()
                    self.vector_store = FAISS.load_local(cache_file, embeddings, allow_dangerous_deserialization=True)
                    return
            except Exception as e:
                logger.warning(f"Failed to load cached vector store: {e}")

        if not self.text_chunks:
            logger.error("No text chunks available to create vector store")
            return

        logger.info("Creating new vector store...")
        texts = [chunk["content"] for chunk in self.text_chunks]
        metadatas = [chunk["metadata"] for chunk in self.text_chunks]

        embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)

        try:
            self.vector_store.save_local(cache_file)
            logger.info("Vector store cached successfully")
        except Exception as e:
            logger.warning(f"Failed to cache vector store: {e}")

    def initialize_advanced_conversation(self):
        if not self.vector_store:
            logger.error("Vector store not available")
            return

        llm = ChatOpenAI(
            temperature=0.3,
            model="gpt-4-turbo-preview",
            max_tokens=500
        )

        prompt_template = """You are an intelligent AI assistant developed by the VRA-DTI (Volta River Authority - Digital Transformation and Innovation Center) Team with deep knowledge of the company's data. You're conversational, helpful, and provide accurate information naturally.

Context Information:
{context}

Previous Conversation Context: Remember our conversation flow and refer back to previous topics when relevant.

User Question: {question}

Response Guidelines:
- Be conversational and natural - avoid robotic language
- Provide specific, actionable information when available
- If you don't have exact information, suggest related topics VRA have covered
- Use examples and analogies to make complex topics clearer
- Ask follow-up questions when appropriate to better help the user
- Remember: You're not just answering - you're having a helpful conversation

Examples of good responses:
User: "What does VRA Say about project management?"
Good: "VRA  shares some great insights on project management! it emphasizes the importance of clear communication and setting realistic deadlines. One thing it mentions is..."

User: "How should I handle difficult clients?"
Good: "That's a common challenge! VRA actually covers client management strategies. It suggests starting with understanding their perspective..."

Your response:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        memory = ConversationBufferWindowMemory(
            k=5,
            return_messages=True
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.7}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

    @lru_cache(maxsize=100)
    def get_cached_response(self, question_hash):
        return self.cache.get(question_hash)

    def add_to_cache(self, question_hash, response):
        self.cache[question_hash] = {
            "response": response,
            "timestamp": time.time()
        }
        current_time = time.time()
        self.cache = {k: v for k, v in self.cache.items() if current_time - v["timestamp"] < 3600}

    def process_uploaded_file(self, file_path):
        """Process an uploaded file: read, chunk, embed, and update vector store."""
        if not os.path.exists(file_path):
            logger.error(f"Uploaded file {file_path} does not exist")
            return False

        ext = file_path.lower().split('.')[-1]
        documents = []

        try:
            metadata = {
                "source": os.path.basename(file_path),
                "file_size": os.path.getsize(file_path),
                "creation_date": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
            }

            if ext == 'pdf':
                with open(file_path, 'rb') as file:
                    pdf_reader = PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            page_metadata = metadata.copy()
                            page_metadata.update({
                                "page": page_num + 1,
                                "word_count": len(page_text.split())
                            })
                            documents.append({"content": page_text, "metadata": page_metadata})

            elif ext in ['txt', 'csv']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        documents.append({"content": content, "metadata": metadata})

            elif ext in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
                content = df.to_string(index=False)
                documents.append({"content": content, "metadata": metadata})

            elif ext == 'docx':
                doc = Document(file_path)
                content = '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
                if content:
                    documents.append({"content": content, "metadata": metadata})

            elif ext == 'html':
                with open(file_path, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f, 'lxml')
                    text = soup.get_text(separator='\n')
                    if text.strip():
                        documents.append({"content": text, "metadata": metadata})

            elif ext == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    content = json.dumps(data, indent=2)
                    documents.append({"content": content, "metadata": metadata})

            elif ext == 'pptx':
                prs = Presentation(file_path)
                content = ""
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            content += shape.text + "\n"
                if content.strip():
                    documents.append({"content": content, "metadata": metadata})

            else:
                logger.warning(f"Unsupported file format: {file_path}")
                return False

            if not documents:
                return False

            # Step 2: Chunk
            self.smart_chunk_text(documents)

            # Step 3: Add to vector store
            self.add_new_chunks_to_vector_store(self.text_chunks)
            logger.info(f"Uploaded file {file_path} processed and added to vector store")
            return True

        except Exception as e:
            logger.error(f"Error processing uploaded file {file_path}: {str(e)}")
            return False

    def add_new_chunks_to_vector_store(self, new_chunks):
        texts = [chunk["content"] for chunk in new_chunks]
        metadatas = [chunk["metadata"] for chunk in new_chunks]
        embeddings = OpenAIEmbeddings()

        if not self.vector_store:
            self.vector_store = FAISS.from_texts(texts, embeddings, metadatas)
        else:
            self.vector_store.add_texts(texts, metadatas)

        logger.info(f"Added {len(new_chunks)} new chunks to vector store")
