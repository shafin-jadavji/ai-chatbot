import chromadb
from redis import Redis
import uuid
import json
from datetime import datetime
import logging
from dataclasses import dataclass
from typing import Optional, List
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from os import getenv

@dataclass
class MemoryConfig:
    redis_host: str = getenv('REDIS_HOST', 'localhost') # Redis host
    redis_port: int = int(getenv('REDIS_PORT', '6379')) # Redis port
    redis_password: Optional[str] = None # Redis password (if required)
    redis_ttl: int = int(getenv('REDIS_TTL', '3600'))  # Default 1 hour TTL    
    chroma_collection_name: str = 'memory' # Collection name in ChromaDB
    max_results: int = 100 # Maximum number of results to retrieve from ChromaDB
    enable_telemetry: bool = False # Enable telemetry for ChromaDB
    chroma_persist_directory: Optional[str] = None # Directory to persist ChromaDB data
    chroma_client_settings: Optional[dict] = None # Additional settings for ChromaDB client
    log_level: int = logging.INFO # Logging level
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' # Logging format
    redis_timeout: int = 5 # Timeout for Redis connection
    chroma_timeout: int = 10 # Timeout for ChromaDB connection

class MemoryManager:
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config
        self.redis_client = Redis(
            host=config.redis_host, 
            port=config.redis_port, 
            decode_responses=True
        )
        
        # Ensure persistence directory exists
        persist_dir = config.chroma_persist_directory or "chroma_db"
        os.makedirs(persist_dir, exist_ok=True)
        
        # Initialize persistent ChromaDB with settings
        self.db = chromadb.PersistentClient(
            path=persist_dir,
            settings=chromadb.Settings(
                persist_directory=persist_dir,
                is_persistent=True
            )
        )
        
        # Get existing or create new collection
        try:
            self.collection = self.db.get_collection(name=config.chroma_collection_name)
            logger.info(f"Connected to existing collection: {config.chroma_collection_name}")
        except:
            self.collection = self.db.create_collection(name=config.chroma_collection_name)
            logger.info(f"Created new collection: {config.chroma_collection_name}")

    def set_short_term_memory(self, session_id: str, intent: str) -> None:
        """Set the user's intent in session context using Redis."""
        try:
            self.redis_client.set(session_id, intent, ex=self.config.redis_ttl)
            logger.info(f"Short-term memory set for session {session_id}")
        except Exception as e:
            logger.error(f"Error setting short-term memory: {str(e)}")

    def get_short_term_memory(self, session_id: str) -> Optional[str]:
        """Get the user's previous intent from session context."""
        try:
            return self.redis_client.get(session_id)
        except Exception as e:
            logger.error(f"Error getting short-term memory: {str(e)}")
            return None

    def clear_short_term_memory(self, session_id: str) -> None:
        """Clear the short-term memory for a specific session."""
        try:
            self.redis_client.delete(session_id)
            logger.info(f"Short-term memory cleared for session {session_id}")
        except Exception as e:
            logger.error(f"Error clearing short-term memory: {str(e)}")

    def store_long_term_memory(self, session_id: str, message: str) -> None:
        """Store user interactions in ChromaDB for long-term memory with metadata."""
        try:
            unique_id = f"{session_id}-{uuid.uuid4()}"
            document_data = json.dumps({
                "session_id": session_id,
                "message": message,
                "timestamp": datetime.utcnow().isoformat()
            })
            self.collection.add(
                documents=[document_data],
                ids=[unique_id],
                metadatas=[{"session_id": session_id}]
            )
            logger.info(f"Long-term memory stored for session {session_id}")
        except Exception as e:
            logger.error(f"Error storing long-term memory: {str(e)}")

    def retrieve_long_term_memory(self, session_id: Optional[str] = None) -> List[dict]:
        """Retrieve stored messages from ChromaDB."""
        try:
            if session_id:
                # Get total count first
                all_results = self.collection.get(
                    where={"session_id": session_id}
                )
                total_docs = len(all_results['documents']) if all_results else 0
                
                # Use min of max_results and total docs
                n_results = min(self.config.max_results, total_docs) if total_docs > 0 else self.config.max_results
                
                results = self.collection.query(
                    query_texts=[session_id],
                    where={"session_id": session_id},
                    n_results=n_results
                )
                documents = [json.loads(doc) for sublist in results['documents'] for doc in sublist]
            else:
                results = self.collection.get()
                documents = [json.loads(doc) for doc in results['documents']]
            
            return sorted(documents, key=lambda x: x['timestamp'])
        except Exception as e:
            logger.error(f"Error retrieving long-term memory: {str(e)}")
            return []
        
    def clear_long_term_memory(self, session_id: Optional[str] = None) -> None:
        """Clear stored messages in long-term memory (ChromaDB)."""
        try:
            if session_id:
                results = self.collection.query(
                    query_texts=[session_id],
                    where={"session_id": session_id},
                    n_results=self.config.max_results
                )
                document_ids = [doc_id for sublist in results['ids'] for doc_id in sublist]
                if document_ids:
                    self.collection.delete(ids=document_ids)
            else:
                all_results = self.collection.get()
                if all_results and 'ids' in all_results and all_results['ids']:
                    self.collection.delete(ids=all_results['ids'])
            logger.info(f"Long-term memory cleared for {'all sessions' if session_id is None else f'session {session_id}'}")
        except Exception as e:
            logger.error(f"Error clearing long-term memory: {str(e)}")

if __name__ == "__main__":
    # Initialize memory manager
    config = MemoryConfig()
    memory_manager = MemoryManager(config)

    # Test short-term memory
    test_session = "test_session_123"
    memory_manager.set_short_term_memory(test_session, "greeting")
    result = memory_manager.get_short_term_memory(test_session)
    print(f"Short-term memory test result: {result}")

    # Test long-term memory
    memory_manager.store_long_term_memory(test_session, "Hello, this is a test message")
    memory_manager.store_long_term_memory(test_session, "Second test message")
    messages = memory_manager.retrieve_long_term_memory(test_session)
    print(f"Long-term memory test messages: {messages}")

    # Test memory clearing
    memory_manager.clear_short_term_memory(test_session)
    memory_manager.clear_long_term_memory(test_session)
    print("Memory cleared successfully")
