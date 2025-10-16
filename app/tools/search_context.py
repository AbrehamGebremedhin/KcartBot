"""Vector search tool for RAG (Retrieval-Augmented Generation) using Milvus."""

from typing import Any, Dict, List, Optional
import asyncio
import json
import logging
from datetime import datetime, timedelta

from app.tools.base import ToolBase
from app.db.milvus_handler import MilvusHandler
from app.core.config import get_settings

from google import genai

logger = logging.getLogger(__name__)


class VectorDBUnavailableError(RuntimeError):
    """Raised when the vector database cannot be reached."""

    def __init__(self, message: str, *, details: Optional[str] = None, retry_after: Optional[int] = None) -> None:
        super().__init__(message)
        self.details = details
        self.retry_after = retry_after


class VectorSearchTool(ToolBase):
    """
    Tool for AI agent to search for relevant context using vector similarity search.
    
    This tool enables Retrieval-Augmented Generation (RAG) by searching through
    vectorized product knowledge, documentation, and context stored in Milvus.
    
    Use cases:
    - Answer questions about products, features, and services
    - Retrieve relevant documentation and knowledge base articles
    - Find contextually similar information for enhanced responses
    - Support decision-making with factual, stored knowledge
    """

    def __init__(
        self,
        collection_name: str = "KCartBot",
        embedding_model: str = "models/text-embedding-004",
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
    ):
        """
        Initialize the Vector Search Tool.
        
        Args:
            collection_name: Name of the Milvus collection to search
            embedding_model: Gemini embedding model to use for query encoding
            milvus_host: Milvus server host
            milvus_port: Milvus server port
        """
        super().__init__(
            name="vector_search",
            description=(
                "Search for relevant context and knowledge using semantic vector similarity. "
                "Provide a natural language query to retrieve the most relevant information "
                "from the knowledge base. "
                "Input format: {'query': 'your search query', 'top_k': 5, 'min_score': 0.5} "
                "Returns: List of relevant text chunks with similarity scores and metadata."
            )
        )
        
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.settings = get_settings()
        
        # Initialize Gemini client for embeddings
        self.client = genai.Client(api_key=self.settings.gemini_api_key)
        
        # Initialize Milvus handler
        self.milvus = MilvusHandler(
            host=milvus_host,
            port=milvus_port,
        )
        
        # Connection state tracking
        self._connected = False
        self._connection_failure: Optional[Dict[str, Any]] = None
        self._next_retry_at: Optional[datetime] = None
        
        logger.info(f"VectorSearchTool initialized with collection '{collection_name}'")

    async def _ensure_connection(self) -> None:
        """Ensure Milvus connection is active."""
        now = datetime.utcnow()

        if self._connection_failure and self._next_retry_at and now < self._next_retry_at:
            remaining = self._retry_delay_seconds(now)
            message = self._connection_failure.get(
                "message",
                "Vector database is currently unavailable."
            )
            raise VectorDBUnavailableError(
                message,
                details=self._connection_failure.get("details"),
                retry_after=remaining,
            )

        try:
            if not self._connected or not self.milvus.is_connected():
                await self.milvus.connect()
                self._connected = True
                self._connection_failure = None
                self._next_retry_at = None

                # Verify collection exists
                if not self.milvus.collection_exists(self.collection_name):
                    logger.warning(
                        f"Collection '{self.collection_name}' does not exist. "
                        "Please load data using the dataloader utility first."
                    )
                    raise ValueError(
                        f"Vector database collection '{self.collection_name}' not found. "
                        "Please ensure the knowledge base has been loaded."
                    )

                # Load collection into memory for searching
                if not self.milvus.is_collection_loaded(self.collection_name):
                    await self.milvus.load_collection(self.collection_name)
                    logger.info(f"Loaded collection '{self.collection_name}' into memory")
        except Exception as exc:  # pragma: no cover - defensive connection handling
            self._connected = False
            self._connection_failure = {
                "message": (
                    "Vector database connection failed. Milvus may be offline or unreachable "
                    f"at {self.milvus.host}:{self.milvus.port}."
                ),
                "details": str(exc),
                "timestamp": now,
            }
            self._next_retry_at = now + timedelta(minutes=5)
            retry_after = self._retry_delay_seconds(now)
            logger.warning(
                "Vector search connection failure (retry in %s seconds): %s",
                retry_after,
                exc,
            )
            raise VectorDBUnavailableError(
                self._connection_failure["message"],
                details=str(exc),
                retry_after=retry_after,
            ) from exc

    def _retry_delay_seconds(self, now: Optional[datetime] = None) -> Optional[int]:
        if self._next_retry_at is None:
            return None
        now = now or datetime.utcnow()
        delta = self._next_retry_at - now
        return max(int(delta.total_seconds()), 0)

    def _generate_query_embedding(self, query_text: str) -> List[float]:
        """
        Generate embedding vector for the query text.
        
        Args:
            query_text: The search query
            
        Returns:
            Embedding vector as a list of floats
        """
        try:
            result = self.client.models.embed_content(
                model=self.embedding_model,
                contents=query_text,
            )
            embedding = result.embeddings[0].values
            logger.debug(f"Generated embedding for query: {query_text[:50]}...")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {str(e)}")
            raise

    async def _search_vectors(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_score: Optional[float] = None,
        filters: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in Milvus.
        
        Args:
            query_embedding: The query vector
            top_k: Number of results to return
            min_score: Minimum similarity score threshold (0-1)
            filters: Optional Milvus boolean expression for filtering
            
        Returns:
            List of search results with text, metadata, and scores
        """
        try:
            # Search in Milvus
            results = await self.milvus.search(
                collection_name=self.collection_name,
                query_vectors=[query_embedding],
                field_name="embedding",
                metric_type="COSINE",  # Cosine similarity (higher is better)
                top_k=top_k,
                output_fields=["text", "source", "chunk_index"],
                params={"nprobe": 10},  # Number of partitions to probe
                expr=filters,
            )
            
            # Format and filter results
            formatted_results = []
            hits_for_query = results[0] if results else []
            for hit in hits_for_query:  # results[0] because we only have one query vector
                # Normalise milvus hit into a plain dictionary
                if isinstance(hit, dict):
                    distance = hit.get("distance")
                    entity = hit.get("entity") or {}
                    raw_score = hit.get("score")
                else:
                    distance = getattr(hit, "distance", None)
                    raw_score = getattr(hit, "score", None)
                    entity = getattr(hit, "entity", None)
                    if hasattr(entity, "to_dict"):
                        entity = entity.to_dict()
                    elif not isinstance(entity, dict):
                        entity = {}

                if distance is None:
                    logger.debug("Skipping hit without distance: %s", hit)
                    continue

                # Convert distance to similarity score (Cosine: 1 - distance)
                # Milvus returns distance, we want similarity (0-1 scale)
                similarity_score = 1 - distance if distance <= 1 else max(0.0, 1 - distance)

                # Apply minimum score threshold if specified
                if min_score is not None and similarity_score < min_score:
                    continue

                text_value = ""
                source_value = "unknown"
                chunk_index_value = -1

                # First check if hit is a dict and has entity field (nested structure)
                if isinstance(hit, dict) and "entity" in hit:
                    entity = hit["entity"]
                    
                    # Handle pymilvus Hit objects
                    if hasattr(entity, 'entity') and hasattr(entity, 'to_dict'):
                        # This is a pymilvus Hit object with nested entity
                        try:
                            entity_dict = entity.to_dict()
                            inner_entity = entity_dict.get("entity", {})
                            if isinstance(inner_entity, dict):
                                text_value = inner_entity.get("text", "")
                                source_value = inner_entity.get("source", "unknown")
                                chunk_index_value = inner_entity.get("chunk_index", -1)
                        except Exception as e:
                            logger.debug(f"Error parsing Hit object: {e}")
                    
                    elif isinstance(entity, dict):
                        # Check for nested entity structure (Milvus returns nested entities)
                        inner_entity = entity.get("entity")
                        if isinstance(inner_entity, dict):
                            # Use the inner entity which contains the actual data
                            text_value = inner_entity.get("text", "")
                            source_value = inner_entity.get("source", "unknown")
                            chunk_index_value = inner_entity.get("chunk_index", -1)
                        else:
                            # Fallback to direct entity access
                            text_value = entity.get("text", "")
                            source_value = entity.get("source", "unknown")
                            chunk_index_value = entity.get("chunk_index", -1)
                # Fallback: check if hit itself has the fields
                elif isinstance(hit, dict):
                    text_value = hit.get("text", "")
                    source_value = hit.get("source", "unknown")
                    chunk_index_value = hit.get("chunk_index", -1)

                formatted_results.append({
                    "text": text_value,
                    "source": source_value,
                    "chunk_index": chunk_index_value,
                    "score": round(similarity_score, 4),
                    "distance": round(distance, 4),
                    "raw_score": raw_score,
                })
            
            logger.info(
                "Found %s results (from %s total)",
                len(formatted_results),
                len(hits_for_query),
            )
            return formatted_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            raise

    async def run(self, input: Any, context: Dict[str, Any] = None) -> Any:
        """
        Execute vector search to retrieve relevant context.
        
        Args:
            input: Search parameters as a dictionary or JSON string:
                - query (str, required): Natural language search query
                - top_k (int, optional): Number of results to return (default: 5)
                - min_score (float, optional): Minimum similarity score 0-1 (default: None)
                - filters (str, optional): Milvus boolean expression for filtering
                - include_metadata (bool, optional): Whether to include metadata (default: True)
                - format (str, optional): 'detailed' or 'text_only' (default: 'detailed')
                
            context: Optional context dictionary (can contain default parameters)
            
        Returns:
            Search results as a dictionary containing:
            - query: The original query
            - results: List of relevant text chunks with scores
            - count: Number of results returned
            - metadata: Search parameters used
        """
        query: Optional[str] = None
        try:
            # Parse input if it's a string
            if isinstance(input, str):
                try:
                    input = json.loads(input)
                except json.JSONDecodeError:
                    # Treat as a simple query string
                    input = {"query": input}
            
            # Extract parameters
            query = input.get("query")
            if not query:
                return {
                    "error": "Query is required. Provide a natural language search query.",
                    "example": {"query": "What are the available vegetables?", "top_k": 5}
                }
            
            top_k = input.get("top_k", 5)
            min_score = input.get("min_score")
            filters = input.get("filters")
            include_metadata = input.get("include_metadata", True)
            output_format = input.get("format", "detailed")
            
            # Validate parameters
            if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
                return {"error": "top_k must be an integer between 1 and 100"}
            
            if min_score is not None:
                if not isinstance(min_score, (int, float)) or min_score < 0 or min_score > 1:
                    return {"error": "min_score must be a number between 0 and 1"}
            
            # Ensure connection
            await self._ensure_connection()
            
            # Generate query embedding
            logger.info(f"Processing search query: '{query}'")
            query_embedding = self._generate_query_embedding(query)
            
            # Perform vector search
            search_results = await self._search_vectors(
                query_embedding=query_embedding,
                top_k=top_k,
                min_score=min_score,
                filters=filters,
            )
            
            # Format response based on requested format
            if output_format == "text_only":
                # Return just the text content for easy integration
                return {
                    "query": query,
                    "results": [r["text"] for r in search_results],
                    "count": len(search_results),
                }
            
            # Detailed format (default)
            response = {
                "query": query,
                "results": search_results,
                "count": len(search_results),
            }
            
            if include_metadata:
                response["metadata"] = {
                    "collection": self.collection_name,
                    "top_k": top_k,
                    "min_score": min_score,
                    "filters": filters,
                    "embedding_model": self.embedding_model,
                }
            
            # Add a helpful context summary for the AI agent
            if search_results:
                response["context_summary"] = self._create_context_summary(search_results)
            
            return response
        except VectorDBUnavailableError as exc:
            retry_after = exc.retry_after
            details: Dict[str, Any] = {
                "host": getattr(self.milvus, "host", "localhost"),
                "port": getattr(self.milvus, "port", "19530"),
            }
            if exc.details:
                details["last_error"] = exc.details
            if retry_after is not None:
                details["retry_after_seconds"] = retry_after

            logger.info("Vector DB unavailable: %s", exc.details or exc)
            return {
                "error": "vector_db_unavailable",
                "message": str(exc),
                "query": query,
                "details": details,
            }
        except Exception as e:
            logger.error(f"Vector search tool error: {str(e)}")
            return {
                "error": f"Search failed: {str(e)}",
                "query": input.get("query") if isinstance(input, dict) else str(input),
            }

    def _create_context_summary(self, results: List[Dict[str, Any]]) -> str:
        """
        Create a concise summary of the retrieved context for the AI agent.
        
        Args:
            results: List of search results
            
        Returns:
            Summary text
        """
        if not results:
            return "No relevant context found."
        
        # Get top 3 most relevant chunks
        top_results = results[:3]
        
        summary_parts = [
            f"Found {len(results)} relevant context chunks.",
            f"Top result (score: {top_results[0]['score']}) from {top_results[0]['source']}.",
        ]
        
        # Calculate average score
        avg_score = sum(r['score'] for r in results) / len(results)
        summary_parts.append(f"Average relevance score: {avg_score:.2f}")
        
        return " ".join(summary_parts)

    async def get_context_for_query(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.5,
    ) -> str:
        """
        Convenience method to get formatted context text for a query.
        Useful for quick RAG integration.
        
        Args:
            query: Search query
            top_k: Number of results
            min_score: Minimum similarity score
            
        Returns:
            Formatted context text ready to be added to prompts
        """
        result = await self.run({
            "query": query,
            "top_k": top_k,
            "min_score": min_score,
            "format": "detailed",
        })
        
        if "error" in result:
            return f"Context retrieval failed: {result['error']}"
        
        if not result.get("results"):
            return "No relevant context found in the knowledge base."
        
        # Format context for prompt injection
        context_parts = ["=== RETRIEVED CONTEXT ===\n"]
        for i, item in enumerate(result["results"], 1):
            context_parts.append(
                f"[{i}] (Relevance: {item['score']:.2f})\n"
                f"{item['text']}\n"
            )
        
        return "\n".join(context_parts)

    async def aclose(self) -> None:
        """Asynchronously close open Milvus connections."""
        if not self._connected or not self.milvus.is_connected():
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.milvus.disconnect)
        self._connected = False

    def __del__(self):
        """Cleanup: disconnect from Milvus."""
        try:
            if self._connected and self.milvus.is_connected():
                self.milvus.disconnect()
                self._connected = False
        except Exception:
            pass


def get_vector_search_tool(**kwargs) -> VectorSearchTool:
    """Convenience factory mirroring previous public API."""
    return VectorSearchTool(**kwargs)
