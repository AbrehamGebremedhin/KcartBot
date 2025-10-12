"""PDF dataloader for processing context documents and storing in Milvus vector database."""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib

# PDF and text processing
from pypdf import PdfReader

# Langchain for text splitting
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    TokenTextSplitter,
)

# Google Gemini
from google import genai
from google.genai import types

# Milvus handler
from app.db.milvus_handler import MilvusHandler
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class PDFDataLoader:
    """
    Loads PDF documents, splits them into chunks, generates embeddings using Gemini,
    and stores them in Milvus vector database.
    """
    
    def __init__(
        self,
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        collection_name: str = "KCartBot",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "models/text-embedding-004",
        chunking_strategy: str = "dynamic",  # "dynamic", "semantic", or "fixed"
    ):
        """
        Initialize the PDF data loader.
        
        Args:
            milvus_host: Milvus server host
            milvus_port: Milvus server port
            collection_name: Name of the Milvus collection to store embeddings
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            embedding_model: Gemini embedding model to use
            chunking_strategy: Strategy for chunking ("dynamic", "semantic", or "fixed")
        """
        self.settings = get_settings()
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.chunking_strategy = chunking_strategy
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=self.settings.gemini_api_key)
        
        # Initialize Milvus handler
        self.milvus = MilvusHandler(
            host=milvus_host,
            port=milvus_port,
        )
        
        # Initialize text splitters based on strategy
        self._init_text_splitters()
        
        logger.info(f"PDFDataLoader initialized with collection '{collection_name}' using '{chunking_strategy}' chunking strategy")
    
    def _init_text_splitters(self) -> None:
        """Initialize text splitters based on the chunking strategy."""
        if self.chunking_strategy == "dynamic":
            # Dynamic chunking: Uses multiple separators with priority
            # Best for general documents with varied structure
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=[
                    "\n\n\n",  # Multiple newlines (section breaks)
                    "\n\n",    # Paragraph breaks
                    "\n",      # Line breaks
                    ". ",      # Sentences
                    "! ",      # Exclamations
                    "? ",      # Questions
                    "; ",      # Semicolons
                    ", ",      # Commas
                    " ",       # Words
                    "",        # Characters
                ],
                keep_separator=True,
            )
        elif self.chunking_strategy == "semantic":
            # Semantic chunking: Preserves sentence boundaries
            # Best for narrative content
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", "! ", "? "],
                keep_separator=True,
            )
        else:  # fixed
            # Fixed chunking: Simple character-based splitting
            # Fastest but may break semantic boundaries
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=[" ", ""],
            )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            reader = PdfReader(pdf_path)
            text_content = []
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    text_content.append(text)
                    logger.debug(f"Extracted text from page {page_num}")
            
            full_text = "\n\n".join(text_content)
            logger.info(f"Extracted {len(full_text)} characters from {len(reader.pages)} pages")
            return full_text
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF '{pdf_path}': {str(e)}")
            raise
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks using RecursiveCharacterTextSplitter.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        try:
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Split text into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Failed to split text: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for text chunks using Gemini.
        
        Args:
            texts: List of text chunks
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = []
            batch_size = 100  # Process in batches to avoid rate limits
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                logger.info(f"Generating embeddings for batch {i//batch_size + 1} ({len(batch)} texts)")
                
                for text in batch:
                    # Generate embedding using Gemini
                    result = self.client.models.embed_content(
                        model=self.embedding_model,
                        contents=text,
                    )
                    embeddings.append(result.embeddings[0].values)
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise
    
    def generate_chunk_id(self, text: str, index: int) -> int:
        """
        Generate a unique ID for a text chunk.
        
        Args:
            text: The text chunk
            index: Index of the chunk
            
        Returns:
            Unique integer ID
        """
        # Create hash of text + index
        hash_input = f"{text}_{index}".encode('utf-8')
        hash_value = hashlib.md5(hash_input).hexdigest()
        # Convert first 8 characters of hex to int
        return int(hash_value[:8], 16)
    
    async def setup_collection(self, dimension: int = 768) -> None:
        """
        Set up the Milvus collection with appropriate schema and index.
        
        Args:
            dimension: Dimension of the embedding vectors (768 for text-embedding-004)
        """
        try:
            # Ensure connection (idempotent)
            if not self.milvus.is_connected():
                await self.milvus.connect()
            
            # Check if collection exists
            if self.milvus.collection_exists(self.collection_name):
                logger.info(f"Collection '{self.collection_name}' already exists")
                # Optionally drop and recreate
                # await self.milvus.drop_collection(self.collection_name)
            else:
                # Create collection with schema
                from pymilvus import FieldSchema, DataType
                
                additional_fields = [
                    FieldSchema(
                        name="text",
                        dtype=DataType.VARCHAR,
                        max_length=65535,
                        description="Original text chunk"
                    ),
                    FieldSchema(
                        name="source",
                        dtype=DataType.VARCHAR,
                        max_length=512,
                        description="Source document path"
                    ),
                    FieldSchema(
                        name="chunk_index",
                        dtype=DataType.INT32,
                        description="Index of chunk in document"
                    ),
                ]
                
                await self.milvus.create_collection(
                    collection_name=self.collection_name,
                    dimension=dimension,
                    id_field="id",
                    vector_field="embedding",
                    additional_fields=additional_fields,
                    description="KCartBot context embeddings from PDF documents",
                    auto_id=False,
                )
                
                # Create index
                await self.milvus.create_index(
                    collection_name=self.collection_name,
                    field_name="embedding",
                    index_type="IVF_FLAT",
                    metric_type="COSINE",
                    params={"nlist": 128},
                )
                
                logger.info(f"Created collection '{self.collection_name}' with index")
            
        except Exception as e:
            logger.error(f"Failed to setup collection: {str(e)}")
            raise
    
    async def load_pdf_to_milvus(
        self,
        pdf_path: str,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Complete pipeline: Extract PDF, chunk, embed, and store in Milvus.
        
        Args:
            pdf_path: Path to the PDF file
            overwrite: Whether to drop existing collection before loading
            
        Returns:
            Dictionary with loading statistics
        """
        try:
            # Verify PDF exists
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            logger.info(f"Starting PDF loading pipeline for: {pdf_path}")
            
            # Step 1: Extract text from PDF
            logger.info("Step 1: Extracting text from PDF...")
            full_text = self.extract_text_from_pdf(pdf_path)
            
            # Step 2: Split into chunks
            logger.info("Step 2: Splitting text into chunks...")
            chunks = self.split_text_into_chunks(full_text)
            
            # Step 3: Generate embeddings
            logger.info("Step 3: Generating embeddings with Gemini...")
            embeddings = self.generate_embeddings(chunks)
            
            # Verify embedding dimension
            embedding_dim = len(embeddings[0]) if embeddings else 768
            logger.info(f"Embedding dimension: {embedding_dim}")
            
            # Step 4: Setup Milvus collection
            logger.info("Step 4: Setting up Milvus collection...")
            await self.milvus.connect()
            
            if overwrite and self.milvus.collection_exists(self.collection_name):
                await self.milvus.drop_collection(self.collection_name)
            
            await self.setup_collection(dimension=embedding_dim)
            
            # Step 5: Prepare data for insertion
            logger.info("Step 5: Preparing data for Milvus insertion...")
            source_path = str(Path(pdf_path).name)
            
            # Prepare data as list of entities
            entities = []
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                entity = {
                    "id": self.generate_chunk_id(chunk, idx),
                    "embedding": embedding,
                    "text": chunk,
                    "source": source_path,
                    "chunk_index": idx,
                }
                entities.append(entity)
            
            # Step 6: Insert into Milvus
            logger.info("Step 6: Inserting data into Milvus...")
            result = await self.milvus.insert(
                collection_name=self.collection_name,
                data=entities,
            )
            
            # Flush to ensure data is persisted
            await self.milvus.flush([self.collection_name])
            
            # Load collection for searching
            await self.milvus.load_collection(self.collection_name)
            
            # Get statistics
            stats = {
                "pdf_path": pdf_path,
                "total_characters": len(full_text),
                "num_chunks": len(chunks),
                "embedding_dimension": embedding_dim,
                "inserted_count": result.insert_count,
                "collection_name": self.collection_name,
            }
            
            logger.info(f"Successfully loaded PDF to Milvus: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to load PDF to Milvus: {str(e)}")
            raise
        finally:
            # Clean up connection
            if self.milvus.is_connected():
                self.milvus.disconnect()
    
    async def search_similar(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar text chunks in the vector database.
        
        Args:
            query_text: Query text to search for
            top_k: Number of results to return
            
        Returns:
            List of search results with text and metadata
        """
        try:
            await self.milvus.connect()
            
            # Generate embedding for query
            result = self.client.models.embed_content(
                model=self.embedding_model,
                contents=query_text,
            )
            query_embedding = result.embeddings[0].values
            
            # Search in Milvus
            results = await self.milvus.search(
                collection_name=self.collection_name,
                query_vectors=[query_embedding],
                field_name="embedding",
                metric_type="COSINE",
                top_k=top_k,
                output_fields=["text", "source", "chunk_index"],
                params={"nprobe": 10},
            )
            
            # Format results
            formatted_results = []
            for hit in results[0]:
                formatted_results.append({
                    "text": hit["entity"]["text"],
                    "source": hit["entity"]["source"],
                    "chunk_index": hit["entity"]["chunk_index"],
                    "score": hit["score"],
                    "distance": hit["distance"],
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search: {str(e)}")
            raise
        finally:
            if self.milvus.is_connected():
                self.milvus.disconnect()


async def load_context_pdf(
    pdf_path: str = "data/context.pdf",
    collection_name: str = "KCartBot",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to load the context.pdf file into Milvus.
    
    Args:
        pdf_path: Path to the PDF file (default: data/context.pdf)
        collection_name: Milvus collection name (default: KCartBot)
        overwrite: Whether to overwrite existing collection
        
    Returns:
        Dictionary with loading statistics
    """
    loader = PDFDataLoader(collection_name=collection_name)
    stats = await loader.load_pdf_to_milvus(pdf_path, overwrite=overwrite)
    return stats


# Example usage
if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def main():
        # Load the context.pdf file
        stats = await load_context_pdf(
            pdf_path="data/context.pdf",
            collection_name="KCartBot",
            overwrite=True,  # Set to False to append instead of replace
        )
        print("\n=== Loading Complete ===")
        print(f"Loaded: {stats['num_chunks']} chunks from {stats['pdf_path']}")
        print(f"Collection: {stats['collection_name']}")
        print(f"Embedding dimension: {stats['embedding_dimension']}")
        
        # Test search
        loader = PDFDataLoader(collection_name="KCartBot")
        results = await loader.search_similar("What is this document about?", top_k=3)
        
        print("\n=== Sample Search Results ===")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (score: {result['score']:.4f}):")
            print(f"Source: {result['source']}, Chunk: {result['chunk_index']}")
            print(f"Text preview: {result['text'][:200]}...")
    
    asyncio.run(main())
