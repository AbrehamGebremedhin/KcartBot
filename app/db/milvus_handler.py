"""Milvus vector database handler for managing vector operations."""

from typing import List, Dict, Any, Optional, Union
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from pymilvus.orm.mutation import MutationResult
from pymilvus.client.types import LoadState
import logging
from functools import wraps

logger = logging.getLogger(__name__)


def ensure_connection(func):
    """Decorator to ensure Milvus connection is active before operations."""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        if not self.is_connected():
            await self.connect()
        return await func(self, *args, **kwargs)
    return wrapper


class MilvusHandler:
    """
    Comprehensive handler for Milvus vector database operations.
    
    Provides functionality for:
    - Connection management
    - Collection management (create, drop, list)
    - Index management
    - Vector insertion, search, and deletion
    - Data querying and retrieval
    - Collection statistics and maintenance
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        user: str = "",
        password: str = "",
        alias: str = "default",
        **kwargs
    ):
        """
        Initialize Milvus handler.
        
        Args:
            host: Milvus server host
            port: Milvus server port
            user: Username for authentication
            password: Password for authentication
            alias: Connection alias name
            **kwargs: Additional connection parameters
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.alias = alias
        self.connection_params = kwargs
        self._collections: Dict[str, Collection] = {}
        
    async def connect(self) -> None:
        """Establish connection to Milvus server."""
        try:
            connections.connect(
                alias=self.alias,
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                **self.connection_params
            )
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            raise
    
    def disconnect(self) -> None:
        """Disconnect from Milvus server."""
        try:
            connections.disconnect(alias=self.alias)
            self._collections.clear()
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Error disconnecting from Milvus: {str(e)}")
            raise
    
    def is_connected(self) -> bool:
        """Check if connected to Milvus server."""
        try:
            return connections.has_connection(self.alias)
        except Exception:
            return False
    
    # ========== Collection Management ==========
    
    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        id_field: str = "id",
        vector_field: str = "embedding",
        additional_fields: Optional[List[FieldSchema]] = None,
        description: str = "",
        auto_id: bool = False,
        primary_field_type: DataType = DataType.INT64,
    ) -> Collection:
        """
        Create a new collection with schema.
        
        Args:
            collection_name: Name of the collection
            dimension: Vector dimension
            id_field: Name of the primary key field
            vector_field: Name of the vector field
            additional_fields: Additional fields to include in schema
            description: Collection description
            auto_id: Whether to auto-generate IDs
            primary_field_type: Data type for primary key field
            
        Returns:
            Created Collection object
        """
        try:
            if self.collection_exists(collection_name):
                logger.warning(f"Collection '{collection_name}' already exists")
                return self.get_collection(collection_name)
            
            # Define primary key field
            fields = [
                FieldSchema(
                    name=id_field,
                    dtype=primary_field_type,
                    is_primary=True,
                    auto_id=auto_id,
                    description="Primary key"
                ),
                # Define vector field
                FieldSchema(
                    name=vector_field,
                    dtype=DataType.FLOAT_VECTOR,
                    dim=dimension,
                    description="Vector embeddings"
                ),
            ]
            
            # Add additional fields if provided
            if additional_fields:
                fields.extend(additional_fields)
            
            schema = CollectionSchema(
                fields=fields,
                description=description or f"Collection for {collection_name}"
            )
            
            collection = Collection(
                name=collection_name,
                schema=schema,
                using=self.alias
            )
            
            self._collections[collection_name] = collection
            logger.info(f"Created collection '{collection_name}'")
            return collection
            
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {str(e)}")
            raise
    
    def get_collection(self, collection_name: str) -> Collection:
        """
        Get a collection object.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection object
        """
        if collection_name not in self._collections:
            self._collections[collection_name] = Collection(
                name=collection_name,
                using=self.alias
            )
        return self._collections[collection_name]
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if collection exists, False otherwise
        """
        return utility.has_collection(collection_name, using=self.alias)
    
    async def drop_collection(self, collection_name: str) -> None:
        """
        Drop a collection.
        
        Args:
            collection_name: Name of the collection to drop
        """
        try:
            if collection_name in self._collections:
                del self._collections[collection_name]
            utility.drop_collection(collection_name, using=self.alias)
            logger.info(f"Dropped collection '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to drop collection '{collection_name}': {str(e)}")
            raise
    
    def list_collections(self) -> List[str]:
        """
        List all collections.
        
        Returns:
            List of collection names
        """
        return utility.list_collections(using=self.alias)
    
    # ========== Index Management ==========
    
    async def create_index(
        self,
        collection_name: str,
        field_name: str = "embedding",
        index_type: str = "IVF_FLAT",
        metric_type: str = "L2",
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Create an index for a field.
        
        Args:
            collection_name: Name of the collection
            field_name: Field to create index on
            index_type: Type of index (IVF_FLAT, IVF_SQ8, IVF_PQ, HNSW, etc.)
            metric_type: Distance metric (L2, IP, COSINE)
            params: Index-specific parameters
        """
        try:
            collection = self.get_collection(collection_name)
            
            default_params = {"nlist": 128}
            if index_type == "HNSW":
                default_params = {"M": 16, "efConstruction": 200}
            elif index_type == "IVF_PQ":
                default_params = {"nlist": 128, "m": 8, "nbits": 8}
            
            index_params = params or default_params
            
            index = {
                "index_type": index_type,
                "metric_type": metric_type,
                "params": index_params
            }
            
            collection.create_index(
                field_name=field_name,
                index_params=index
            )
            logger.info(f"Created {index_type} index on '{collection_name}.{field_name}'")
            
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            raise
    
    async def drop_index(self, collection_name: str, field_name: str = "embedding") -> None:
        """
        Drop an index from a field.
        
        Args:
            collection_name: Name of the collection
            field_name: Field to drop index from
        """
        try:
            collection = self.get_collection(collection_name)
            collection.drop_index(field_name=field_name)
            logger.info(f"Dropped index on '{collection_name}.{field_name}'")
        except Exception as e:
            logger.error(f"Failed to drop index: {str(e)}")
            raise
    
    def get_index_info(self, collection_name: str, field_name: str = "embedding") -> Dict[str, Any]:
        """
        Get index information for a field.
        
        Args:
            collection_name: Name of the collection
            field_name: Field name
            
        Returns:
            Index information dictionary
        """
        collection = self.get_collection(collection_name)
        return collection.index(field_name=field_name).params
    
    # ========== Data Operations ==========
    
    async def insert(
        self,
        collection_name: str,
        data: Union[List[List], Dict[str, List]],
        partition_name: Optional[str] = None,
    ) -> MutationResult:
        """
        Insert data into collection.
        
        Args:
            collection_name: Name of the collection
            data: Data to insert (list of lists or dict of field_name: values)
            partition_name: Optional partition name
            
        Returns:
            MutationResult containing insert information
        """
        try:
            collection = self.get_collection(collection_name)
            result = collection.insert(data, partition_name=partition_name)
            logger.info(f"Inserted {result.insert_count} entities into '{collection_name}'")
            return result
        except Exception as e:
            logger.error(f"Failed to insert data into '{collection_name}': {str(e)}")
            raise
    
    async def upsert(
        self,
        collection_name: str,
        data: Union[List[List], Dict[str, List]],
        partition_name: Optional[str] = None,
    ) -> MutationResult:
        """
        Upsert (insert or update) data into collection.
        
        Args:
            collection_name: Name of the collection
            data: Data to upsert
            partition_name: Optional partition name
            
        Returns:
            MutationResult containing upsert information
        """
        try:
            collection = self.get_collection(collection_name)
            result = collection.upsert(data, partition_name=partition_name)
            logger.info(f"Upserted {result.upsert_count} entities in '{collection_name}'")
            return result
        except Exception as e:
            logger.error(f"Failed to upsert data into '{collection_name}': {str(e)}")
            raise
    
    async def delete(
        self,
        collection_name: str,
        expr: str,
        partition_name: Optional[str] = None,
    ) -> MutationResult:
        """
        Delete entities from collection.
        
        Args:
            collection_name: Name of the collection
            expr: Boolean expression for deletion (e.g., "id in [1, 2, 3]")
            partition_name: Optional partition name
            
        Returns:
            MutationResult containing deletion information
        """
        try:
            collection = self.get_collection(collection_name)
            result = collection.delete(expr, partition_name=partition_name)
            logger.info(f"Deleted {result.delete_count} entities from '{collection_name}'")
            return result
        except Exception as e:
            logger.error(f"Failed to delete data from '{collection_name}': {str(e)}")
            raise
    
    # ========== Search Operations ==========
    
    async def search(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        field_name: str = "embedding",
        metric_type: str = "L2",
        top_k: int = 10,
        params: Optional[Dict[str, Any]] = None,
        expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        partition_names: Optional[List[str]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Search for similar vectors.
        
        Args:
            collection_name: Name of the collection
            query_vectors: Query vectors to search for
            field_name: Vector field name
            metric_type: Distance metric
            top_k: Number of results to return
            params: Search parameters
            expr: Boolean expression for filtering
            output_fields: Fields to return in results
            partition_names: Partitions to search in
            
        Returns:
            List of search results for each query vector
        """
        try:
            collection = self.get_collection(collection_name)
            
            # Load collection if not loaded
            try:
                if utility.load_state(collection_name, using=self.alias) != LoadState.Loaded:
                    collection.load()
            except Exception:
                # If load_state check fails, just try to load
                collection.load()
            
            default_params = {"nprobe": 10}
            search_params = params or default_params
            
            results = collection.search(
                data=query_vectors,
                anns_field=field_name,
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=output_fields,
                partition_names=partition_names,
            )
            
            # Format results
            formatted_results = []
            for hits in results:
                hit_list = []
                for hit in hits:
                    hit_dict = {
                        "id": hit.id,
                        "distance": hit.distance,
                        "score": hit.score,
                    }
                    if output_fields:
                        hit_dict["entity"] = hit.entity.to_dict()
                    hit_list.append(hit_dict)
                formatted_results.append(hit_list)
            
            logger.info(f"Searched '{collection_name}' with {len(query_vectors)} query vectors")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search in '{collection_name}': {str(e)}")
            raise
    
    async def query(
        self,
        collection_name: str,
        expr: str,
        output_fields: Optional[List[str]] = None,
        partition_names: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query entities by expression.
        
        Args:
            collection_name: Name of the collection
            expr: Boolean expression for filtering
            output_fields: Fields to return
            partition_names: Partitions to query
            limit: Maximum number of results
            
        Returns:
            List of matching entities
        """
        try:
            collection = self.get_collection(collection_name)
            
            # Load collection if not loaded
            if not self.is_collection_loaded(collection_name):
                collection.load()
            
            results = collection.query(
                expr=expr,
                output_fields=output_fields,
                partition_names=partition_names,
                limit=limit,
            )
            
            logger.info(f"Queried '{collection_name}' with expression: {expr}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to query '{collection_name}': {str(e)}")
            raise
    
    # ========== Collection State Management ==========
    
    async def load_collection(
        self,
        collection_name: str,
        partition_names: Optional[List[str]] = None,
    ) -> None:
        """
        Load collection into memory.
        
        Args:
            collection_name: Name of the collection
            partition_names: Optional list of partitions to load
        """
        try:
            collection = self.get_collection(collection_name)
            if partition_names:
                collection.load(partition_names=partition_names)
            else:
                collection.load()
            logger.info(f"Loaded collection '{collection_name}' into memory")
        except Exception as e:
            logger.error(f"Failed to load collection '{collection_name}': {str(e)}")
            raise
    
    async def release_collection(
        self,
        collection_name: str,
        partition_names: Optional[List[str]] = None,
    ) -> None:
        """
        Release collection from memory.
        
        Args:
            collection_name: Name of the collection
            partition_names: Optional list of partitions to release
        """
        try:
            collection = self.get_collection(collection_name)
            if partition_names:
                collection.release(partition_names=partition_names)
            else:
                collection.release()
            logger.info(f"Released collection '{collection_name}' from memory")
        except Exception as e:
            logger.error(f"Failed to release collection '{collection_name}': {str(e)}")
            raise
    
    def get_load_state(self, collection_name: str) -> str:
        """
        Get collection load state.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Load state string
        """
        try:
            state = utility.load_state(collection_name, using=self.alias)
            return str(state)
        except Exception:
            collection = self.get_collection(collection_name)
            load_attr = getattr(collection, "load_state", None)
            return str(load_attr) if load_attr is not None else "unknown"

    def is_collection_loaded(self, collection_name: str) -> bool:
        """Return True when a collection is loaded in memory."""
        try:
            state = utility.load_state(collection_name, using=self.alias)
            return state == LoadState.Loaded
        except Exception:
            try:
                collection = self.get_collection(collection_name)
            except Exception:
                return False
            has_load_attr = getattr(collection, "has_load", None)
            if callable(has_load_attr):
                try:
                    return bool(has_load_attr())
                except Exception:
                    pass
            load_attr = getattr(collection, "load_state", None)
            if load_attr is not None:
                return str(load_attr) in {"LoadState.Loaded", "Loaded"}
            return False
    
    # ========== Partition Management ==========
    
    async def create_partition(
        self,
        collection_name: str,
        partition_name: str,
        description: str = "",
    ) -> None:
        """
        Create a partition in a collection.
        
        Args:
            collection_name: Name of the collection
            partition_name: Name of the partition
            description: Partition description
        """
        try:
            collection = self.get_collection(collection_name)
            collection.create_partition(
                partition_name=partition_name,
                description=description
            )
            logger.info(f"Created partition '{partition_name}' in '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to create partition: {str(e)}")
            raise
    
    async def drop_partition(self, collection_name: str, partition_name: str) -> None:
        """
        Drop a partition from a collection.
        
        Args:
            collection_name: Name of the collection
            partition_name: Name of the partition to drop
        """
        try:
            collection = self.get_collection(collection_name)
            collection.drop_partition(partition_name=partition_name)
            logger.info(f"Dropped partition '{partition_name}' from '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to drop partition: {str(e)}")
            raise
    
    def list_partitions(self, collection_name: str) -> List[str]:
        """
        List all partitions in a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            List of partition names
        """
        collection = self.get_collection(collection_name)
        return [p.name for p in collection.partitions]
    
    # ========== Statistics and Info ==========
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary containing collection statistics
        """
        collection = self.get_collection(collection_name)
        return {
            "row_count": collection.num_entities,
            "schema": str(collection.schema),
            "partitions": len(collection.partitions),
            "load_state": self.get_load_state(collection_name),
        }
    
    def get_entity_count(
        self,
        collection_name: str,
        partition_names: Optional[List[str]] = None,
    ) -> int:
        """
        Get number of entities in collection.
        
        Args:
            collection_name: Name of the collection
            partition_names: Optional list of partitions
            
        Returns:
            Number of entities
        """
        collection = self.get_collection(collection_name)
        if partition_names:
            count = 0
            for partition_name in partition_names:
                partition = collection.partition(partition_name)
                count += partition.num_entities
            return count
        return collection.num_entities
    
    async def flush(self, collection_names: Optional[List[str]] = None) -> None:
        """
        Flush data to persistent storage.
        
        Args:
            collection_names: Optional list of collection names to flush
        """
        try:
            if collection_names:
                for name in collection_names:
                    collection = self.get_collection(name)
                    collection.flush()
            else:
                utility.flush(using=self.alias)
            logger.info("Flushed data to storage")
        except Exception as e:
            logger.error(f"Failed to flush data: {str(e)}")
            raise
    
    async def compact(self, collection_name: str) -> None:
        """
        Compact a collection to improve performance.
        
        Args:
            collection_name: Name of the collection
        """
        try:
            collection = self.get_collection(collection_name)
            collection.compact()
            logger.info(f"Compacted collection '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to compact collection '{collection_name}': {str(e)}")
            raise
    
    # ========== Context Manager Support ==========
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.disconnect()
        return False
