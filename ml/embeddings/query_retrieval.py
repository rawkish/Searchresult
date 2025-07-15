import os
import numpy as np
import sqlite3
import json
import faiss
from typing import List, Dict, Tuple, Any, Optional
from sentence_transformers import SentenceTransformer

class QueryEmbeddingRetrieval:
    """
    Module for embedding search queries and retrieving candidate products using ANN index.
    This handles the first stage of the search process: retrieval of candidate products.
    """
    def __init__(
        self, 
        db_path: str = r'data\ecommerce.db',
        index_dir: str = r'models\ann_index',
        embedding_model_name: str = 'all-MiniLM-L6-v2'
    ):
        """
        Args:
            db_path: Path to SQLite database
            index_dir: Directory containing the ANN index
            embedding_model_name: Name of the SentenceTransformer model to use
        """
        self.db_path = db_path
        self.index_dir = index_dir
        self.embedding_model_name = embedding_model_name
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self._load_index()
    
    def _load_index(self) -> None:
        try:
            product_ids_path = os.path.join(self.index_dir, "product_ids.npy")
            self.product_ids = np.load(product_ids_path)
            index_path = os.path.join(self.index_dir, "product_index.faiss")
            self.index = faiss.read_index(index_path)
            print(f"Loaded ANN index with {len(self.product_ids)} products")
        except Exception as e:
            print(f"Error loading ANN index: {e}")
            self.index = None
            self.product_ids = None
            raise ValueError("Failed to load ANN index. Please ensure it has been built.")
    
    def embed_query(self, query_text: str) -> np.ndarray:
        query_text = query_text.lower().strip()
        query_embedding = self.embedding_model.encode([query_text])[0]
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        return query_embedding
    
    def retrieve_candidates(
        self, 
        query_embedding: np.ndarray, 
        k: int = 100,
        nprobe: int = 16  
    ) -> Tuple[List[int], List[float]]:
        
        if self.index is None:
            raise ValueError("ANN index not loaded")
        
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe
        
        distances, indices = self.index.search(query_embedding, k)
        
        candidate_product_ids = [int(self.product_ids[idx]) for idx in indices[0] if idx >= 0 and idx < len(self.product_ids)]
        candidate_distances = [float(dist) for dist in distances[0]][:len(candidate_product_ids)]
        
        return candidate_product_ids, candidate_distances
    
    def get_product_details(self, product_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Args:
            product_ids: List of product IDs to retrieve
        Returns:
            List of product details dictionaries
        """
        if not product_ids:
            return []
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        placeholders = ','.join(['?' for _ in product_ids])
        cursor.execute(f"""
            SELECT p.id, p.name, p.description, p.category, p.price, p.features,
                   COALESCE(i.quantity, 0) as inventory_count,
                   COALESCE(pp.score, 0) as popularity_score
            FROM products p
            LEFT JOIN inventory i ON p.id = i.product_id
            LEFT JOIN product_popularity pp ON p.id = pp.product_id
            WHERE p.id IN ({placeholders})
        """, product_ids)
        products = cursor.fetchall()
        conn.close()
        results = []
        for product in products:
            product_id, name, description, category, price, features_json, inventory_count, popularity_score = product
            try:
                features = json.loads(features_json)
            except:
                features = {}
            results.append({
                "product_id": product_id,
                "name": name,
                "description": description,
                "category": category,
                "price": price,
                "features": features,
                "inventory_count": inventory_count,
                "popularity_score": popularity_score
            })
        return results
    
    def search(
        self, 
        query_text: str, 
        k: int = 100,
        filter_func: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        embed query, retrieve candidates, and get product details.
        Args:
            query_text: The search query text
            k: Number of candidates to retrieve
            filter_func: Optional function to filter products (takes product dict, returns bool)            
        Returns:
            List of product details for the candidates
        """
        query_embedding = self.embed_query(query_text)
        candidate_ids, distances = self.retrieve_candidates(query_embedding, k=k)
        candidates = self.get_product_details(candidate_ids)
        id_to_distance = {pid: dist for pid, dist in zip(candidate_ids, distances)}
        for product in candidates:
            product["distance_score"] = id_to_distance.get(product["product_id"], float('inf'))
        if filter_func is not None:
            candidates = [product for product in candidates if filter_func(product)]
        return candidates


# Example 
if __name__ == "__main__":
    retriever = QueryEmbeddingRetrieval()
    query = "Apple Laptop with good battery"

    def filter_in_stock(product):
        return product["inventory_count"] > 0
    
    candidates = retriever.search(query, k=20, filter_func=filter_in_stock)
    print(f"Found {len(candidates)} candidates for query: '{query}'")

    for i, product in enumerate(candidates[:10]):  # Show top 5
        print(f"{i+1}. {product['name']} (ID: {product['product_id']}, Distance: {product['distance_score']:.4f})")
        print(f"   Category: {product['category']}")

        print(f"   Price: ${product['price']:.2f}")
        print(f"   Popularity: {product['popularity_score']:.2f}")
        
        print()