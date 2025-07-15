import sqlite3
import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

class ProductSimilaritySearch:
    def __init__(self, db_path=r'data\ecommerce.db'):
        self.db_path = db_path
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def search_by_text(self, query_text, top_k=10):
        """Search products by text query using embeddings"""
        # Generate embedding for the query text
        query_embedding = self.embedding_model.encode(query_text)
        
        # Find similar products using the query embedding
        similar_products = self.find_similar_products_by_embedding(query_embedding, top_k)
        
        return similar_products
    
    def search_by_product_id(self, product_id, top_k=10):
        """Find products similar to a given product ID"""
        # Retrieve the product embedding
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT embedding FROM product_embeddings WHERE product_id = ?", (product_id,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return []
        
        # Convert bytes to numpy array
        product_embedding = np.frombuffer(result[0], dtype=np.float32)
        
        # Find similar products
        similar_products = self.find_similar_products_by_embedding(product_embedding, top_k)
        
        conn.close()
        return similar_products
    
    def find_similar_products_by_embedding(self, query_embedding, top_k=10):
        """Find products similar to the given embedding"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all product embeddings
        cursor.execute("""
            SELECT p.id, p.name, p.description, p.category, p.price, pe.embedding
            FROM products p
            JOIN product_embeddings pe ON p.id = pe.product_id
        """)
        
        products = cursor.fetchall()
        
        # Calculate similarity and rank products
        similarities = []
        for product in products:
            product_id, name, description, category, price, embedding_bytes = product
            
            # Convert embedding bytes to numpy array
            product_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            
            # Calculate cosine similarity (1 - cosine distance)
            similarity = 1 - cosine(query_embedding, product_embedding)
            
            similarities.append({
                'product_id': product_id,
                'name': name,
                'description': description,
                'category': category,
                'price': price,
                'similarity_score': similarity
            })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Return top_k results
        conn.close()
        return similarities[:top_k]
    
    def search_demo(self):
        # """Run a demo of different search types"""
        # print("===== Product Similarity Search Demo =====")
        
        # # Text search examples
        # print("\n--- Search by Text Query ---")
        # text_queries = [
        #     "wireless noise cancelling headphones",
        #     "comfortable running shoes",
        #     "organic skincare products for sensitive skin",
        #     "latest smartphone with good camera"
        # ]
        
        # for query in text_queries:
        #     print(f"\nQuery: '{query}'")
        #     results = self.search_by_text(query, top_k=3)
        #     for i, result in enumerate(results):
        #         print(f"{i+1}. {result['name']} (${result['price']:.2f}) - {result['category']}")
        #         print(f"   Score: {result['similarity_score']:.4f}")
        
        # # Product-based search example
        # print("\n--- Search by Similar Product ---")
        # # First get a random product ID
        # conn = sqlite3.connect(self.db_path)
        # cursor = conn.cursor()
        # cursor.execute("SELECT id, name, category FROM products ORDER BY RANDOM() LIMIT 1")
        # seed_product = cursor.fetchone()
        # conn.close()
        
        # if seed_product:
        #     seed_id, seed_name, seed_category = seed_product
        #     print(f"Seed Product: {seed_name} (ID: {seed_id}, Category: {seed_category})")
            
        #     results = self.search_by_product_id(seed_id, top_k=5)
        #     print("\nSimilar Products:")
        #     for i, result in enumerate(results):
        #         if result['product_id'] == seed_id:  # Skip the seed product itself
        #             continue
        #         print(f"{i+1}. {result['name']} (${result['price']:.2f}) - {result['category']}")
        #         print(f"   Score: {result['similarity_score']:.4f}")
        
        # # Direct embedding search example
        print("\n--- Search by Custom Embedding ---")
        print("Creating a custom embedding for 'outdoor camping gear'...")
        
        # Generate a custom embedding
        custom_embedding = self.embedding_model.encode("waterproof headphones")
        
        # Find similar products using the custom embedding
        results = self.find_similar_products_by_embedding(custom_embedding, top_k=5)
        
        print("\nProducts similar to 'outdoor camping gear':")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['name']} (${result['price']:.2f}) - {result['category']}")
            print(f"   Score: {result['similarity_score']:.4f}")
        
        # Advanced example: Combining embeddings
        print("\n--- Search with Combined Embeddings ---")
        print("Creating a combined embedding for 'affordable luxury watches'...")
        
        # Generate embeddings for different aspects
        affordable_embedding = self.embedding_model.encode("affordable")
        luxury_embedding = self.embedding_model.encode("luxury")
        watch_embedding = self.embedding_model.encode("watches")
        
        # Combine embeddings with different weights
        combined_embedding = (0.3 * affordable_embedding + 0.3 * luxury_embedding + 0.4 * watch_embedding)
        # Normalize the combined embedding
        combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
        
        # Find similar products using the combined embedding
        results = self.find_similar_products_by_embedding(combined_embedding, top_k=5)
        
        print("\nProducts similar to 'affordable luxury watches':")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['name']} (${result['price']:.2f}) - {result['category']}")
            print(f"   Score: {result['similarity_score']:.4f}")

if __name__ == "__main__":
    search = ProductSimilaritySearch()
    search.search_demo()