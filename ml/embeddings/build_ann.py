
import os
import sqlite3
import numpy as np
import faiss

class ANNIndexBuilder:
    def __init__(self, db_path=r'data\ecommerce.db', index_dir=r'models\ann_index'):
        self.db_path = db_path
        self.index_dir = index_dir
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
        self.embedding_dim = 384  
        
    def load_embeddings_from_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT product_id, embedding FROM product_embeddings")
        results = cursor.fetchall()
        product_ids = []
        embeddings = []
        
        for product_id, embedding_bytes in results:
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            product_ids.append(product_id)
            embeddings.append(embedding)
        
        conn.close()
        return np.array(product_ids), np.array(embeddings)
    
    def build_index(self):
        product_ids, embeddings = self.load_embeddings_from_db()
        if len(embeddings) == 0:
            print("No embeddings found in database.")
            return None
        index = faiss.IndexFlatL2(self.embedding_dim)
        index.add(embeddings.astype(np.float32))
        np.save(f"{self.index_dir}/product_ids.npy", product_ids)
        faiss.write_index(index, f"{self.index_dir}/product_index.faiss")
        print(f"Built ANN index with {len(product_ids)} products")
        
        return index, product_ids
    
    def load_index(self):
        try:
            index = faiss.read_index(f"{self.index_dir}/product_index.faiss")
            product_ids = np.load(f"{self.index_dir}/product_ids.npy")
            print(f"Loaded ANN index with {len(product_ids)} products")
            return index, product_ids
        except:
            print("New index was built")
            return self.build_index()
        

if __name__ == "__main__":
    search = ANNIndexBuilder()
    print(search.load_index())