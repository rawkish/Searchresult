import sqlite3
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pickle
import os

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

class ProductEmbeddingGenerator:
    def __init__(self, db_path=r'data\ecommerce.db', embedding_dim=128, model_dir=r'models\embeddings'):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.model_dir = model_dir
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        print("Loading SentenceTransformer model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model for embeddings
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english'
        )
        self.svd = TruncatedSVD(n_components=self.embedding_dim)
    
    def preprocess_text(self, text):
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)
    
    def extract_product_text(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, description, category, features FROM products")
        products = cursor.fetchall()
        product_texts = {}
        combined_texts = []
        
        for product in products:
            product_id, name, description, category, features_json = product
            features = json.loads(features_json)
            feature_text = ' '.join(f"{k} {v}" for k, v in features.items())
            category_parts = category.split('>')
            category_text = ' '.join(part.strip() for part in category_parts)
            combined_text = f"{name} {name} {category_text} {category_text} {description} {feature_text}"
            preprocessed_text = self.preprocess_text(combined_text)
            product_texts[product_id] = preprocessed_text
            combined_texts.append(preprocessed_text)
        conn.close()
        return product_texts, combined_texts
    
    def generate_embeddings_with_transformer(self, product_texts):
        """Using Sentence Transformer"""
        product_ids = list(product_texts.keys())
        texts = list(product_texts.values())
        
        print(f"Generating embeddings for {len(texts)} products...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        embedding_dict = {product_id: embedding for product_id, embedding in zip(product_ids, embeddings)}
        
        return embedding_dict
    
    def generate_embeddings_with_tfidf_svd(self, product_texts, combined_texts):
        """Using TF-IDF + SVD"""
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined_texts)
        self.svd.fit(tfidf_matrix)
        product_ids = list(product_texts.keys())
        embedding_dict = {}
        
        for product_id in product_ids:
            # Transform the product text to TF-IDF
            product_tfidf = self.tfidf_vectorizer.transform([product_texts[product_id]])
            # Apply SVD to get the embedding
            embedding = self.svd.transform(product_tfidf)[0]
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            embedding_dict[product_id] = embedding
        return embedding_dict
    
    def save_embeddings_to_db(self, embedding_dict, method='transformer'):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM product_embeddings")
        for product_id, embedding in embedding_dict.items():
            embedding_bytes = embedding.astype(np.float32).tobytes()
            cursor.execute(
                "INSERT INTO product_embeddings (product_id, embedding) VALUES (?, ?)",
                (product_id, embedding_bytes)
            )
        conn.commit()
        conn.close()
        print(f"Saved {len(embedding_dict)} {method} embeddings to database")
    
    def save_models(self):
        with open(f"{self.model_dir}/tfidf_vectorizer.pkl", 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
            
        with open(f"{self.model_dir}/svd_model.pkl", 'wb') as f:
            pickle.dump(self.svd, f)
        print(f"Models saved to {self.model_dir}")
    
    def run_pipeline(self, method='transformer'):
        print(f"Starting embedding generation pipeline using {method} method...")
        
        product_texts, combined_texts = self.extract_product_text()
        print(f"Extracted text features for {len(product_texts)} products")
        
        if method == 'transformer':
            embedding_dict = self.generate_embeddings_with_transformer(product_texts)
        elif method == 'tfidf_svd':
            embedding_dict = self.generate_embeddings_with_tfidf_svd(product_texts, combined_texts)
        else:
            raise ValueError(f"Unknown embedding method: {method}")
        self.save_embeddings_to_db(embedding_dict, method)
        self.save_models()
        print("Embedding generation pipeline completed successfully!")
        
        return embedding_dict

# Example usage
if __name__ == "__main__":
    generator = ProductEmbeddingGenerator()
    embedding_dict = generator.run_pipeline(method='transformer')
    