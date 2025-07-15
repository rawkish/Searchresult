import json
import pickle
import sqlite3
import numpy as np
from datetime import datetime
from typing import List, Optional

import faiss
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ml.embeddings.embeddings_generator import ProductEmbeddingGenerator
from ml.embeddings.build_ann import ANNIndexBuilder
from ml.training.feature_store import FeatureStoreBuilder
from ml.training.rank_candidates import RankingModelTrainer

from contextlib import asynccontextmanager

DB_PATH = r"data/ecommerce.db"
MODEL_DIR = r"models"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

embedding_model = None
ann_index = None
product_ids = None
feature_store = None
ranking_model = None

class SearchResult(BaseModel):
    product_id: int
    name: str
    description: str
    category: str
    price: float
    popularity_score: float
    score: float
    features: dict
    inventory_count: int

class OrderRequest(BaseModel):
    user_id: int
    product_id: int
    quantity: int = 1

class OrderResponse(BaseModel):
    order_id: int
    product_id: int
    timestamp: str
    message: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, ann_index, product_ids, feature_store, ranking_model
    
    print("Loading models and indexes...")
    
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    ann_builder = ANNIndexBuilder(db_path=DB_PATH)
    ann_index, product_ids = ann_builder.load_index()
    
    feature_store_builder = FeatureStoreBuilder(db_path=DB_PATH)
    feature_store = feature_store_builder.load_feature_store()
    
    ranking_trainer = RankingModelTrainer(db_path=DB_PATH)
    ranking_model = ranking_trainer.load_model()
    
    print("All components loaded successfully!")
    
    yield  # This is where FastAPI serves requests
    
    print("Shutting down and cleaning up resources...")


app = FastAPI(title="E-Commerce Search & Recommendation API",lifespan=lifespan)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)


def get_product_details(product_ids):
    """Get product details from database"""
    conn = sqlite3.connect(DB_PATH)
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
        
        results.append({
            "product_id": product_id,
            "name": name,
            "description": description,
            "category": category,
            "price": price,
            "features": json.loads(features_json),
            "inventory_count": inventory_count,
            "popularity_score": popularity_score
        })
    
    return results

def update_popularity_score(product_id, increment=1.0):
    """Update popularity score for a product"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if product exists in popularity table
    cursor.execute("SELECT score FROM product_popularity WHERE product_id = ?", (product_id,))
    result = cursor.fetchone()
    
    timestamp = datetime.now().isoformat()
    increment = 0.001 * increment
    if result:
        current_score = result[0]
        new_score = current_score + increment
        cursor.execute(
            "UPDATE product_popularity SET score = ?, last_updated = ? WHERE product_id = ?",
            (new_score, timestamp, product_id)
        )
    else:
        cursor.execute(
            "INSERT INTO product_popularity (product_id, score, last_updated) VALUES (?, ?, ?)",
            (product_id, increment, timestamp)
        )
    
    conn.commit()
    conn.close()

def get_feature_vector(product_data):
    """Convert product data to feature vector for ranking"""
    return [
        product_data["price"],
        product_data["inventory_count"],
        product_data["popularity_score"],
        len(product_data["category"].split('>')),  
        1 if product_data["inventory_count"] > 0 else 0,  
        1 if len(product_data["description"]) > 10 else 0,  
        len(product_data["description"]),  
        get_price_bucket(product_data["price"]) 
    ]

def get_price_bucket(price):
    """Helper function for price bucket"""
    if price < 10:
        return 0
    elif price < 50:
        return 1
    elif price < 100:
        return 2
    elif price < 500:
        return 3
    else:
        return 4

@app.get("/api/query", response_model=List[SearchResult])
async def search_products(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Number of results to return"),
    category: Optional[str] = Query(None, description="Filter by category"),
    min_price: Optional[float] = Query(None, description="Minimum price"),
    max_price: Optional[float] = Query(None, description="Maximum price")
):
    """Search products by keyword and rank results"""
    if not q:
        raise HTTPException(status_code=400, detail="Query parameter 'q' cannot be empty")
    
    query_embedding = embedding_model.encode([q])[0].astype(np.float32).reshape(1, -1)
    
    k = limit * 5  # will take more candidates for filtering
    distances, indices = ann_index.search(query_embedding, k)

    candidate_product_ids = [int(product_ids[idx]) for idx in indices[0]]
    
    products = get_product_details(candidate_product_ids)
    
    filtered_products = []
    for product in products:
        if product["inventory_count"] <= 0:
            continue
            
        if category and not product["category"].startswith(category):
            continue
            
        if min_price is not None and product["price"] < min_price:
            continue
        if max_price is not None and product["price"] > max_price:
            continue
            
        filtered_products.append(product)
    
    if ranking_model and filtered_products:
        feature_vectors = [get_feature_vector(p) for p in filtered_products]
        scores = ranking_model.predict(feature_vectors)
        
        for i, score in enumerate(scores):
            filtered_products[i]["score"] = float(score)
        
        filtered_products.sort(key=lambda p: p["score"], reverse=True)
    else:
        # Use popularity score
        for product in filtered_products:
            product["score"] = product["popularity_score"]
        filtered_products.sort(key=lambda p: p["popularity_score"], reverse=True)
    
    
    results = filtered_products[:limit]
    
    return results

@app.post("/api/buy", response_model=OrderResponse)
async def buy_product(order: OrderRequest):
    """Simulate a purchase and update product popularity"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM products WHERE id = ?", (order.product_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Product not found")
    
    cursor.execute("SELECT quantity FROM inventory WHERE product_id = ?", (order.product_id,))
    inventory = cursor.fetchone()
    
    if not inventory or inventory[0] < order.quantity:
        conn.close()
        raise HTTPException(status_code=400, detail="Insufficient inventory")
    
    # Create order
    timestamp = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO orders (user_id, product_id, timestamp, quantity) VALUES (?, ?, ?, ?)",
        (order.user_id, order.product_id, timestamp, order.quantity)
    )
    order_id = cursor.lastrowid
    
    # Update inventory
    new_quantity = inventory[0] - order.quantity
    cursor.execute(
        "UPDATE inventory SET quantity = ?, last_updated = ? WHERE product_id = ?",
        (new_quantity, timestamp, order.product_id)
    )
    
    conn.commit()
    conn.close()
    
    
    update_popularity_score(order.product_id, increment=order.quantity)
    
    return {
        "order_id": order_id,
        "product_id": order.product_id,
        "timestamp": timestamp,
        "message": f"Order placed successfully for {order.quantity} item(s)"
    }

@app.get("/api/rebuild", status_code=200)
async def rebuild_system():
    """Rebuild embeddings, indexes and models (admin endpoint)"""
    global embedding_model, ann_index, product_ids, feature_store, ranking_model
    
    try:
        # Rebuild embeddings
        generator = ProductEmbeddingGenerator(db_path=DB_PATH)
        generator.run_pipeline(method='transformer')
        
        # Rebuild ANN index
        ann_builder = ANNIndexBuilder(db_path=DB_PATH)
        ann_index, product_ids = ann_builder.build_index()
        
        # Rebuild feature store
        feature_store_builder = FeatureStoreBuilder(db_path=DB_PATH)
        feature_store = feature_store_builder.build_feature_store()
        
        # Retrain ranking model
        ranking_trainer = RankingModelTrainer(db_path=DB_PATH)
        ranking_model = ranking_trainer.train_model()
        
        return {"message": "System rebuilt successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rebuilding system: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)