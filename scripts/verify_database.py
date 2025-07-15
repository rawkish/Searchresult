import sqlite3
import numpy as np
import json
from collections import Counter

def verify_database():
    # Connect to the database
    conn = sqlite3.connect(r'data\ecommerce.db')
    cursor = conn.cursor()
    
    # Check table counts
    tables = ['products', 'users', 'orders', 'inventory', 'product_embeddings', 'product_popularity']
    print("======== Database Statistics ========")
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"{table}: {count} records")
    
    # Sample products
    print("\n======== Sample Products ========")
    cursor.execute("SELECT id, name, category, price FROM products LIMIT 5")
    products = cursor.fetchall()
    for product in products:
        print(f"ID: {product[0]}, Name: {product[1]}, Category: {product[2]}, Price: ${product[3]:.2f}")
    
    # Check category distribution
    print("\n======== Category Distribution ========")
    cursor.execute("SELECT SUBSTR(category, 1, INSTR(category, '>')-2) as main_category, COUNT(*) FROM products GROUP BY main_category")
    categories = cursor.fetchall()
    for category in categories:
        print(f"{category[0]}: {category[1]} products")
    
    # Check order distribution over time
    print("\n======== Order Timeline ========")
    cursor.execute("SELECT strftime('%Y-%m', timestamp) as month, COUNT(*) FROM orders GROUP BY month ORDER BY month")
    months = cursor.fetchall()
    for month in months:
        print(f"{month[0]}: {month[1]} orders")
    
    # Popular products
    print("\n======== Most Popular Products ========")
    cursor.execute("""
        SELECT p.id, p.name, pp.score 
        FROM product_popularity pp
        JOIN products p ON p.id = pp.product_id
        ORDER BY pp.score DESC
        LIMIT 5
    """)
    popular = cursor.fetchall()
    for prod in popular:
        print(f"ID: {prod[0]}, Name: {prod[1]}, Popularity Score: {prod[2]:.4f}")
    
    # Check inventory levels
    print("\n======== Inventory Levels ========")
    cursor.execute("""
        SELECT 
            CASE 
                WHEN quantity = 0 THEN 'Out of stock'
                WHEN quantity < 10 THEN 'Low stock'
                WHEN quantity < 30 THEN 'Medium stock'
                ELSE 'High stock'
            END as stock_level,
            COUNT(*)
        FROM inventory
        GROUP BY stock_level
    """)
    stock_levels = cursor.fetchall()
    for level in stock_levels:
        print(f"{level[0]}: {level[1]} products")
    
    # Verify product embeddings
    print("\n======== Embeddings Verification ========")
    cursor.execute("SELECT product_id, embedding FROM product_embeddings LIMIT 1")
    embedding_data = cursor.fetchone()
    if embedding_data:
        product_id, embedding_bytes = embedding_data
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        print(f"Product ID: {product_id}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding sample: {embedding[:5]}...")
        print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")  # Should be close to 1 if normalized
    
    # Sample product features
    print("\n======== Sample Product Features ========")
    cursor.execute("SELECT id, name, features FROM products LIMIT 3")
    for prod in cursor.fetchall():
        id, name, features_json = prod
        features = json.loads(features_json)
        print(f"Product: {name} (ID: {id})")
        for key, value in features.items():
            print(f"  - {key}: {value}")
    
    # Close connection
    conn.close()

if __name__ == "__main__":
    verify_database()