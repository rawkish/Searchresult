import sqlite3
import pickle
import json

class FeatureStoreBuilder:
    def __init__(self, db_path=r'data\ecommerce.db', feature_store_path=r'data\feature_store.pkl'):
        self.db_path = db_path
        self.feature_store_path = feature_store_path
    
    def build_feature_store(self):
        """Build feature store with product, inventory, and popularity data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Get product info
        cursor.execute("""
            SELECT p.id, p.name, p.description, p.category, p.price, p.features,
                   i.quantity as inventory_count, 
                   COALESCE(pp.score, 0) as popularity_score,
                   (SELECT COUNT(*) FROM orders WHERE product_id = p.id) as order_count
            FROM products p
            LEFT JOIN inventory i ON p.id = i.product_id
            LEFT JOIN product_popularity pp ON p.id = pp.product_id
        """)
        products_data = cursor.fetchall()
        feature_store = {}
        
        for product in products_data:
            product_id, name, description, category, price, features_json, inventory_count, popularity_score, order_count = product
            features = json.loads(features_json)
            # Store all features for this product
            feature_store[product_id] = {
                "product_id": product_id,
                "name": name,
                "description": description,
                "category": category,
                "price": price,
                "features": features,
                "inventory_count": inventory_count if inventory_count else 0,
                "popularity_score": popularity_score,
                "order_count": order_count,
                # Add more derived features
                "category_depth": len(category.split('>')),
                "is_in_stock": 1 if (inventory_count if inventory_count else 0) > 0 else 0,
                "has_description": 1 if len(description) > 10 else 0,
                "description_length": len(description),
                "price_bucket": self._get_price_bucket(price)
            }
        conn.close()
        # Save feature store
        with open(self.feature_store_path, 'wb') as f:
            pickle.dump(feature_store, f)
        print(f"Built feature store with {len(feature_store)} products")
        return feature_store
    
    def _get_price_bucket(self, price):
        """Get price bucket (0-4) for given price"""
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
    
    def load_feature_store(self):
        """Load existing feature store or build new one"""
        try:
            with open(self.feature_store_path, 'rb') as f:
                feature_store = pickle.load(f)
            print(f"Loaded feature store with {len(feature_store)} products")
            return feature_store
        except:
            print("Failed to load feature store, building new one...")
            return self.build_feature_store()