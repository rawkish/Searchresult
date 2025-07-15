import os
from feature_store import FeatureStoreBuilder
import sqlite3
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import numpy as np

class RankingModelTrainer:
    def __init__(self, db_path=r'data\ecommerce.db', model_dir=r'models\ranking'):
        self.db_path = db_path
        self.model_dir = model_dir
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        self.feature_store_builder = FeatureStoreBuilder(db_path)
    
    def prepare_training_data(self):
        """Prepare training data for ranking model"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT o.product_id, COUNT(*) as order_count,
                   p.price, p.category,
                   COALESCE(i.quantity, 0) as inventory_count,
                   COALESCE(pp.score, 0) as popularity_score
            FROM orders o
            JOIN products p ON o.product_id = p.id
            LEFT JOIN inventory i ON p.id = i.product_id
            LEFT JOIN product_popularity pp ON p.id = pp.product_id
            GROUP BY o.product_id
        """)
        
        order_data = cursor.fetchall()
        
        feature_store = self.feature_store_builder.load_feature_store()
        
        X = []
        y = []
        
        for row in order_data:
            product_id, order_count, price, category, inventory_count, popularity_score = row
            
            if product_id in feature_store:
                features = feature_store[product_id]
                
                feature_vector = [
                    features["price"],
                    features["inventory_count"],
                    features["popularity_score"],
                    features["category_depth"],
                    features["is_in_stock"],
                    features["has_description"],
                    features["description_length"],
                    features["price_bucket"]
                ]
                
                X.append(feature_vector)
                y.append(order_count)  # Use order count as the target variable
        
        conn.close()
        
        return np.array(X), np.array(y)
    
    def train_model(self):
        """Train ranking model on order data"""
        X, y = self.prepare_training_data()
        
        if len(X) == 0:
            print("No training data available.")
            return None
        
        # Training a basic ranking model using GradientBoostingRegressor
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3
        )
        
        model.fit(X, y)
        
        with open(f"{self.model_dir}/ranking_model.pkl", 'wb') as f:
            pickle.dump(model, f)
        
        print(f"Trained ranking model with {len(X)} examples")
        
        return model
    
    def test_model(self, test_size=0.2, random_state=42):
        """Test the ranking model with train/test split"""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        
        X, y = self.prepare_training_data()
        
        if len(X) == 0:
            print("No data available for testing.")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Test Results:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        feature_names = [
            "price", "inventory_count", "popularity_score", 
            "category_depth", "is_in_stock", "has_description", 
            "description_length", "price_bucket"
        ]
        importances = model.feature_importances_
        
        print("\nFeature Importance:")
        for feature, importance in sorted(zip(feature_names, importances), 
                                        key=lambda x: x[1], reverse=True):
            print(f"{feature}: {importance:.4f}")
        
        return model, mse, r2


    def load_model(self):
        """Load existing ranking model or train new one"""
        try:
            with open(f"{self.model_dir}/ranking_model.pkl", 'rb') as f:
                model = pickle.load(f)
            print("Loaded ranking model")
            return model
        except:
            print("Failed to load ranking model, training new one...")
            return self.train_model()
        

if __name__ == "__main__":
    model = RankingModelTrainer()
    model.load_model()