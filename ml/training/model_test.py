import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from sklearn.model_selection import cross_val_score
from rank_candidates import RankingModelTrainer

def run_basic_tests():
    """Run basic tests on the ranking model"""
    print("=== Running Basic Model Tests ===")
    trainer = RankingModelTrainer()
    
    # Test 1: Check if data preparation works
    print("\nTest 1: Data Preparation")
    X, y = trainer.prepare_training_data()
    print(f"Data shape: X={X.shape}, y={y.shape}")
    if len(X) > 0:
        print("✓ Data preparation successful")
    else:
        print("✗ Data preparation failed - no data found")
    
    # Test 2: Model training and evaluation
    print("\nTest 2: Model Training and Evaluation")
    model, mse, r2 = trainer.test_model(test_size=0.3)
    
    # Test 3: Cross-validation
    print("\nTest 3: Cross-validation")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"Cross-validation MSE scores: {-cv_scores}")
    print(f"Mean CV MSE: {-cv_scores.mean():.4f}")
    print(f"CV MSE Std Dev: {cv_scores.std():.4f}")
    
    return trainer, model

def test_ranking_quality():
    """Test the quality of rankings produced by the model"""
    print("\n=== Testing Ranking Quality ===")
    trainer = RankingModelTrainer()
    model = trainer.load_model()
    
    X, y = trainer.prepare_training_data()
    
    predictions = model.predict(X)
    
    conn = sqlite3.connect(trainer.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, category FROM products")
    product_categories = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    
    feature_store = trainer.feature_store_builder.load_feature_store()
    product_ids = list(feature_store.keys())
    
    category_groups = {}
    for i, product_id in enumerate(product_ids[:len(X)]):
        if product_id in product_categories:
            category = product_categories[product_id]
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append((product_id, y[i], predictions[i]))
    
    # Calculate NDCG for each category
    ndcg_scores = []
    for category, products in category_groups.items():
        if len(products) < 2:
            continue
            
        # Extract true relevance and predicted scores
        true_relevance = np.array([p[1] for p in products])
        predicted_scores = np.array([p[2] for p in products])
        
        # Reshape for ndcg_score function (which expects 2D arrays)
        true_relevance = true_relevance.reshape(1, -1)
        predicted_scores = predicted_scores.reshape(1, -1)
        
        # Calculate NDCG@k where k is the number of products in the category
        k = min(10, len(products))
        score = ndcg_score(true_relevance, predicted_scores, k=k)
        ndcg_scores.append(score)
        print(f"Category: {category}, NDCG@{k}: {score:.4f}")
    
    print(f"\nAverage NDCG: {np.mean(ndcg_scores):.4f}")

def test_model_on_new_products():
    """Test the model on new product scenarios"""
    print("\n=== Testing on New Product Scenarios ===")
    trainer = RankingModelTrainer()
    model = trainer.load_model()
    
    # Create synthetic test products with various feature combinations
    test_products = [
        # High price, high inventory, high popularity
        [299.99, 500, 9.8, 2, 1, 1, 500, 3],
        # Low price, low inventory, high popularity
        [19.99, 5, 9.5, 1, 1, 1, 300, 1],
        # High price, low inventory, low popularity
        [399.99, 2, 3.2, 3, 1, 0, 0, 3],
        # Low price, high inventory, low popularity
        [9.99, 1000, 2.5, 1, 1, 1, 100, 0],
        # Mid-range across all features
        [99.99, 100, 6.5, 2, 1, 1, 250, 2]
    ]
    
    predictions = model.predict(np.array(test_products))
    
    print("\nRanking predictions for synthetic products:")
    feature_names = [
        "price", "inventory_count", "popularity_score", 
        "category_depth", "is_in_stock", "has_description", 
        "description_length", "price_bucket"
    ]
    
    results = []
    for i, (features, pred) in enumerate(zip(test_products, predictions)):
        results.append({
            "Product": f"Synthetic {i+1}",
            "Predicted Orders": f"{pred:.2f}",
            **{name: val for name, val in zip(feature_names, features)}
        })
    
    results_df = pd.DataFrame(results)
    print(results_df)

if __name__ == "__main__":
    # Import inside the if block to avoid circular imports
    import sqlite3
    
    # Run all tests
    trainer, model = run_basic_tests()
    test_ranking_quality()
    test_model_on_new_products()