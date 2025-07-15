import sqlite3
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cosine
from mplcursors import cursor

def test_embeddings(db_path=r'data\ecommerce.db'):
    """Test and visualize the product embeddings"""
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Retrieve product data along with embeddings
    cursor.execute("""
        SELECT p.id, p.name, p.category, pe.embedding 
        FROM products p
        JOIN product_embeddings pe ON p.id = pe.product_id
        LIMIT 1000  -- Limit for visualization purposes
    """)
    
    products = []
    embeddings = []
    product_ids = []
    categories = []
    names = []
    
    for row in cursor.fetchall():
        product_id, name, category, embedding_bytes = row
        
        # Convert embedding from bytes to numpy array
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        
        # Extract main category (before the ">" symbol)
        main_category = category.split('>')[0].strip()
        
        product_ids.append(product_id)
        names.append(name)
        categories.append(main_category)
        embeddings.append(embedding)
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings)
    
    # Check embedding dimensions
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings_array.shape[1]}")
    
    # Test similarity between products
    test_similar_products(product_ids, names, categories, embeddings_array)
    
    # Visualize embeddings with t-SNE
    visualize_embeddings(embeddings_array, categories, names)
    
    conn.close()

def test_similar_products(product_ids, names, categories, embeddings_array, n_samples=5):
    """Find and print similar products"""
    print("\n==== Testing Product Similarities ====")
    
    # Select random products as query products
    np.random.seed(42)  # For reproducibility
    query_indices = np.random.choice(len(product_ids), n_samples, replace=False)
    
    for idx in query_indices:
        query_embedding = embeddings_array[idx]
        query_id = product_ids[idx]
        query_name = names[idx]
        query_category = categories[idx]
        
        print(f"\nQuery Product: {query_name} (ID: {query_id}, Category: {query_category})")
        
        # Calculate similarity to all other products
        similarities = []
        for i in range(len(embeddings_array)):
            if i == idx:  # Skip self
                continue
                
            # Calculate cosine similarity (1 - cosine distance)
            similarity = 1 - cosine(query_embedding, embeddings_array[i])
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Print top 5 similar products
        print("Top 5 most similar products:")
        for i, sim in similarities[:5]:
            print(f"- {names[i]} (ID: {product_ids[i]}, Category: {categories[i]}, Similarity: {sim:.4f})")

def visualize_embeddings(embeddings_array, categories, names, perplexity=30):
    """Visualize embeddings using t-SNE"""
    print("\n==== Visualizing Embeddings with t-SNE ====")
    
    # Reduce to 2D using t-SNE
    print("Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_array)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'category': categories,
        'name': names
    })
    
    # Plot
    plt.figure(figsize=(12, 10))
    scatter = sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='category',
        palette='viridis',
        alpha=0.7
    )
    
    # Remove some of the clutter
    plt.title('t-SNE Visualization of Product Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save and show the plot
    plt.tight_layout()
    plt.savefig('product_embeddings_tsne.png')
    print("Visualization saved as 'product_embeddings_tsne.png'")
    
    # Add hover annotations for interactive exploration
    # Note: This only works in Jupyter notebook or interactive environment
    try:
        
        cursor(scatter).connect(
            "add", lambda sel: sel.annotation.set_text(df['name'].iloc[sel.target.index])
        )
        print("Hover over points to see product names (if in interactive environment)")
    except ImportError:
        print("Install mplcursors for interactive hover annotations")

if __name__ == "__main__":
    test_embeddings()