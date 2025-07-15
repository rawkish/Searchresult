import sqlite3
import random
import datetime
import uuid
import json
import numpy as np
from faker import Faker
from collections import defaultdict

fake = Faker()

conn = sqlite3.connect(r'data\ecommerce.db')
cursor = conn.cursor()

def setup_database():
    # Product table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT NOT NULL,
        category TEXT NOT NULL,
        price REAL NOT NULL,
        features TEXT NOT NULL,  -- JSON string of features
        -- image_url TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Orders table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        quantity INTEGER NOT NULL,
        FOREIGN KEY (product_id) REFERENCES products (id)
    )
    ''')
    
    # Inventory table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS inventory (
        product_id INTEGER PRIMARY KEY,
        quantity INTEGER NOT NULL,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (product_id) REFERENCES products (id)
    )
    ''')
    
    # Product embeddings table for content-based filtering
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS product_embeddings (
        product_id INTEGER PRIMARY KEY,
        embedding BLOB NOT NULL,  -- Store embeddings as binary blob
        FOREIGN KEY (product_id) REFERENCES products (id)
    )
    ''')

    # Users table (for order simulation)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE
    )
    ''')
    
    # Product popularity scores
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS product_popularity (
        product_id INTEGER PRIMARY KEY,
        score REAL NOT NULL DEFAULT 0,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (product_id) REFERENCES products (id)
    )
    ''')
    
    conn.commit()
    print("Database schema created successfully")

CATEGORIES = {
    "Electronics": {
        "subcategories": ["Smartphones", "Laptops", "Tablets", "Cameras", "Headphones", "Speakers"],
        "features": ["brand", "model", "processor", "memory", "storage", "screen_size", "battery_life", "color"],
        "price_range": (99, 2500)
    },
    "Clothing": {
        "subcategories": ["Men's", "Women's", "Children's", "Sports", "Formal", "Casual"],
        "features": ["brand", "size", "color", "material", "fit", "season"],
        "price_range": (10, 300)
    },
    "Home & Kitchen": {
        "subcategories": ["Furniture", "Appliances", "Cookware", "Bedding", "Decor", "Storage"],
        "features": ["brand", "material", "color", "dimensions", "weight", "warranty"],
        "price_range": (15, 1200)
    },
    "Books": {
        "subcategories": ["Fiction", "Non-fiction", "Educational", "Children's", "Biography", "Science"],
        "features": ["author", "publisher", "language", "pages", "format", "publication_year"],
        "price_range": (5, 50)
    },
    "Beauty & Personal Care": {
        "subcategories": ["Skincare", "Haircare", "Makeup", "Fragrance", "Bath & Body", "Men's Grooming"],
        "features": ["brand", "skin_type", "ingredients", "volume", "age_range", "organic"],
        "price_range": (5, 150)
    }
}

def generate_feature_value(feature_name):
    if feature_name == "brand":
        return random.choice(["Apple", "Samsung", "Sony", "LG", "Nike", "Adidas", "Amazon", "IKEA", "Zara", "H&M", "Logitech", "Dell", "Philips"])
    elif feature_name == "color":
        return random.choice(["Black", "White", "Red", "Blue", "Green", "Gray", "Silver", "Gold", "Pink", "Purple", "Brown"])
    elif feature_name == "material":
        return random.choice(["Cotton", "Polyester", "Leather", "Plastic", "Metal", "Glass", "Wood", "Silicone", "Nylon"])
    elif feature_name == "size":
        return random.choice(["XS", "S", "M", "L", "XL", "XXL"])
    elif feature_name == "processor":
        return random.choice(["Intel i5", "Intel i7", "Intel i9", "AMD Ryzen 5", "AMD Ryzen 7", "Apple M1", "Apple M2", "Snapdragon 8"])
    elif feature_name == "memory":
        return random.choice(["4GB", "8GB", "16GB", "32GB", "64GB"])
    elif feature_name == "storage":
        return random.choice(["128GB", "256GB", "512GB", "1TB", "2TB"])
    elif feature_name == "screen_size":
        return f"{random.randint(5, 34)} inch"
    elif feature_name == "battery_life":
        return f"{random.randint(4, 24)} hours"
    elif feature_name == "author":
        return fake.name()
    elif feature_name == "publisher":
        return fake.company()
    elif feature_name == "pages":
        return str(random.randint(100, 1000))
    elif feature_name == "publication_year":
        return str(random.randint(1990, 2024))
    elif feature_name == "dimensions":
        return f"{random.randint(10, 200)}x{random.randint(10, 200)}x{random.randint(5, 100)} cm"
    elif feature_name == "weight":
        return f"{round(random.uniform(0.5, 25), 1)} kg"
    elif feature_name == "warranty":
        return f"{random.randint(1, 5)} years"
    elif feature_name == "model":
        return f"Model {fake.lexify(text='???-###')}"
    elif feature_name == "fit":
        return random.choice(["Regular", "Slim", "Relaxed", "Athletic", "Tailored"])
    elif feature_name == "season":
        return random.choice(["Spring", "Summer", "Fall", "Winter", "All Season"])
    elif feature_name == "format":
        return random.choice(["Hardcover", "Paperback", "E-book", "Audiobook"])
    elif feature_name == "language":
        return random.choice(["English", "Spanish", "French", "German", "Chinese", "Japanese"])
    elif feature_name == "skin_type":
        return random.choice(["Normal", "Dry", "Oily", "Combination", "Sensitive"])
    elif feature_name == "ingredients":
        ingredients = ["Aloe Vera", "Vitamin E", "Retinol", "Hyaluronic Acid", "Collagen", "Glycerin"]
        return ", ".join(random.sample(ingredients, random.randint(1, 3)))
    elif feature_name == "volume":
        return f"{random.randint(30, 500)} ml"
    elif feature_name == "age_range":
        return random.choice(["Teens", "20s", "30s", "40+", "All ages"])
    elif feature_name == "organic":
        return random.choice(["Yes", "No"])
    else:
        return "N/A"

def generate_product_name(category, subcategory, features):
    brand = features.get('brand', '')
    
    if category == "Electronics":
        model = features.get('model', '')
        if subcategory == "Smartphones":
            return f"{brand} {model} Smartphone"
        elif subcategory == "Laptops":
            return f"{brand} {model} Laptop {features.get('screen_size', '')}"
        else:
            return f"{brand} {model} {subcategory.rstrip('s')}"
            
    elif category == "Clothing":
        return f"{brand} {features.get('material', '')} {subcategory.rstrip('s')} {features.get('fit', '')} {features.get('color', '')}"
        
    elif category == "Home & Kitchen":
        if subcategory == "Furniture":
            return f"{brand} {features.get('material', '')} {fake.word().capitalize()} {subcategory.rstrip('s')}"
        else:
            return f"{brand} {subcategory.rstrip('s')} {features.get('model', '')}"
            
    elif category == "Books":
        words = fake.words(nb=3)
        title = " ".join(word.capitalize() for word in words)
        return f"{title} by {features.get('author', '')}"
        
    elif category == "Beauty & Personal Care":
        return f"{brand} {subcategory.rstrip('s')} {features.get('volume', '')} for {features.get('skin_type', 'All Skin Types')}"
        
    return f"{brand} {subcategory.rstrip('s')}"

def generate_product_description(category, subcategory, features):
    base_desc = fake.paragraph(nb_sentences=3)
    feature_text = ""
    
    for feature, value in features.items():
        if feature in ["brand", "model"]:  
            continue
        feature_display = feature.replace("_", " ").title()
        feature_text += f"{feature_display}: {value}. "
    
    quality_phrases = [
        "High-quality", "Premium", "Durable", "Elegant", "Modern", 
        "Classic", "Innovative", "Reliable", "Eco-friendly", "Comfortable"
    ]
    
    benefit_phrases = [
        "designed for maximum comfort",
        "built to last",
        "perfect for everyday use",
        "a must-have addition to your collection",
        "created with you in mind",
        "engineered for performance",
        "made with premium materials",
    ]
    
    quality = random.choice(quality_phrases)
    benefit = random.choice(benefit_phrases)
    
    description = f"{base_desc} This {quality} {subcategory.rstrip('s')} is {benefit}. {feature_text}"
    return description

def generate_products(num_products=500):
    product_data = []
    
    for i in range(1, num_products + 1):
        category = random.choice(list(CATEGORIES.keys()))
        category_info = CATEGORIES[category]
        subcategory = random.choice(category_info["subcategories"])
        
        features = {}
        for feature in random.sample(category_info["features"], min(len(category_info["features"]), random.randint(3, len(category_info["features"])))):
            features[feature] = generate_feature_value(feature)
        
        min_price, max_price = category_info["price_range"]
        price = round(random.uniform(min_price, max_price), 2)
        
        name = generate_product_name(category, subcategory, features)
        description = generate_product_description(category, subcategory, features)
        
        # Generate image URL (placeholder)
        # image_url = f"https://placeholder.com/products/{i}.jpg"
        
        full_category = f"{category} > {subcategory}"
        
        product = {
            "id": i,
            "name": name,
            "description": description,
            "category": full_category,
            "price": price,
            "features": json.dumps(features),
            # "image_url": image_url
        }
        
        product_data.append(product)
    
    return product_data

def insert_products(products):
    cursor.executemany(
        "INSERT INTO products (id, name, description, category, price, features) VALUES (:id, :name, :description, :category, :price, :features)",
        products
    )#imageurl 
    conn.commit()
    print(f"Inserted {len(products)} products")

def generate_users(num_users=100):
    users = []
    for i in range(1, num_users + 1):
        users.append({
            "id": i,
            "name": fake.name(),
            "email": fake.email()
        })
    return users

def insert_users(users):
    cursor.executemany(
        "INSERT INTO users (id, name, email) VALUES (:id, :name, :email)",
        users
    )
    conn.commit()
    print(f"Inserted {len(users)} users")

def generate_inventory(products):
    inventory = []
    now = datetime.datetime.now()
    popularity_factor = {i: random.random() for i in range(1, len(products) + 1)}
    
    for product in products:
        product_id = product["id"]
        base_quantity = random.randint(0, 50) 
        popularity = popularity_factor[product_id]
        adjusted_quantity = int(base_quantity * (1 + popularity * 5))
        inventory.append({
            "product_id": product_id,
            "quantity": adjusted_quantity,
            "last_updated": now.strftime("%Y-%m-%d %H:%M:%S")
        })
    
    return inventory

def insert_inventory(inventory):
    cursor.executemany(
        "INSERT INTO inventory (product_id, quantity, last_updated) VALUES (:product_id, :quantity, :last_updated)",
        inventory
    )
    conn.commit()
    print(f"Inserted inventory for {len(inventory)} products")

def generate_orders(num_orders=2000, num_users=100, num_products=500):
    orders = []
    product_popularity = defaultdict(int)
    now = datetime.datetime.now()
    start_date = now - datetime.timedelta(days=365)
    # Products have different popularities
    product_weights = np.random.pareto(1, num_products) + 1
    product_weights = product_weights / sum(product_weights)  # Normalize
    
    for i in range(1, num_orders + 1):
        # Use weighted selection to favor popular products
        product_id = np.random.choice(range(1, num_products + 1), p=product_weights)
        user_id = random.randint(1, num_users)
        days_ago = int(np.random.beta(2, 5) * 365)  # Beta distribution for more recent date
        order_date = now - datetime.timedelta(days=days_ago)
        # Quantity is usually 1, but sometimes more
        quantity = min(5, int(np.random.exponential(0.5)) + 1)  # Mostly 1, occasionally more
        # Track product popularity
        product_popularity[product_id] += quantity
        
        orders.append({
            "id": i,
            "user_id": user_id,
            "product_id": int(product_id),
            "timestamp": order_date.strftime("%Y-%m-%d %H:%M:%S"),
            "quantity": quantity
        })
    
    # Generate popularity scores
    popularity_scores = []
    for product_id, count in product_popularity.items():
        # Normalize by dividing by max count
        score = count / max(product_popularity.values())
        popularity_scores.append({
            "product_id": int(product_id),
            "score": score,
            "last_updated": now.strftime("%Y-%m-%d %H:%M:%S")
        })
    
    return orders, popularity_scores

def insert_orders(orders):
    print(orders)
    cont = input("Press Enter to continue:")
    cursor.executemany(
        "INSERT INTO orders (id, user_id, product_id, timestamp, quantity) VALUES (:id, :user_id, :product_id, :timestamp, :quantity)",
        orders
    )
    conn.commit()
    print(f"Inserted {len(orders)} orders")

def insert_popularity_scores(scores):
    # product_ids_seen = set()
    # unique_popularity_scores = []

    # for score_data in scores:
    #     product_id = score_data["product_id"]
    #     if product_id not in product_ids_seen:
    #         product_ids_seen.add(product_id)
    #         unique_popularity_scores.append(score_data)

    # print(scores)
    # cont= input("press enter to continue:")
    cursor.executemany(
        "INSERT INTO product_popularity (product_id, score, last_updated) VALUES (:product_id, :score, :last_updated)",
        scores
    )
    conn.commit()
    print(f"Inserted popularity scores for {len(scores)} products")

# Dummy embeddings for products
def generate_product_embeddings(products, embedding_dim=50):
    embeddings = []
    for product in products:
        # Create a random embedding vector as placeholder
        # In a real system, this would be generated from text using a language model
        embedding = np.random.normal(0, 1, embedding_dim).astype(np.float32)
        # Normalize to unit length
        embedding = embedding / np.linalg.norm(embedding)
        embedding_bytes = embedding.tobytes()
        embeddings.append({
            "product_id": product["id"],
            "embedding": embedding_bytes
        })
    return embeddings

def insert_embeddings(embeddings):
    cursor.executemany(
        "INSERT INTO product_embeddings (product_id, embedding) VALUES (:product_id, :embedding)",
        embeddings
    )
    conn.commit()
    print(f"Inserted embeddings for {len(embeddings)} products")

def generate_all_data(num_products=500, num_users=100, num_orders=2000):
    setup_database()
    
    products = generate_products(num_products)
    insert_products(products)
    
    users = generate_users(num_users)
    insert_users(users)
    
    inventory = generate_inventory(products)
    insert_inventory(inventory)
    
    orders, popularity_scores = generate_orders(num_orders, num_users, num_products)
    insert_orders(orders)
    insert_popularity_scores(popularity_scores)
    
    embeddings = generate_product_embeddings(products)
    insert_embeddings(embeddings)
    
    print("All data generated and inserted successfully!")

if __name__ == "__main__":
    generate_all_data(num_products=500, num_users=100, num_orders=2000)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables:", tables)
    conn.close()