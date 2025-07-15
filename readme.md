# This is a product search and recommendation system built using python libraries and other necessary ai models.

Run query.py in app\api to launch backend built using fastapi.
A simple frontend with basic operations and views is also provided here.


📌 Problem Statement

    Modern e-commerce platforms host millions of products, requiring:

    Fast and relevant search results

    Personalized recommendations

    Continuous updates based on new data via machine learning

🎯 Objective

    Design an efficient search and recommendation system that:

    - Ranks products by relevance

    - Learns continuously from incoming data

🛠️ Technologies Used

    FastAPI – Backend service development

    Faker – Synthetic product database generation

    SQLite3 – Lightweight database management

    FAISS – Vector indexing and similarity search

    Scikit-learn – Machine learning algorithms

    NLTK – Natural Language Processing (NLP) and preprocessing

    NumPy & Pandas – Data manipulation

Pickle – Model serialization

🧩 Implementation Approach
1. 🔍 Data Collection & Preprocessing

    Generated synthetic data (Users, Products, Orders, Inventory) using Faker

    Preprocessed product descriptions with NLTK

    Extracted meaningful metadata features

2. 🧠 Embedding Generation

    Converted descriptions and metadata to vector embeddings

    Tuned embedding dimensions for performance

3. 📈 Indexing & Search

    Used FAISS for high-dimensional vector indexing

    Built fast similarity search for product retrieval

4. 🤝 Recommendation System
    
    Implemented collaborative filtering with Scikit-learn

    Used content-based filtering for cold-start problems

    Applied gradient descent regression for optimization

5. 🌐 API Development
    
    Built RESTful API endpoints with FastAPI

    Optimized API performance for quick response

    Enabled continuous model learning and updates

6. 🗃️ Database Integration
    
    Designed SQLite3 schema for product information

    Wrote optimized queries for retrieval

✅ Outcomes
    
    Efficient search system handling thousands of queries

    Recommendation engine with personalized suggestions

    Architecture built for continuous learning and updates