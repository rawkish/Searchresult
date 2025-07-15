import sqlite3
import pandas as pd
import nltk
conn = sqlite3.connect(r'data\ecommerce.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables:", tables)
#Verbose view
# cursor.execute("SELECT * FROM products;")
# rows = cursor.fetchall()
# for row in rows:
#     print(row)
# Brief view
for i in tables:
    for j in i:
        print(j)
        df = pd.read_sql_query("SELECT * FROM "+j, conn)
        print(df)
conn.close()

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt_tab')