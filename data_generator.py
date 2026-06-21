import pandas as pd
import numpy as np
import os
import random

def generate_customer_data(filepath='data/customer_data.csv', num_records=2000):
    """
    Generates a synthetic e-commerce dataset for training.
    """
    os.makedirs('data', exist_ok=True)
    
    if os.path.exists(filepath):
        print(f"Dataset already exists at {filepath}. Skipping generation.")
        return

    print("Generating synthetic customer data...")
    np.random.seed(42)
    random.seed(42)

    # Features
    ages = np.random.randint(18, 70, size=num_records)
    incomes = np.random.normal(loc=60000, scale=20000, size=num_records).astype(int)
    incomes = np.clip(incomes, 20000, 150000)
    
    # Categories
    categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Beauty', 'Sports']
    favorite_categories = np.random.choice(categories, size=num_records)
    
    browsing_time_mins = np.random.normal(loc=15, scale=10, size=num_records)
    browsing_time_mins = np.clip(browsing_time_mins, 1, 120).round(1)
    
    pages_visited = np.clip(np.random.normal(loc=5, scale=3, size=num_records), 1, 30).astype(int)
    
    past_purchases = np.clip(np.random.normal(loc=3, scale=2, size=num_records), 0, 20).astype(int)
    
    # Target Variable Logic: Probability of Purchase based on features
    # Higher income, more browsing time, more pages, more past purchases -> higher chance
    
    purchase_prob = (
        (incomes - 20000) / 130000 * 0.2 +
        (browsing_time_mins / 120) * 0.3 +
        (pages_visited / 30) * 0.2 +
        (past_purchases / 20) * 0.3
    )
    
    # Add some random noise
    purchase_prob += np.random.normal(loc=0, scale=0.1, size=num_records)
    purchase_prob = np.clip(purchase_prob, 0, 1)
    
    # Convert to binary label
    purchased = (purchase_prob > 0.45).astype(int)

    # Convert to float to allow NaN injection
    ages = ages.astype(float)
    incomes = incomes.astype(float)

    # Add missing values randomly (approx 5%) to simulate real-world data and require preprocessing
    for i in range(num_records):
        if random.random() < 0.05:
            ages[i] = np.nan
        if random.random() < 0.05:
            incomes[i] = np.nan
            
    df = pd.DataFrame({
        'CustomerID': range(1, num_records + 1),
        'Age': ages,
        'Income': incomes,
        'FavoriteCategory': favorite_categories,
        'BrowsingTimeMins': browsing_time_mins,
        'PagesVisited': pages_visited,
        'PastPurchases': past_purchases,
        'Purchased': purchased
    })

    # Add some duplicates to test preprocessing
    duplicates = df.sample(n=50, random_state=42)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    df.to_csv(filepath, index=False)
    print(f"Generated synthetic dataset with {len(df)} records at {filepath}")

def generate_product_catalog(filepath='data/product_catalog.csv'):
    os.makedirs('data', exist_ok=True)
    
    if os.path.exists(filepath):
        return

    # Basic product catalog for recommendation engine
    products = [
        # Electronics
        {"ProductID": 101, "Category": "Electronics", "Name": "Wireless Noise-Canceling Headphones", "Price": 299.99, "Rating": 4.8},
        {"ProductID": 102, "Category": "Electronics", "Name": "Smartphone 5G", "Price": 899.99, "Rating": 4.6},
        {"ProductID": 103, "Category": "Electronics", "Name": "4K Ultra HD Smart TV", "Price": 499.99, "Rating": 4.5},
        {"ProductID": 104, "Category": "Electronics", "Name": "Gaming Laptop 16GB RAM", "Price": 1200.00, "Rating": 4.7},
        {"ProductID": 105, "Category": "Electronics", "Name": "Smartwatch with Heart Rate Monitor", "Price": 199.99, "Rating": 4.3},
        # Clothing
        {"ProductID": 201, "Category": "Clothing", "Name": "Men's Classic T-Shirt", "Price": 19.99, "Rating": 4.2},
        {"ProductID": 202, "Category": "Clothing", "Name": "Women's Denim Jacket", "Price": 59.99, "Rating": 4.5},
        {"ProductID": 203, "Category": "Clothing", "Name": "Running Shoes Lightweight", "Price": 89.99, "Rating": 4.4},
        {"ProductID": 204, "Category": "Clothing", "Name": "Winter Puffer Coat", "Price": 120.00, "Rating": 4.6},
        {"ProductID": 205, "Category": "Clothing", "Name": "Yoga Pants High Waist", "Price": 35.00, "Rating": 4.8},
        # Home & Kitchen
        {"ProductID": 301, "Category": "Home & Kitchen", "Name": "Programmable Coffee Maker", "Price": 79.99, "Rating": 4.1},
        {"ProductID": 302, "Category": "Home & Kitchen", "Name": "Non-Stick Cookware Set", "Price": 149.99, "Rating": 4.5},
        {"ProductID": 303, "Category": "Home & Kitchen", "Name": "Robot Vacuum Cleaner", "Price": 249.99, "Rating": 4.3},
        {"ProductID": 304, "Category": "Home & Kitchen", "Name": "Microfiber Bed Sheets", "Price": 29.99, "Rating": 4.7},
        {"ProductID": 305, "Category": "Home & Kitchen", "Name": "Air Purifier HEPA Filter", "Price": 119.99, "Rating": 4.6},
        # Beauty
        {"ProductID": 401, "Category": "Beauty", "Name": "Hydrating Face Moisturizer", "Price": 24.99, "Rating": 4.6},
        {"ProductID": 402, "Category": "Beauty", "Name": "Vitamin C Serum", "Price": 19.99, "Rating": 4.4},
        {"ProductID": 403, "Category": "Beauty", "Name": "Eyeshadow Palette", "Price": 39.99, "Rating": 4.8},
        {"ProductID": 404, "Category": "Beauty", "Name": "Matte Liquid Lipstick", "Price": 15.00, "Rating": 4.2},
        {"ProductID": 405, "Category": "Beauty", "Name": "Exfoliating Body Scrub", "Price": 22.00, "Rating": 4.5},
        # Sports
        {"ProductID": 501, "Category": "Sports", "Name": "Yoga Mat with Alignment Lines", "Price": 29.99, "Rating": 4.7},
        {"ProductID": 502, "Category": "Sports", "Name": "Adjustable Dumbbells Set", "Price": 199.99, "Rating": 4.9},
        {"ProductID": 503, "Category": "Sports", "Name": "Resistance Bands Set", "Price": 14.99, "Rating": 4.3},
        {"ProductID": 504, "Category": "Sports", "Name": "Foam Roller for Muscle Massage", "Price": 18.99, "Rating": 4.5},
        {"ProductID": 505, "Category": "Sports", "Name": "Protein Powder Whey Isolate", "Price": 49.99, "Rating": 4.6},
    ]
    df = pd.DataFrame(products)
    df.to_csv(filepath, index=False)
    print(f"Generated product catalog at {filepath}")

def generate_user_interactions(filepath='data/user_interactions.csv', num_users=500, num_interactions=3000):
    os.makedirs('data', exist_ok=True)
    if os.path.exists(filepath):
        return
        
    print("Generating user interactions for collaborative filtering...")
    
    np.random.seed(42)
    random.seed(42)
    
    product_df = pd.read_csv('data/product_catalog.csv')
    product_ids = product_df['ProductID'].tolist()
    
    interactions = []
    for _ in range(num_interactions):
        user_id = np.random.randint(1, num_users + 1)
        product_id = np.random.choice(product_ids)
        
        # Interactions: 1=View, 2=Cart, 3=Purchase
        interaction_type = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
        interactions.append({
            'CustomerID': user_id,
            'ProductID': product_id,
            'InteractionScore': interaction_type
        })
        
    df = pd.DataFrame(interactions)
    # Aggregate scores
    df = df.groupby(['CustomerID', 'ProductID'])['InteractionScore'].sum().reset_index()
    # Normalize a bit
    df['InteractionScore'] = np.clip(df['InteractionScore'], 1, 5)
    
    df.to_csv(filepath, index=False)
    print(f"Generated user interactions at {filepath}")

if __name__ == "__main__":
    generate_customer_data()
    generate_product_catalog()
    generate_user_interactions()
