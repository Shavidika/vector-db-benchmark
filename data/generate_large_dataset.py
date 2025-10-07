"""
Generate larger datasets for comprehensive benchmarking
Creates 500 businesses with 50,000 products (100 products per business)
"""

import csv
import json
import random
from pathlib import Path
from faker import Faker

fake = Faker()
random.seed(42)

# Constants
NUM_BUSINESSES = 500
PRODUCTS_PER_BUSINESS = 100
TOTAL_PRODUCTS = NUM_BUSINESSES * PRODUCTS_PER_BUSINESS

BUSINESS_TYPES = ["transport", "online retail", "hotel", "restaurant", "technology", "healthcare"]
PRODUCT_CATEGORIES = ["Electronics", "Clothing", "Food", "Books", "Sports", "Home", "Toys", "Beauty"]

def generate_businesses(num_businesses):
    """Generate business data"""
    businesses = []
    
    for i in range(1, num_businesses + 1):
        business = {
            "business_id": f"B{i:04d}",
            "business_name": fake.company(),
            "email": fake.company_email(),
            "business_type": random.choice(BUSINESS_TYPES),
            "branches": "|".join([fake.address().replace("\n", ", ") for _ in range(random.randint(1, 3))])
        }
        businesses.append(business)
    
    return businesses

def generate_products(businesses):
    """Generate product data for all businesses"""
    products = []
    
    for business in businesses:
        business_id = business["business_id"]
        
        for j in range(PRODUCTS_PER_BUSINESS):
            product = {
                "product_id": f"{business_id}_P{j:04d}",
                "product_name": f"{fake.word().capitalize()} {random.choice(PRODUCT_CATEGORIES)} {fake.word().capitalize()}",
                "quantity": random.randint(1, 1000),
                "price": round(random.uniform(9.99, 9999.99), 2),
                "business_id": business_id
            }
            products.append(product)
    
    return products

def save_data(businesses, products):
    """Save data to CSV and JSON files"""
    data_dir = Path(__file__).parent
    
    # Save businesses
    businesses_csv = data_dir / "businesses.csv"
    businesses_json = data_dir / "businesses.json"
    
    print(f"Saving {len(businesses)} businesses...")
    
    with open(businesses_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=businesses[0].keys())
        writer.writeheader()
        writer.writerows(businesses)
    
    with open(businesses_json, 'w', encoding='utf-8') as f:
        json.dump(businesses, f, indent=2)
    
    print(f"  ✓ Saved to {businesses_csv}")
    
    # Save products
    products_csv = data_dir / "products.csv"
    products_json = data_dir / "products.json"
    
    print(f"Saving {len(products)} products...")
    
    with open(products_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=products[0].keys())
        writer.writeheader()
        writer.writerows(products)
    
    with open(products_json, 'w', encoding='utf-8') as f:
        json.dump(products, f, indent=2)
    
    print(f"  ✓ Saved to {products_csv}")
    
    print(f"\n{'='*60}")
    print(f"Dataset Generation Complete!")
    print(f"{'='*60}")
    print(f"Businesses: {len(businesses):,}")
    print(f"Products: {len(products):,}")
    print(f"Products per business: {PRODUCTS_PER_BUSINESS}")
    print(f"{'='*60}")

if __name__ == "__main__":
    print(f"Generating larger dataset...")
    print(f"  Businesses: {NUM_BUSINESSES}")
    print(f"  Products per business: {PRODUCTS_PER_BUSINESS}")
    print(f"  Total products: {TOTAL_PRODUCTS:,}\n")
    
    businesses = generate_businesses(NUM_BUSINESSES)
    products = generate_products(businesses)
    save_data(businesses, products)
