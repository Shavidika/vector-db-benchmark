"""
Multi-Tenant Data Generator for Vector Database Benchmarking
Generates 1,000 businesses with 500 products each (500,000 total products)
"""

import csv
import json
import random
from datetime import datetime
from pathlib import Path

# Product categories and attributes
PRODUCT_CATEGORIES = [
    "Electronics", "Clothing", "Home & Garden", "Sports", "Books",
    "Toys", "Food & Beverage", "Beauty", "Automotive", "Office Supplies"
]

PRODUCT_PREFIXES = [
    "Premium", "Deluxe", "Professional", "Eco-Friendly", "Smart",
    "Ultra", "Classic", "Modern", "Vintage", "Essential"
]

PRODUCT_TYPES = [
    "Widget", "Gadget", "Tool", "Kit", "Set", "System", "Device",
    "Accessory", "Component", "Package", "Bundle", "Solution"
]

BUSINESS_TYPES = ["retail", "wholesale", "online", "manufacturer", "distributor"]

CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
    "Austin", "Jacksonville", "Fort Worth", "Columbus", "Indianapolis",
    "Charlotte", "San Francisco", "Seattle", "Denver", "Washington DC"
]

def generate_business_name(business_id):
    """Generate a unique business name"""
    prefixes = ["Global", "Premier", "United", "First", "National", "Metro", "Elite"]
    suffixes = ["Corp", "Inc", "LLC", "Group", "Solutions", "Enterprises", "Co"]
    
    prefix = random.choice(prefixes)
    suffix = random.choice(suffixes)
    category = random.choice(PRODUCT_CATEGORIES)
    
    return f"{prefix} {category} {suffix} #{business_id}"

def generate_products_for_business(business_id, num_products=500):
    """Generate products for a specific business"""
    products = []
    
    for i in range(num_products):
        product_id = f"B{business_id:04d}_P{i:04d}"
        
        # Generate product name
        prefix = random.choice(PRODUCT_PREFIXES)
        category = random.choice(PRODUCT_CATEGORIES)
        product_type = random.choice(PRODUCT_TYPES)
        product_name = f"{prefix} {category} {product_type}"
        
        # Generate attributes
        quantity = random.randint(1, 1000)
        price = round(random.uniform(9.99, 999.99), 2)
        
        products.append({
            "product_id": product_id,
            "product_name": product_name,
            "business_id": f"B{business_id:04d}",
            "quantity": quantity,
            "price": price,
            "category": category
        })
    
    return products

def generate_multitenant_data(num_businesses=1000, products_per_business=500):
    """Generate multi-tenant data for benchmarking"""
    
    print(f"Generating {num_businesses} businesses with {products_per_business} products each...")
    print(f"Total products: {num_businesses * products_per_business:,}")
    
    businesses = []
    all_products = []
    
    for business_id in range(1, num_businesses + 1):
        if business_id % 100 == 0:
            print(f"  Generated {business_id}/{num_businesses} businesses...")
        
        # Generate business
        business = {
            "business_id": f"B{business_id:04d}",
            "business_name": generate_business_name(business_id),
            "email": f"contact{business_id}@business{business_id}.com",
            "business_type": random.choice(BUSINESS_TYPES),
            "city": random.choice(CITIES),
            "num_products": products_per_business
        }
        businesses.append(business)
        
        # Generate products for this business
        products = generate_products_for_business(business_id, products_per_business)
        all_products.extend(products)
    
    return businesses, all_products

def save_data(businesses, products, output_dir="../../data"):
    """Save generated data to CSV and JSON files"""
    
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save businesses
    businesses_csv = output_path / "multitenant_businesses.csv"
    businesses_json = output_path / "multitenant_businesses.json"
    
    print(f"\nSaving {len(businesses)} businesses...")
    
    with open(businesses_csv, 'w', newline='', encoding='utf-8') as f:
        if businesses:
            writer = csv.DictWriter(f, fieldnames=businesses[0].keys())
            writer.writeheader()
            writer.writerows(businesses)
    
    with open(businesses_json, 'w', encoding='utf-8') as f:
        json.dump(businesses, f, indent=2)
    
    print(f"  ✓ Saved to {businesses_csv}")
    print(f"  ✓ Saved to {businesses_json}")
    
    # Save products
    products_csv = output_path / "multitenant_products.csv"
    products_json = output_path / "multitenant_products.json"
    
    print(f"\nSaving {len(products)} products...")
    
    with open(products_csv, 'w', newline='', encoding='utf-8') as f:
        if products:
            writer = csv.DictWriter(f, fieldnames=products[0].keys())
            writer.writeheader()
            writer.writerows(products)
    
    with open(products_json, 'w', encoding='utf-8') as f:
        json.dump(products, f, indent=2)
    
    print(f"  ✓ Saved to {products_csv}")
    print(f"  ✓ Saved to {products_json}")
    
    # Save summary
    summary = {
        "generated_at": datetime.now().isoformat(),
        "num_businesses": len(businesses),
        "num_products": len(products),
        "products_per_business": len(products) // len(businesses) if businesses else 0,
        "business_types": list(set(b["business_type"] for b in businesses)),
        "cities": list(set(b["city"] for b in businesses)),
        "product_categories": list(set(p["category"] for p in products))
    }
    
    summary_file = output_path / "multitenant_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved to {summary_file}")
    print(f"\n{'='*60}")
    print(f"Data Generation Complete!")
    print(f"{'='*60}")
    print(f"Businesses: {len(businesses):,}")
    print(f"Products: {len(products):,}")
    print(f"Products per business: {len(products) // len(businesses)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Generate data
    businesses, products = generate_multitenant_data(
        num_businesses=1000,
        products_per_business=500
    )
    
    # Save to files
    save_data(businesses, products)
