"""
Generate test queries for vector database benchmarking
Creates 100 diverse queries with ground truth for accuracy testing
"""

import json
import csv
import random
from typing import List, Dict

# Read the data to create realistic queries
def load_data():
    businesses = []
    with open('../../data/businesses.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        businesses = [row for row in reader]
    
    products = []
    with open('../../data/products.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        products = [row for row in reader]
    
    return businesses, products

def generate_queries(businesses: List[Dict], products: List[Dict]) -> List[Dict]:
    """Generate 100 diverse test queries"""
    queries = []
    
    # Business type queries (20 queries)
    business_types = list(set([b['business_type'] for b in businesses if b.get('business_type')]))
    for btype in business_types:
        queries.append({
            'query': f"Find {btype} businesses",
            'type': 'business_type',
            'expected_type': 'business',
            'filter': {'business_type': btype}
        })
        queries.append({
            'query': f"List all {btype} companies",
            'type': 'business_type',
            'expected_type': 'business',
            'filter': {'business_type': btype}
        })
        if len(queries) >= 20:
            break
    
    # Specific business queries (15 queries)
    sample_businesses = random.sample(businesses, min(15, len(businesses)))
    for biz in sample_businesses:
        queries.append({
            'query': f"Information about {biz.get('business_name', '')}",
            'type': 'business_name',
            'expected_type': 'business',
            'filter': {'business_id': biz.get('business_id')}
        })
    
    # Product name queries (25 queries)
    sample_products = random.sample(products, min(25, len(products)))
    for prod in sample_products:
        queries.append({
            'query': f"Find product {prod.get('product_name', '')}",
            'type': 'product_name',
            'expected_type': 'product',
            'filter': {'product_id': prod.get('product_id')}
        })
    
    # Price range queries (20 queries)
    price_ranges = [
        (0, 1000, "cheap products under $1000"),
        (1000, 2000, "moderately priced products $1000-$2000"),
        (2000, 3000, "products priced between $2000-$3000"),
        (3000, 4000, "premium products $3000-$4000"),
        (4000, 5000, "expensive products over $4000"),
    ]
    for low, high, desc in price_ranges:
        for _ in range(4):
            queries.append({
                'query': f"Show me {desc}",
                'type': 'price_range',
                'expected_type': 'product',
                'filter': {'price_min': low, 'price_max': high}
            })
    
    # Quantity queries (10 queries)
    quantity_ranges = [
        (0, 100, "low stock products"),
        (100, 500, "medium stock products"),
        (500, 1000, "high stock products"),
    ]
    for low, high, desc in quantity_ranges:
        for _ in range(3):
            queries.append({
                'query': f"Find {desc}",
                'type': 'quantity_range',
                'expected_type': 'product',
                'filter': {'quantity_min': low, 'quantity_max': high}
            })
    
    # Generic search queries (fill to 100)
    generic_queries = [
        "Best deals on products", "Top rated businesses", "Affordable options",
        "Premium services", "Budget friendly products", "High quality items",
        "Popular choices", "Trending products", "Featured businesses",
        "Special offers", "Best value products", "Exclusive services",
        "Recommended items", "Customer favorites", "Top sellers",
        "New arrivals", "Seasonal deals", "Limited edition", 
        "Luxury items", "Everyday essentials"
    ]
    for gq in generic_queries:
        if len(queries) >= 100:
            break
        queries.append({
            'query': gq,
            'type': 'generic',
            'expected_type': 'mixed',
            'filter': {}
        })
    
    # Ensure exactly 100 queries
    while len(queries) < 100:
        queries.append({
            'query': f"Search query {len(queries) + 1}",
            'type': 'generic',
            'expected_type': 'mixed',
            'filter': {}
        })
    
    return queries[:100]

if __name__ == "__main__":
    print("Generating test queries...")
    businesses, products = load_data()
    queries = generate_queries(businesses, products)
    
    # Save queries
    with open('test_queries.json', 'w', encoding='utf-8') as f:
        json.dump(queries, f, indent=2)
    
    print(f"✓ Generated {len(queries)} test queries")
    print(f"  - Business queries: {sum(1 for q in queries if q['expected_type'] == 'business')}")
    print(f"  - Product queries: {sum(1 for q in queries if q['expected_type'] == 'product')}")
    print(f"  - Mixed queries: {sum(1 for q in queries if q['expected_type'] == 'mixed')}")
    print(f"\nSaved to: test_queries.json")
