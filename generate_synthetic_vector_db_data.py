import random
import string
import csv
import json
from faker import Faker

fake = Faker()

BUSINESS_TYPES = ['online retail', 'hotel', 'transport']
NUM_BUSINESSES = 100
MIN_PRODUCTS = 50
MAX_PRODUCTS = 200

# Helper functions
def random_string(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def generate_business(business_id):
    business_name = fake.company()
    email = fake.company_email()
    business_type = random.choice(BUSINESS_TYPES)
    num_branches = random.randint(1, 10)
    branches = [fake.address().replace('\n', ', ') for _ in range(num_branches)]
    return {
        'business_id': business_id,
        'business_name': business_name,
        'email': email,
        'business_type': business_type,
        'branches': branches
    }

def generate_product(product_id, business_id):
    product_name = fake.word().capitalize() + ' ' + fake.word().capitalize()
    quantity = random.randint(1, 1000)
    price = round(random.uniform(5, 5000), 2)
    return {
        'product_id': product_id,
        'product_name': product_name,
        'quantity': quantity,
        'price': price,
        'business_id': business_id
    }

def main():
    businesses = []
    products = []
    product_id_counter = 1

    print('Generating businesses and products...')
    for business_id in range(1, NUM_BUSINESSES + 1):
        business = generate_business(business_id)
        businesses.append(business)
        num_products = random.randint(MIN_PRODUCTS, MAX_PRODUCTS)
        for _ in range(num_products):
            product = generate_product(product_id_counter, business_id)
            products.append(product)
            product_id_counter += 1

    print(f'Generated {len(businesses)} businesses and {len(products)} products.')

    # Write businesses to JSON and CSV
    with open('businesses.json', 'w', encoding='utf-8') as f:
        json.dump(businesses, f, indent=2)
    with open('businesses.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['business_id', 'business_name', 'email', 'business_type', 'branches'])
        for b in businesses:
            writer.writerow([
                b['business_id'],
                b['business_name'],
                b['email'],
                b['business_type'],
                '|'.join(b['branches'])
            ])

    # Write products to JSON and CSV
    with open('products.json', 'w', encoding='utf-8') as f:
        json.dump(products, f, indent=2)
    with open('products.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['product_id', 'product_name', 'quantity', 'price', 'business_id'])
        for p in products:
            writer.writerow([
                p['product_id'],
                p['product_name'],
                p['quantity'],
                p['price'],
                p['business_id']
            ])
    print('Data written to businesses.json, businesses.csv, products.json, and products.csv')

if __name__ == '__main__':
    main()
