"""
Multi-Tenant Vector Database Benchmark - Quick Test Version
Tests with 10 tenants to verify functionality before full benchmark
"""

import csv
import json
import time
import random
import psutil
import docker
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

# Import vector DB clients
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import weaviate
from weaviate.classes.config import Configure, Property, DataType
import chromadb
from chromadb.config import Settings


class ResourceMonitor:
    """Monitor CPU and memory usage"""
    
    def __init__(self):
        try:
            self.docker_client = docker.from_env()
        except:
            self.docker_client = None
            print("Warning: Docker monitoring not available")
        self.process = psutil.Process()
        
    def get_system_stats(self):
        """Get current system resource usage"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3)
        }
    
    def get_container_stats(self, container_name):
        """Get Docker container resource usage"""
        if not self.docker_client:
            return {"error": "Docker client not available"}
            
        try:
            container = self.docker_client.containers.get(container_name)
            stats = container.stats(stream=False)
            
            # Calculate CPU percentage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                        stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            cpu_percent = (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100.0 if system_delta > 0 else 0
            
            # Calculate memory usage
            memory_usage = stats['memory_stats']['usage'] / (1024**2)  # MB
            memory_limit = stats['memory_stats']['limit'] / (1024**2)  # MB
            memory_percent = (memory_usage / memory_limit) * 100 if memory_limit > 0 else 0
            
            return {
                "cpu_percent": cpu_percent,
                "memory_mb": memory_usage,
                "memory_percent": memory_percent
            }
        except Exception as e:
            return {"error": str(e)}


class MultiTenantBenchmark:
    """Benchmark multi-tenant performance across vector databases"""
    
    def __init__(self, num_tenants=10, products_per_tenant=500):
        self.num_tenants = num_tenants
        self.products_per_tenant = products_per_tenant
        self.queries_per_tenant = 10
        
        # Load embedding model
        print("Loading embedding model...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_dim = 384
        
        # Resource monitor
        self.monitor = ResourceMonitor()
        
        # Results storage
        self.results = {
            "qdrant": {},
            "weaviate": {},
            "chromadb": {}
        }
        
    def load_data(self):
        """Load multi-tenant data"""
        print("\nLoading multi-tenant data...")
        
        data_path = Path(__file__).parent / "../../data"
        
        # Load businesses
        businesses_file = data_path / "multitenant_businesses.csv"
        with open(businesses_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.businesses = list(reader)
        
        # Load products
        products_file = data_path / "multitenant_products.csv"
        with open(products_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.products = list(reader)
        
        # Group products by business
        self.products_by_business = {}
        for product in self.products:
            business_id = product['business_id']
            if business_id not in self.products_by_business:
                self.products_by_business[business_id] = []
            self.products_by_business[business_id].append(product)
        
        print(f"✓ Loaded {len(self.businesses)} businesses (using first {self.num_tenants})")
        print(f"✓ Loaded {len(self.products)} products total")
        
    def generate_embedding(self, text):
        """Generate embedding for text"""
        return self.model.encode(text).tolist()
    
    def benchmark_qdrant(self):
        """Benchmark Qdrant multi-tenant performance"""
        print("\n" + "="*60)
        print("BENCHMARKING QDRANT - MULTI-TENANT")
        print("="*60)
        
        client = QdrantClient(host="localhost", port=6333)
        results = {
            "insertion_times": [],
            "query_latencies": [],
            "memory_usage": [],
            "cpu_usage": [],
            "cross_tenant_leakage": 0,
            "total_queries": 0
        }
        
        start_time = time.time()
        
        # Phase 1: Insert data for each tenant (separate collections)
        print(f"\nPhase 1: Inserting {self.products_per_tenant} products for each of {self.num_tenants} tenants...")
        
        tenants_to_test = self.businesses[:self.num_tenants]
        
        for idx, business in enumerate(tqdm(tenants_to_test, desc="Creating tenant collections")):
            business_id = business['business_id']
            collection_name = f"tenant_{business_id}"
            
            # Create collection for this tenant
            try:
                client.delete_collection(collection_name)
            except:
                pass
            
            insert_start = time.time()
            
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
            )
            
            # Get products for this tenant
            products = self.products_by_business.get(business_id, [])[:self.products_per_tenant]
            
            # Prepare points
            points = []
            for product in products:
                text = f"{product['product_name']} {product['category']} price: {product['price']}"
                embedding = self.generate_embedding(text)
                
                point = PointStruct(
                    id=len(points),
                    vector=embedding,
                    payload={
                        "product_id": product['product_id'],
                        "product_name": product['product_name'],
                        "business_id": product['business_id'],
                        "price": float(product['price']),
                        "category": product['category']
                    }
                )
                points.append(point)
            
            # Insert in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                client.upsert(collection_name=collection_name, points=batch)
            
            insert_time = time.time() - insert_start
            results["insertion_times"].append(insert_time)
            
            # Monitor resources
            stats = self.monitor.get_container_stats("vector-db-benchmark-qdrant-1")
            if "error" not in stats:
                results["memory_usage"].append(stats["memory_mb"])
                results["cpu_usage"].append(stats["cpu_percent"])
        
        total_insertion_time = time.time() - start_time
        
        # Phase 2: Query each tenant's data
        print(f"\nPhase 2: Running {self.queries_per_tenant} queries per tenant...")
        
        query_start = time.time()
        
        for business in tqdm(tenants_to_test, desc="Querying tenants"):
            business_id = business['business_id']
            collection_name = f"tenant_{business_id}"
            
            products = self.products_by_business.get(business_id, [])[:self.products_per_tenant]
            
            for _ in range(self.queries_per_tenant):
                # Random query from this tenant's products
                random_product = random.choice(products)
                query_text = random_product['product_name']
                query_embedding = self.generate_embedding(query_text)
                
                # Measure query latency
                q_start = time.time()
                search_results = client.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=10
                )
                q_latency = (time.time() - q_start) * 1000  # ms
                results["query_latencies"].append(q_latency)
                results["total_queries"] += 1
                
                # Check for cross-tenant leakage
                for result in search_results:
                    if result.payload.get("business_id") != business_id:
                        results["cross_tenant_leakage"] += 1
        
        total_query_time = time.time() - query_start
        
        # Calculate statistics
        self.results["qdrant"] = {
            "total_insertion_time": total_insertion_time,
            "avg_insertion_time_per_tenant": np.mean(results["insertion_times"]),
            "total_query_time": total_query_time,
            "avg_query_latency": np.mean(results["query_latencies"]),
            "min_query_latency": np.min(results["query_latencies"]),
            "max_query_latency": np.max(results["query_latencies"]),
            "std_query_latency": np.std(results["query_latencies"]),
            "avg_memory_mb": np.mean(results["memory_usage"]) if results["memory_usage"] else 0,
            "avg_cpu_percent": np.mean(results["cpu_usage"]) if results["cpu_usage"] else 0,
            "cross_tenant_leakage": results["cross_tenant_leakage"],
            "total_queries": results["total_queries"],
            "leakage_rate": results["cross_tenant_leakage"] / results["total_queries"] if results["total_queries"] > 0 else 0
        }
        
        print(f"\n✓ Qdrant benchmark complete")
        print(f"  Total insertion time: {total_insertion_time:.2f}s")
        print(f"  Total queries: {results['total_queries']}")
        print(f"  Avg query latency: {self.results['qdrant']['avg_query_latency']:.2f}ms")
        print(f"  Cross-tenant leakage: {results['cross_tenant_leakage']} ({self.results['qdrant']['leakage_rate']*100:.4f}%)")
        
    def benchmark_weaviate(self):
        """Benchmark Weaviate multi-tenant performance"""
        print("\n" + "="*60)
        print("BENCHMARKING WEAVIATE - MULTI-TENANT")
        print("="*60)
        
        client = weaviate.connect_to_local(host="localhost", port=8081)
        
        results = {
            "insertion_times": [],
            "query_latencies": [],
            "memory_usage": [],
            "cpu_usage": [],
            "cross_tenant_leakage": 0,
            "total_queries": 0
        }
        
        start_time = time.time()
        
        # Phase 1: Insert data (using multi-tenancy feature)
        print(f"\nPhase 1: Setting up multi-tenancy and inserting {self.products_per_tenant} products per tenant...")
        
        # Create a single collection with multi-tenancy enabled
        collection_name = "MultiTenantProducts"
        
        try:
            client.collections.delete(collection_name)
        except:
            pass
        
        # Create collection with multi-tenancy
        collection = client.collections.create(
            name=collection_name,
            properties=[
                Property(name="product_id", data_type=DataType.TEXT),
                Property(name="product_name", data_type=DataType.TEXT),
                Property(name="business_id", data_type=DataType.TEXT),
                Property(name="category", data_type=DataType.TEXT),
                Property(name="price", data_type=DataType.NUMBER)
            ],
            vectorizer_config=Configure.Vectorizer.none(),
            multi_tenancy_config=Configure.multi_tenancy(enabled=True)
        )
        
        tenants_to_test = self.businesses[:self.num_tenants]
        
        # Create tenants
        print("Creating tenants...")
        tenant_names = [business['business_id'] for business in tenants_to_test]
        collection.tenants.create(tenant_names)
        
        # Insert data for each tenant
        for idx, business in enumerate(tqdm(tenants_to_test, desc="Inserting tenant data")):
            business_id = business['business_id']
            
            insert_start = time.time()
            
            # Get tenant-specific collection
            tenant_collection = client.collections.get(collection_name).with_tenant(business_id)
            
            # Get products for this tenant
            products = self.products_by_business.get(business_id, [])[:self.products_per_tenant]
            
            # Prepare data objects
            data_objects = []
            for product in products:
                text = f"{product['product_name']} {product['category']} price: {product['price']}"
                embedding = self.generate_embedding(text)
                
                data_objects.append({
                    "product_id": product['product_id'],
                    "product_name": product['product_name'],
                    "business_id": product['business_id'],
                    "category": product['category'],
                    "price": float(product['price']),
                    "vector": embedding
                })
            
            # Insert in batches
            batch_size = 100
            for i in range(0, len(data_objects), batch_size):
                batch = data_objects[i:i+batch_size]
                with tenant_collection.batch.dynamic() as batch_insert:
                    for obj in batch:
                        vector = obj.pop("vector")
                        batch_insert.add_object(properties=obj, vector=vector)
            
            insert_time = time.time() - insert_start
            results["insertion_times"].append(insert_time)
            
            # Monitor resources
            stats = self.monitor.get_container_stats("vector-db-benchmark-weaviate-1")
            if "error" not in stats:
                results["memory_usage"].append(stats["memory_mb"])
                results["cpu_usage"].append(stats["cpu_percent"])
        
        total_insertion_time = time.time() - start_time
        
        # Phase 2: Query each tenant's data
        print(f"\nPhase 2: Running {self.queries_per_tenant} queries per tenant...")
        
        query_start = time.time()
        
        for business in tqdm(tenants_to_test, desc="Querying tenants"):
            business_id = business['business_id']
            
            # Get tenant-specific collection
            tenant_collection = client.collections.get(collection_name).with_tenant(business_id)
            
            products = self.products_by_business.get(business_id, [])[:self.products_per_tenant]
            
            for _ in range(self.queries_per_tenant):
                # Random query from this tenant's products
                random_product = random.choice(products)
                query_text = random_product['product_name']
                query_embedding = self.generate_embedding(query_text)
                
                # Measure query latency
                q_start = time.time()
                search_results = tenant_collection.query.near_vector(
                    near_vector=query_embedding,
                    limit=10
                )
                q_latency = (time.time() - q_start) * 1000  # ms
                results["query_latencies"].append(q_latency)
                results["total_queries"] += 1
                
                # Check for cross-tenant leakage
                for result in search_results.objects:
                    if result.properties.get("business_id") != business_id:
                        results["cross_tenant_leakage"] += 1
        
        total_query_time = time.time() - query_start
        
        # Clean up
        client.close()
        
        # Calculate statistics
        self.results["weaviate"] = {
            "total_insertion_time": total_insertion_time,
            "avg_insertion_time_per_tenant": np.mean(results["insertion_times"]),
            "total_query_time": total_query_time,
            "avg_query_latency": np.mean(results["query_latencies"]),
            "min_query_latency": np.min(results["query_latencies"]),
            "max_query_latency": np.max(results["query_latencies"]),
            "std_query_latency": np.std(results["query_latencies"]),
            "avg_memory_mb": np.mean(results["memory_usage"]) if results["memory_usage"] else 0,
            "avg_cpu_percent": np.mean(results["cpu_usage"]) if results["cpu_usage"] else 0,
            "cross_tenant_leakage": results["cross_tenant_leakage"],
            "total_queries": results["total_queries"],
            "leakage_rate": results["cross_tenant_leakage"] / results["total_queries"] if results["total_queries"] > 0 else 0
        }
        
        print(f"\n✓ Weaviate benchmark complete")
        print(f"  Total insertion time: {total_insertion_time:.2f}s")
        print(f"  Total queries: {results['total_queries']}")
        print(f"  Avg query latency: {self.results['weaviate']['avg_query_latency']:.2f}ms")
        print(f"  Cross-tenant leakage: {results['cross_tenant_leakage']} ({self.results['weaviate']['leakage_rate']*100:.4f}%)")
    
    def benchmark_chromadb(self):
        """Benchmark ChromaDB multi-tenant performance"""
        print("\n" + "="*60)
        print("BENCHMARKING CHROMADB - MULTI-TENANT")
        print("="*60)
        
        client = chromadb.HttpClient(host="localhost", port=8000)
        
        results = {
            "insertion_times": [],
            "query_latencies": [],
            "memory_usage": [],
            "cpu_usage": [],
            "cross_tenant_leakage": 0,
            "total_queries": 0
        }
        
        start_time = time.time()
        
        # Phase 1: Insert data for each tenant (separate collections)
        print(f"\nPhase 1: Inserting {self.products_per_tenant} products for each of {self.num_tenants} tenants...")
        
        tenants_to_test = self.businesses[:self.num_tenants]
        
        for idx, business in enumerate(tqdm(tenants_to_test, desc="Creating tenant collections")):
            business_id = business['business_id']
            collection_name = f"tenant_{business_id}"
            
            # Create collection for this tenant
            try:
                client.delete_collection(collection_name)
            except:
                pass
            
            insert_start = time.time()
            
            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Get products for this tenant
            products = self.products_by_business.get(business_id, [])[:self.products_per_tenant]
            
            # Prepare data
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for product in products:
                text = f"{product['product_name']} {product['category']} price: {product['price']}"
                embedding = self.generate_embedding(text)
                
                ids.append(product['product_id'])
                embeddings.append(embedding)
                metadatas.append({
                    "product_id": product['product_id'],
                    "product_name": product['product_name'],
                    "business_id": product['business_id'],
                    "price": float(product['price']),
                    "category": product['category']
                })
                documents.append(text)
            
            # Insert in batches
            batch_size = 5000
            for i in range(0, len(ids), batch_size):
                collection.add(
                    ids=ids[i:i+batch_size],
                    embeddings=embeddings[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size],
                    documents=documents[i:i+batch_size]
                )
            
            insert_time = time.time() - insert_start
            results["insertion_times"].append(insert_time)
            
            # Monitor resources
            stats = self.monitor.get_container_stats("vector-db-benchmark-chroma-1")
            if "error" not in stats:
                results["memory_usage"].append(stats["memory_mb"])
                results["cpu_usage"].append(stats["cpu_percent"])
        
        total_insertion_time = time.time() - start_time
        
        # Phase 2: Query each tenant's data
        print(f"\nPhase 2: Running {self.queries_per_tenant} queries per tenant...")
        
        query_start = time.time()
        
        for business in tqdm(tenants_to_test, desc="Querying tenants"):
            business_id = business['business_id']
            collection_name = f"tenant_{business_id}"
            
            collection = client.get_collection(collection_name)
            
            products = self.products_by_business.get(business_id, [])[:self.products_per_tenant]
            
            for _ in range(self.queries_per_tenant):
                # Random query from this tenant's products
                random_product = random.choice(products)
                query_text = random_product['product_name']
                query_embedding = self.generate_embedding(query_text)
                
                # Measure query latency
                q_start = time.time()
                search_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=10
                )
                q_latency = (time.time() - q_start) * 1000  # ms
                results["query_latencies"].append(q_latency)
                results["total_queries"] += 1
                
                # Check for cross-tenant leakage
                if search_results['metadatas']:
                    for metadata in search_results['metadatas'][0]:
                        if metadata.get("business_id") != business_id:
                            results["cross_tenant_leakage"] += 1
        
        total_query_time = time.time() - query_start
        
        # Calculate statistics
        self.results["chromadb"] = {
            "total_insertion_time": total_insertion_time,
            "avg_insertion_time_per_tenant": np.mean(results["insertion_times"]),
            "total_query_time": total_query_time,
            "avg_query_latency": np.mean(results["query_latencies"]),
            "min_query_latency": np.min(results["query_latencies"]),
            "max_query_latency": np.max(results["query_latencies"]),
            "std_query_latency": np.std(results["query_latencies"]),
            "avg_memory_mb": np.mean(results["memory_usage"]) if results["memory_usage"] else 0,
            "avg_cpu_percent": np.mean(results["cpu_usage"]) if results["cpu_usage"] else 0,
            "cross_tenant_leakage": results["cross_tenant_leakage"],
            "total_queries": results["total_queries"],
            "leakage_rate": results["cross_tenant_leakage"] / results["total_queries"] if results["total_queries"] > 0 else 0
        }
        
        print(f"\n✓ ChromaDB benchmark complete")
        print(f"  Total insertion time: {total_insertion_time:.2f}s")
        print(f"  Total queries: {results['total_queries']}")
        print(f"  Avg query latency: {self.results['chromadb']['avg_query_latency']:.2f}ms")
        print(f"  Cross-tenant leakage: {results['cross_tenant_leakage']} ({self.results['chromadb']['leakage_rate']*100:.4f}%)")
    
    def save_results(self):
        """Save benchmark results"""
        results_dir = Path(__file__).parent / "../../results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter out empty results
        valid_results = {k: v for k, v in self.results.items() if v}
        
        if not valid_results:
            print("\n❌ No valid results to save")
            return
        
        # Save detailed JSON results
        json_file = results_dir / "multitenant_benchmark_results.json"
        with open(json_file, 'w') as f:
            json.dump(valid_results, f, indent=2)
        
        print(f"\n✓ Detailed results saved to {json_file}")
        
        # Save CSV summary
        csv_file = results_dir / "multitenant_benchmark_summary.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Database", "Total Insertion Time (s)", "Avg Insertion Time/Tenant (s)",
                "Total Query Time (s)", "Avg Query Latency (ms)", "Min Latency (ms)",
                "Max Latency (ms)", "Std Latency (ms)", "Avg Memory (MB)", "Avg CPU (%)",
                "Cross-Tenant Leakage", "Total Queries", "Leakage Rate (%)"
            ])
            
            for db_name, results in valid_results.items():
                writer.writerow([
                    db_name.capitalize(),
                    f"{results['total_insertion_time']:.2f}",
                    f"{results['avg_insertion_time_per_tenant']:.4f}",
                    f"{results['total_query_time']:.2f}",
                    f"{results['avg_query_latency']:.2f}",
                    f"{results['min_query_latency']:.2f}",
                    f"{results['max_query_latency']:.2f}",
                    f"{results['std_query_latency']:.2f}",
                    f"{results['avg_memory_mb']:.2f}",
                    f"{results['avg_cpu_percent']:.2f}",
                    results['cross_tenant_leakage'],
                    results['total_queries'],
                    f"{results['leakage_rate']*100:.4f}"
                ])
        
        print(f"✓ Summary saved to {csv_file}")
        
        # Print summary table
        self.print_summary()
    
    def print_summary(self):
        """Print benchmark summary table"""
        # Filter out empty results
        valid_results = {k: v for k, v in self.results.items() if v}
        
        if not valid_results:
            print("\n❌ No valid results to display")
            return
            
        print("\n" + "="*100)
        print("MULTI-TENANT BENCHMARK SUMMARY")
        print("="*100)
        print(f"Configuration: {self.num_tenants} tenants, {self.products_per_tenant} products/tenant, {self.queries_per_tenant} queries/tenant")
        print("="*100)
        
        header = f"{'Database':<12} {'Insert Time':<12} {'Query Time':<12} {'Avg Latency':<12} {'Memory (MB)':<12} {'CPU %':<8} {'Leakage':<10}"
        print(header)
        print("-"*100)
        
        for db_name, results in valid_results.items():
            row = (
                f"{db_name.capitalize():<12} "
                f"{results['total_insertion_time']:>10.2f}s  "
                f"{results['total_query_time']:>10.2f}s  "
                f"{results['avg_query_latency']:>10.2f}ms  "
                f"{results['avg_memory_mb']:>10.2f}   "
                f"{results['avg_cpu_percent']:>6.2f}  "
                f"{results['cross_tenant_leakage']:>8}"
            )
            print(row)
        
        print("="*100)
        
        # Find winners
        min_insert = min(valid_results.items(), key=lambda x: x[1]['total_insertion_time'])
        min_query = min(valid_results.items(), key=lambda x: x[1]['avg_query_latency'])
        zero_leakage = [db for db, res in valid_results.items() if res['cross_tenant_leakage'] == 0]
        
        print("\n🏆 WINNERS:")
        print(f"  Fastest Insertion: {min_insert[0].capitalize()} ({min_insert[1]['total_insertion_time']:.2f}s)")
        print(f"  Fastest Queries: {min_query[0].capitalize()} ({min_query[1]['avg_query_latency']:.2f}ms)")
        print(f"  Zero Leakage: {', '.join([db.capitalize() for db in zero_leakage]) if zero_leakage else 'None'}")
        print("="*100)


def main():
    """Main benchmark execution"""
    print("="*60)
    print("MULTI-TENANT VECTOR DATABASE BENCHMARK - QUICK TEST")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create benchmark instance with smaller numbers for testing
    benchmark = MultiTenantBenchmark(
        num_tenants=10,
        products_per_tenant=500
    )
    
    # Load data
    benchmark.load_data()
    
    # Run benchmarks
    try:
        benchmark.benchmark_qdrant()
    except Exception as e:
        print(f"\n❌ Qdrant benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        benchmark.benchmark_weaviate()
    except Exception as e:
        print(f"\n❌ Weaviate benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        benchmark.benchmark_chromadb()
    except Exception as e:
        print(f"\n❌ ChromaDB benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Save and display results
    benchmark.save_results()


if __name__ == "__main__":
    main()
