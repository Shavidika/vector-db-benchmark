"""
Vector Database Benchmarking Script
Compares ingestion performance of Qdrant, Weaviate, and Chroma
"""

import json
import csv
import time
import os
from typing import List, Dict, Tuple
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import weaviate
from weaviate.classes.init import Auth
import chromadb
from chromadb.config import Settings


class VectorDBBenchmark:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        self.results = {}
        
    def load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load businesses and products from CSV files"""
        print("Loading data from CSV files...")
        
        businesses = []
        with open('businesses.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                businesses.append(row)
        
        products = []
        with open('products.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                products.append(row)
        
        print(f"Loaded {len(businesses)} businesses and {len(products)} products")
        return businesses, products
    
    def create_text_for_embedding(self, item: Dict, item_type: str) -> str:
        """Create text representation for embedding"""
        if item_type == 'business':
            return f"{item.get('business_name', '')} {item.get('email', '')} {item.get('business_type', '')}"
        else:  # product
            return f"{item.get('product_name', '')} quantity: {item.get('quantity', '')} price: {item.get('price', '')}"
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using sentence-transformers"""
        return self.model.encode(texts, show_progress_bar=True)
    
    def get_storage_size(self, path: str) -> int:
        """Calculate total size of directory in bytes"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception as e:
            print(f"Error calculating size for {path}: {e}")
        return total_size
    
    def benchmark_qdrant(self, businesses: List[Dict], products: List[Dict]) -> Dict:
        """Benchmark Qdrant vector database"""
        print("\n" + "="*60)
        print("BENCHMARKING QDRANT")
        print("="*60)
        
        client = QdrantClient(host="localhost", port=6333)
        collection_name = "benchmark_collection"
        
        # Delete collection if exists
        try:
            client.delete_collection(collection_name)
        except:
            pass
        
        # Create collection
        print("Creating collection...")
        index_start = time.time()
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
        )
        index_time = time.time() - index_start
        
        # Prepare data
        print("Preparing embeddings...")
        all_items = []
        all_texts = []
        
        for business in businesses:
            text = self.create_text_for_embedding(business, 'business')
            all_texts.append(text)
            all_items.append({
                'id': f"b_{business['business_id']}",
                'payload': {**business, 'type': 'business'},
                'text': text
            })
        
        for product in products:
            text = self.create_text_for_embedding(product, 'product')
            all_texts.append(text)
            all_items.append({
                'id': f"p_{product['product_id']}",
                'payload': {**product, 'type': 'product'},
                'text': text
            })
        
        embeddings = self.generate_embeddings(all_texts)
        
        # Insert data
        print(f"Inserting {len(all_items)} records...")
        ingestion_start = time.time()
        
        points = []
        for idx, item in enumerate(all_items):
            points.append(PointStruct(
                id=idx,
                vector=embeddings[idx].tolist(),
                payload=item['payload']
            ))
        
        # Batch upload
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            client.upsert(collection_name=collection_name, points=batch)
        
        ingestion_time = time.time() - ingestion_start
        
        # Get storage size (Docker volume)
        # Note: This is an approximation as Docker volumes are harder to measure
        storage_size = 0  # Will be estimated
        
        total_records = len(all_items)
        throughput = total_records / ingestion_time if ingestion_time > 0 else 0
        
        results = {
            'database': 'Qdrant',
            'total_records': total_records,
            'ingestion_time': round(ingestion_time, 2),
            'index_build_time': round(index_time, 2),
            'throughput': round(throughput, 2),
            'storage_size_mb': 'N/A (Docker volume)'
        }
        
        print(f"✓ Completed - {total_records} records in {ingestion_time:.2f}s")
        return results
    
    def benchmark_weaviate(self, businesses: List[Dict], products: List[Dict]) -> Dict:
        """Benchmark Weaviate vector database"""
        print("\n" + "="*60)
        print("BENCHMARKING WEAVIATE")
        print("="*60)
        
        import weaviate.classes.config as wvc
        from weaviate.classes.init import AdditionalConfig, Timeout
        
        client = weaviate.connect_to_local(
            host="localhost", 
            port=8081,
            grpc_port=50051,
            skip_init_checks=True,
            additional_config=AdditionalConfig(
                timeout=Timeout(init=30, query=60, insert=120)
            )
        )
        
        try:
            collection_name = "BenchmarkCollection"
            
            # Delete collection if exists
            try:
                client.collections.delete(collection_name)
            except:
                pass
            
            # Create collection
            print("Creating collection...")
            index_start = time.time()
            from weaviate.classes.config import Property, DataType as WeaviateDataType
            client.collections.create(
                name=collection_name,
                properties=[
                    Property(name="business_id", data_type=WeaviateDataType.TEXT),
                    Property(name="business_name", data_type=WeaviateDataType.TEXT),
                    Property(name="email", data_type=WeaviateDataType.TEXT),
                    Property(name="business_type", data_type=WeaviateDataType.TEXT),
                    Property(name="branches", data_type=WeaviateDataType.TEXT),
                    Property(name="product_id", data_type=WeaviateDataType.TEXT),
                    Property(name="product_name", data_type=WeaviateDataType.TEXT),
                    Property(name="quantity", data_type=WeaviateDataType.TEXT),
                    Property(name="price", data_type=WeaviateDataType.TEXT),
                    Property(name="item_type", data_type=WeaviateDataType.TEXT),
                ]
            )
            index_time = time.time() - index_start
            
            collection = client.collections.get(collection_name)
            
            # Prepare data
            print("Preparing embeddings...")
            all_items = []
            all_texts = []
            
            for business in businesses:
                text = self.create_text_for_embedding(business, 'business')
                all_texts.append(text)
                all_items.append({
                    'properties': {
                        'business_id': business.get('business_id', ''),
                        'business_name': business.get('business_name', ''),
                        'email': business.get('email', ''),
                        'business_type': business.get('business_type', ''),
                        'branches': business.get('branches', ''),
                        'item_type': 'business'
                    },
                    'text': text
                })
            
            for product in products:
                text = self.create_text_for_embedding(product, 'product')
                all_texts.append(text)
                all_items.append({
                    'properties': {
                        'product_id': product.get('product_id', ''),
                        'product_name': product.get('product_name', ''),
                        'quantity': product.get('quantity', ''),
                        'price': product.get('price', ''),
                        'business_id': product.get('business_id', ''),
                        'item_type': 'product'
                    },
                    'text': text
                })
            
            embeddings = self.generate_embeddings(all_texts)
            
            # Insert data
            print(f"Inserting {len(all_items)} records...")
            ingestion_start = time.time()
            
            with collection.batch.dynamic() as batch:
                for idx, item in enumerate(all_items):
                    batch.add_object(
                        properties=item['properties'],
                        vector=embeddings[idx].tolist()
                    )
            
            ingestion_time = time.time() - ingestion_start
            
            total_records = len(all_items)
            throughput = total_records / ingestion_time if ingestion_time > 0 else 0
            
            results = {
                'database': 'Weaviate',
                'total_records': total_records,
                'ingestion_time': round(ingestion_time, 2),
                'index_build_time': round(index_time, 2),
                'throughput': round(throughput, 2),
                'storage_size_mb': 'N/A (Docker volume)'
            }
            
            print(f"✓ Completed - {total_records} records in {ingestion_time:.2f}s")
            return results
            
        finally:
            client.close()
    
    def benchmark_chroma(self, businesses: List[Dict], products: List[Dict]) -> Dict:
        """Benchmark ChromaDB vector database"""
        print("\n" + "="*60)
        print("BENCHMARKING CHROMADB")
        print("="*60)
        
        try:
            client = chromadb.HttpClient(host="localhost", port=8000)
            collection_name = "benchmark_collection"
            
            # Delete collection if exists
            try:
                client.delete_collection(collection_name)
            except:
                pass
            
            # Create collection
            print("Creating collection...")
            index_start = time.time()
            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            index_time = time.time() - index_start
            
            # Prepare data
            print("Preparing embeddings...")
            all_ids = []
            all_embeddings = []
            all_metadatas = []
            all_documents = []
            
            for business in businesses:
                text = self.create_text_for_embedding(business, 'business')
                all_ids.append(f"b_{business['business_id']}")
                all_documents.append(text)
                all_metadatas.append({
                    'business_id': business.get('business_id', ''),
                    'business_name': business.get('business_name', ''),
                    'email': business.get('email', ''),
                    'business_type': business.get('business_type', ''),
                    'type': 'business'
                })
            
            for product in products:
                text = self.create_text_for_embedding(product, 'product')
                all_ids.append(f"p_{product['product_id']}")
                all_documents.append(text)
                all_metadatas.append({
                    'product_id': product.get('product_id', ''),
                    'product_name': product.get('product_name', ''),
                    'quantity': product.get('quantity', ''),
                    'price': product.get('price', ''),
                    'business_id': product.get('business_id', ''),
                    'type': 'product'
                })
            
            embeddings = self.generate_embeddings(all_documents)
            all_embeddings = embeddings.tolist()
            
            # Insert data
            print(f"Inserting {len(all_ids)} records...")
            ingestion_start = time.time()
            
            # Batch insert
            batch_size = 5000
            for i in range(0, len(all_ids), batch_size):
                collection.add(
                    ids=all_ids[i:i+batch_size],
                    embeddings=all_embeddings[i:i+batch_size],
                    metadatas=all_metadatas[i:i+batch_size],
                    documents=all_documents[i:i+batch_size]
                )
            
            ingestion_time = time.time() - ingestion_start
            
            total_records = len(all_ids)
            throughput = total_records / ingestion_time if ingestion_time > 0 else 0
            
            results = {
                'database': 'ChromaDB',
                'total_records': total_records,
                'ingestion_time': round(ingestion_time, 2),
                'index_build_time': round(index_time, 2),
                'throughput': round(throughput, 2),
                'storage_size_mb': 'N/A (Docker volume)'
            }
            
            print(f"✓ Completed - {total_records} records in {ingestion_time:.2f}s")
            return results
        finally:
            # Clean up client connection
            try:
                del client
            except:
                pass
    
    def print_results_table(self, all_results: List[Dict]):
        """Print results in a formatted table"""
        print("\n" + "="*100)
        print("BENCHMARK RESULTS")
        print("="*100)
        print(f"{'Database':<15} {'Records':<12} {'Ingestion Time (s)':<20} {'Index Time (s)':<18} {'Throughput (rec/s)':<20} {'Storage (MB)':<15}")
        print("-"*100)
        
        for result in all_results:
            print(f"{result['database']:<15} "
                  f"{result['total_records']:<12} "
                  f"{result['ingestion_time']:<20} "
                  f"{result['index_build_time']:<18} "
                  f"{result['throughput']:<20} "
                  f"{result['storage_size_mb']:<15}")
        
        print("="*100)
        
        # Save to file
        with open('benchmark_results.txt', 'w') as f:
            f.write("="*100 + "\n")
            f.write("BENCHMARK RESULTS\n")
            f.write("="*100 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"{'Database':<15} {'Records':<12} {'Ingestion Time (s)':<20} {'Index Time (s)':<18} {'Throughput (rec/s)':<20} {'Storage (MB)':<15}\n")
            f.write("-"*100 + "\n")
            
            for result in all_results:
                f.write(f"{result['database']:<15} "
                       f"{result['total_records']:<12} "
                       f"{result['ingestion_time']:<20} "
                       f"{result['index_build_time']:<18} "
                       f"{result['throughput']:<20} "
                       f"{result['storage_size_mb']:<15}\n")
            
            f.write("="*100 + "\n")
        
        print("\n✓ Results saved to benchmark_results.txt")
    
    def run_benchmarks(self):
        """Run all benchmarks"""
        print("Starting Vector Database Benchmark Suite")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Load data
        businesses, products = self.load_data()
        
        all_results = []
        
        # Benchmark each database
        try:
            result = self.benchmark_qdrant(businesses, products)
            all_results.append(result)
        except Exception as e:
            print(f"✗ Qdrant benchmark failed: {e}")
            all_results.append({
                'database': 'Qdrant',
                'total_records': 0,
                'ingestion_time': 0,
                'index_build_time': 0,
                'throughput': 0,
                'storage_size_mb': 'ERROR'
            })
        
        try:
            result = self.benchmark_weaviate(businesses, products)
            all_results.append(result)
        except Exception as e:
            print(f"✗ Weaviate benchmark failed: {e}")
            all_results.append({
                'database': 'Weaviate',
                'total_records': 0,
                'ingestion_time': 0,
                'index_build_time': 0,
                'throughput': 0,
                'storage_size_mb': 'ERROR'
            })
        
        try:
            result = self.benchmark_chroma(businesses, products)
            all_results.append(result)
        except Exception as e:
            print(f"✗ ChromaDB benchmark failed: {e}")
            all_results.append({
                'database': 'ChromaDB',
                'total_records': 0,
                'ingestion_time': 0,
                'index_build_time': 0,
                'throughput': 0,
                'storage_size_mb': 'ERROR'
            })
        
        # Print results
        self.print_results_table(all_results)


if __name__ == "__main__":
    benchmark = VectorDBBenchmark()
    benchmark.run_benchmarks()
