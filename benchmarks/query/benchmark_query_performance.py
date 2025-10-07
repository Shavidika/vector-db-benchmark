"""
Vector Database Query Benchmarking Script
Tests query latency and accuracy across Qdrant, Weaviate, and ChromaDB
"""

import json
import csv
import time
import os
from typing import List, Dict, Tuple
from datetime import datetime
import statistics

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout
import weaviate.classes.config as wvc
import chromadb
import matplotlib.pyplot as plt
import pandas as pd


class QueryBenchmark:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.k = 10  # Top-k results
        self.queries = []
        self.ground_truth = {}
        
    def load_queries(self) -> List[Dict]:
        """Load test queries"""
        print("Loading test queries...")
        with open('test_queries.json', 'r', encoding='utf-8') as f:
            self.queries = json.load(f)
        print(f"✓ Loaded {len(self.queries)} queries")
        return self.queries
    
    def load_ground_truth(self) -> Dict:
        """Load data for ground truth comparison"""
        print("Loading ground truth data...")
        
        businesses = []
        with open('../../data/businesses.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            businesses = [row for row in reader]
        
        products = []
        with open('../../data/products.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            products = [row for row in reader]
        
        self.ground_truth = {
            'businesses': businesses,
            'products': products
        }
        print(f"✓ Loaded {len(businesses)} businesses and {len(products)} products")
        return self.ground_truth
    
    def generate_query_embedding(self, query_text: str) -> np.ndarray:
        """Generate embedding for a query"""
        return self.model.encode([query_text])[0]
    
    def calculate_precision_recall(self, retrieved_ids: List[str], expected_ids: List[str], k: int) -> Tuple[float, float]:
        """Calculate precision@k and recall@k"""
        if not expected_ids:
            return 0.0, 0.0
        
        retrieved_set = set(retrieved_ids[:k])
        expected_set = set(expected_ids)
        
        true_positives = len(retrieved_set & expected_set)
        
        precision = true_positives / k if k > 0 else 0.0
        recall = true_positives / len(expected_set) if expected_set else 0.0
        
        return precision, recall
    
    def get_expected_ids(self, query: Dict) -> List[str]:
        """Get expected result IDs based on query filter"""
        expected_ids = []
        filter_dict = query.get('filter', {})
        
        if query['expected_type'] == 'business':
            for biz in self.ground_truth['businesses']:
                match = True
                for key, value in filter_dict.items():
                    if key == 'business_type' and biz.get('business_type') != value:
                        match = False
                        break
                    elif key == 'business_id' and biz.get('business_id') != value:
                        match = False
                        break
                if match:
                    expected_ids.append(f"b_{biz['business_id']}")
        
        elif query['expected_type'] == 'product':
            for prod in self.ground_truth['products']:
                match = True
                for key, value in filter_dict.items():
                    if key == 'product_id' and prod.get('product_id') != value:
                        match = False
                        break
                    elif key == 'price_min':
                        try:
                            if float(prod.get('price', 0)) < value:
                                match = False
                                break
                        except:
                            match = False
                            break
                    elif key == 'price_max':
                        try:
                            if float(prod.get('price', 0)) > value:
                                match = False
                                break
                        except:
                            match = False
                            break
                    elif key == 'quantity_min':
                        try:
                            if int(prod.get('quantity', 0)) < value:
                                match = False
                                break
                        except:
                            match = False
                            break
                    elif key == 'quantity_max':
                        try:
                            if int(prod.get('quantity', 0)) > value:
                                match = False
                                break
                        except:
                            match = False
                            break
                if match:
                    expected_ids.append(f"p_{prod['product_id']}")
        
        else:  # mixed
            # For generic queries, we consider any result valid
            expected_ids = [f"b_{b['business_id']}" for b in self.ground_truth['businesses'][:20]]
            expected_ids += [f"p_{p['product_id']}" for p in self.ground_truth['products'][:20]]
        
        return expected_ids
    
    def benchmark_qdrant(self, queries: List[Dict]) -> List[Dict]:
        """Benchmark Qdrant queries"""
        print("\n" + "="*60)
        print("BENCHMARKING QDRANT QUERIES")
        print("="*60)
        
        client = QdrantClient(host="localhost", port=6333)
        collection_name = "benchmark_collection"
        
        results = []
        
        for idx, query in enumerate(queries):
            query_text = query['query']
            query_embedding = self.generate_query_embedding(query_text)
            
            # Measure latency
            start_time = time.time()
            search_results = client.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                limit=self.k
            )
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract IDs
            retrieved_ids = []
            for result in search_results:
                payload = result.payload
                if payload.get('type') == 'business':
                    retrieved_ids.append(f"b_{payload.get('business_id')}")
                else:
                    retrieved_ids.append(f"p_{payload.get('product_id')}")
            
            # Calculate accuracy
            expected_ids = self.get_expected_ids(query)
            precision, recall = self.calculate_precision_recall(retrieved_ids, expected_ids, self.k)
            
            results.append({
                'database': 'Qdrant',
                'query_id': idx,
                'query': query_text,
                'latency_ms': round(latency_ms, 2),
                'precision@10': round(precision, 4),
                'recall@10': round(recall, 4),
                'retrieved_count': len(retrieved_ids)
            })
            
            if (idx + 1) % 20 == 0:
                print(f"  Processed {idx + 1}/{len(queries)} queries...")
        
        avg_latency = statistics.mean([r['latency_ms'] for r in results])
        avg_precision = statistics.mean([r['precision@10'] for r in results])
        avg_recall = statistics.mean([r['recall@10'] for r in results])
        
        print(f"✓ Completed - Avg latency: {avg_latency:.2f}ms, Avg precision: {avg_precision:.4f}, Avg recall: {avg_recall:.4f}")
        
        return results
    
    def benchmark_weaviate(self, queries: List[Dict]) -> List[Dict]:
        """Benchmark Weaviate queries"""
        print("\n" + "="*60)
        print("BENCHMARKING WEAVIATE QUERIES")
        print("="*60)
        
        client = weaviate.connect_to_local(
            host="localhost",
            port=8081,
            grpc_port=50051,
            skip_init_checks=True,
            additional_config=AdditionalConfig(
                timeout=Timeout(init=30, query=60, insert=120)
            )
        )
        
        results = []
        
        try:
            collection = client.collections.get("BenchmarkCollection")
            
            for idx, query in enumerate(queries):
                query_text = query['query']
                query_embedding = self.generate_query_embedding(query_text)
                
                # Measure latency
                start_time = time.time()
                search_results = collection.query.near_vector(
                    near_vector=query_embedding.tolist(),
                    limit=self.k,
                    return_properties=['business_id', 'product_id', 'item_type']
                )
                latency_ms = (time.time() - start_time) * 1000
                
                # Extract IDs
                retrieved_ids = []
                for result in search_results.objects:
                    props = result.properties
                    if props.get('item_type') == 'business':
                        retrieved_ids.append(f"b_{props.get('business_id')}")
                    else:
                        retrieved_ids.append(f"p_{props.get('product_id')}")
                
                # Calculate accuracy
                expected_ids = self.get_expected_ids(query)
                precision, recall = self.calculate_precision_recall(retrieved_ids, expected_ids, self.k)
                
                results.append({
                    'database': 'Weaviate',
                    'query_id': idx,
                    'query': query_text,
                    'latency_ms': round(latency_ms, 2),
                    'precision@10': round(precision, 4),
                    'recall@10': round(recall, 4),
                    'retrieved_count': len(retrieved_ids)
                })
                
                if (idx + 1) % 20 == 0:
                    print(f"  Processed {idx + 1}/{len(queries)} queries...")
            
            avg_latency = statistics.mean([r['latency_ms'] for r in results])
            avg_precision = statistics.mean([r['precision@10'] for r in results])
            avg_recall = statistics.mean([r['recall@10'] for r in results])
            
            print(f"✓ Completed - Avg latency: {avg_latency:.2f}ms, Avg precision: {avg_precision:.4f}, Avg recall: {avg_recall:.4f}")
            
        finally:
            client.close()
        
        return results
    
    def benchmark_chroma(self, queries: List[Dict]) -> List[Dict]:
        """Benchmark ChromaDB queries"""
        print("\n" + "="*60)
        print("BENCHMARKING CHROMADB QUERIES")
        print("="*60)
        
        try:
            client = chromadb.HttpClient(host="localhost", port=8000)
            collection = client.get_collection("benchmark_collection")
            
            results = []
            
            for idx, query in enumerate(queries):
                query_text = query['query']
                query_embedding = self.generate_query_embedding(query_text)
                
                # Measure latency
                start_time = time.time()
                search_results = collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=self.k
                )
                latency_ms = (time.time() - start_time) * 1000
                
                # Extract IDs
                retrieved_ids = search_results['ids'][0] if search_results['ids'] else []
                
                # Calculate accuracy
                expected_ids = self.get_expected_ids(query)
                precision, recall = self.calculate_precision_recall(retrieved_ids, expected_ids, self.k)
                
                results.append({
                    'database': 'ChromaDB',
                    'query_id': idx,
                    'query': query_text,
                    'latency_ms': round(latency_ms, 2),
                    'precision@10': round(precision, 4),
                    'recall@10': round(recall, 4),
                    'retrieved_count': len(retrieved_ids)
                })
                
                if (idx + 1) % 20 == 0:
                    print(f"  Processed {idx + 1}/{len(queries)} queries...")
            
            avg_latency = statistics.mean([r['latency_ms'] for r in results])
            avg_precision = statistics.mean([r['precision@10'] for r in results])
            avg_recall = statistics.mean([r['recall@10'] for r in results])
            
            print(f"✓ Completed - Avg latency: {avg_latency:.2f}ms, Avg precision: {avg_precision:.4f}, Avg recall: {avg_recall:.4f}")
            
        finally:
            try:
                del client
            except:
                pass
        
        return results
    
    def save_results(self, all_results: List[Dict]):
        """Save results to CSV"""
        print("\nSaving results...")
        
        # Save detailed results
        csv_path = '../../results/query_benchmark_detailed.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if all_results:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)
        print(f"✓ Detailed results saved to: {csv_path}")
        
        # Calculate summary statistics
        summary = []
        for db_name in ['Qdrant', 'Weaviate', 'ChromaDB']:
            db_results = [r for r in all_results if r['database'] == db_name]
            if db_results:
                summary.append({
                    'Database': db_name,
                    'Avg Latency (ms)': round(statistics.mean([r['latency_ms'] for r in db_results]), 2),
                    'Min Latency (ms)': round(min([r['latency_ms'] for r in db_results]), 2),
                    'Max Latency (ms)': round(max([r['latency_ms'] for r in db_results]), 2),
                    'Avg Precision@10': round(statistics.mean([r['precision@10'] for r in db_results]), 4),
                    'Avg Recall@10': round(statistics.mean([r['recall@10'] for r in db_results]), 4),
                    'Total Queries': len(db_results)
                })
        
        # Save summary
        summary_path = '../../results/query_benchmark_summary.csv'
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            if summary:
                writer = csv.DictWriter(f, fieldnames=summary[0].keys())
                writer.writeheader()
                writer.writerows(summary)
        print(f"✓ Summary saved to: {summary_path}")
        
        return summary
    
    def generate_charts(self, all_results: List[Dict], summary: List[Dict]):
        """Generate visualization charts"""
        print("\nGenerating charts...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Vector Database Query Performance Comparison', fontsize=16, fontweight='bold')
        
        # Chart 1: Average Latency
        databases = [s['Database'] for s in summary]
        latencies = [s['Avg Latency (ms)'] for s in summary]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        axes[0, 0].bar(databases, latencies, color=colors)
        axes[0, 0].set_title('Average Query Latency', fontweight='bold')
        axes[0, 0].set_ylabel('Latency (ms)')
        axes[0, 0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(latencies):
            axes[0, 0].text(i, v + 0.5, f'{v:.2f}ms', ha='center', va='bottom', fontweight='bold')
        
        # Chart 2: Precision vs Recall
        precisions = [s['Avg Precision@10'] for s in summary]
        recalls = [s['Avg Recall@10'] for s in summary]
        
        x = np.arange(len(databases))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, precisions, width, label='Precision@10', color='#95E1D3')
        axes[0, 1].bar(x + width/2, recalls, width, label='Recall@10', color='#F38181')
        axes[0, 1].set_title('Accuracy Metrics', fontweight='bold')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(databases)
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Chart 3: Latency Distribution (Box Plot)
        qdrant_latencies = [r['latency_ms'] for r in all_results if r['database'] == 'Qdrant']
        weaviate_latencies = [r['latency_ms'] for r in all_results if r['database'] == 'Weaviate']
        chroma_latencies = [r['latency_ms'] for r in all_results if r['database'] == 'ChromaDB']
        
        axes[1, 0].boxplot([qdrant_latencies, weaviate_latencies, chroma_latencies],
                          labels=databases,
                          patch_artist=True,
                          boxprops=dict(facecolor='lightblue'))
        axes[1, 0].set_title('Latency Distribution', fontweight='bold')
        axes[1, 0].set_ylabel('Latency (ms)')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Chart 4: Latency vs Accuracy Scatter
        for db_name, color in zip(['Qdrant', 'Weaviate', 'ChromaDB'], colors):
            db_results = [r for r in all_results if r['database'] == db_name]
            latencies_scatter = [r['latency_ms'] for r in db_results]
            precisions_scatter = [r['precision@10'] for r in db_results]
            axes[1, 1].scatter(latencies_scatter, precisions_scatter, alpha=0.5, label=db_name, color=color, s=30)
        
        axes[1, 1].set_title('Latency vs Precision Trade-off', fontweight='bold')
        axes[1, 1].set_xlabel('Latency (ms)')
        axes[1, 1].set_ylabel('Precision@10')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        chart_path = '../../results/query_benchmark_charts.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"✓ Charts saved to: {chart_path}")
        plt.close()
    
    def print_summary(self, summary: List[Dict]):
        """Print summary table"""
        print("\n" + "="*100)
        print("QUERY BENCHMARK SUMMARY")
        print("="*100)
        print(f"{'Database':<15} {'Avg Latency':<15} {'Min Latency':<15} {'Max Latency':<15} {'Precision@10':<15} {'Recall@10':<15}")
        print("-"*100)
        
        for row in summary:
            print(f"{row['Database']:<15} "
                  f"{row['Avg Latency (ms)']:<15} "
                  f"{row['Min Latency (ms)']:<15} "
                  f"{row['Max Latency (ms)']:<15} "
                  f"{row['Avg Precision@10']:<15} "
                  f"{row['Avg Recall@10']:<15}")
        
        print("="*100)
    
    def run_benchmarks(self):
        """Run all benchmarks"""
        print("Starting Query Benchmark Suite")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Load data
        self.load_queries()
        self.load_ground_truth()
        
        all_results = []
        
        # Benchmark each database
        try:
            qdrant_results = self.benchmark_qdrant(self.queries)
            all_results.extend(qdrant_results)
        except Exception as e:
            print(f"✗ Qdrant benchmark failed: {e}")
        
        try:
            weaviate_results = self.benchmark_weaviate(self.queries)
            all_results.extend(weaviate_results)
        except Exception as e:
            print(f"✗ Weaviate benchmark failed: {e}")
        
        try:
            chroma_results = self.benchmark_chroma(self.queries)
            all_results.extend(chroma_results)
        except Exception as e:
            print(f"✗ ChromaDB benchmark failed: {e}")
        
        # Save and visualize results
        if all_results:
            summary = self.save_results(all_results)
            self.generate_charts(all_results, summary)
            self.print_summary(summary)
        else:
            print("✗ No results to save!")


if __name__ == "__main__":
    benchmark = QueryBenchmark()
    benchmark.run_benchmarks()
