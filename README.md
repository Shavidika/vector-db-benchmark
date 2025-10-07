# Vector Database Benchmarking Project

This project benchmarks the ingestion performance of three popular vector databases: **Qdrant**, **Weaviate**, and **ChromaDB** using synthetic business and product data.

## Overview

The benchmark measures:

1. **Insertion throughput** (records per second)
2. **Total ingestion time**
3. **Index build time**
4. **Final storage size on disk**

All databases use consistent vector embeddings generated from `sentence-transformers/all-MiniLM-L6-v2` model.

## Project Structure

```
vector-db-benchmark/
├── benchmark_vector_db_ingestion.py  # Main benchmarking script
├── docker-compose.yml                # Docker configuration for vector DBs
├── requirements.txt                  # Python dependencies
├── businesses.csv                    # Synthetic business data
├── businesses.json                   # Synthetic business data (JSON)
├── products.csv                      # Synthetic product data
├── products.json                     # Synthetic product data (JSON)
└── benchmark_results.txt            # Benchmark results output
```

## Prerequisites

- Docker Desktop installed and running
- Python 3.8+ with virtual environment
- At least 4GB of free RAM

## Setup Instructions

### 1. Start Vector Databases in Docker

```powershell
docker-compose up -d
```

This will start:

- **Qdrant** on ports 6333 (HTTP) and 6334 (gRPC)
- **Weaviate** on ports 8081 (HTTP) and 50051 (gRPC)
- **ChromaDB** on port 8000

### 2. Install Python Dependencies

```powershell
# Activate virtual environment (if using one)
.\vector_db_env\Scripts\Activate.ps1

# Install required packages
pip install -r requirements.txt
```

### 3. Run the Benchmark

```powershell
python benchmark_vector_db_ingestion.py
```

## Benchmark Results

### Latest Test Results (2025-10-07)

| Database | Records | Ingestion Time (s) | Index Time (s) | Throughput (rec/s) |
| -------- | ------- | ------------------ | -------------- | ------------------ |
| Qdrant   | 12,173  | 12.39              | 0.25           | 982.43             |
| Weaviate | 12,173  | 7.14               | 0.22           | **1,704.00**       |
| ChromaDB | 12,173  | 15.78              | 0.01           | 771.44             |

### Key Findings

- **Weaviate** showed the highest throughput at 1,704 records/second
- **ChromaDB** had the fastest index build time (0.01s)
- **Qdrant** provided balanced performance with 982.43 records/second
- All databases successfully ingested 12,173 records (100 businesses + 12,073 products)

## Data Schema

### Businesses

- `business_id`: Unique identifier
- `business_name`: Company name
- `email`: Contact email
- `business_type`: Category (transport, online retail, hotel)
- `branches`: Pipe-separated list of branch addresses

### Products

- `product_id`: Unique identifier
- `product_name`: Product name
- `quantity`: Stock quantity
- `price`: Product price
- `business_id`: Foreign key to business

## Vector Embeddings

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension**: 384
- **Text Representation**:
  - Businesses: `{business_name} {email} {business_type}`
  - Products: `{product_name} quantity: {quantity} price: {price}`

## Stopping the Services

To stop and remove all Docker containers:

```powershell
docker-compose down
```

To stop and remove containers along with volumes (deletes all data):

```powershell
docker-compose down -v
```

## Troubleshooting

### Port Conflicts

If you encounter port conflicts:

1. Edit `docker-compose.yml` to use different ports
2. Update the port numbers in `benchmark_vector_db_ingestion.py` accordingly

### Container Not Starting

Check container logs:

```powershell
docker-compose logs [service-name]
# Example: docker-compose logs weaviate
```

### Resource Warnings

Some minor resource warnings from ChromaDB client are expected and don't affect benchmark results.

## Technical Details

### Batch Sizes

- **Qdrant**: 100 records per batch
- **Weaviate**: Dynamic batching
- **ChromaDB**: 5,000 records per batch

### Distance Metrics

All databases use **Cosine similarity** for vector search.

## License

This is a benchmarking project for educational purposes.

## Contributing

Feel free to extend this benchmark with:

- Additional vector databases
- Different embedding models
- Query performance tests
- Larger datasets
- Memory usage metrics
