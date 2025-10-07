# Multi-Tenant Vector Database Benchmark

This benchmark tests the multi-tenant capabilities of Qdrant, Weaviate, and ChromaDB, measuring performance, isolation, and resource usage when handling multiple separate tenants.

## Overview

### What is Multi-Tenancy?

Multi-tenancy allows a single database instance to serve multiple independent customers (tenants) while ensuring:

- **Data Isolation**: Each tenant's data remains completely separate
- **No Cross-Tenant Leakage**: Queries from one tenant never return another tenant's data
- **Scalable Performance**: System handles many tenants efficiently

## Test Configuration

- **Number of Tenants**: 1,000 businesses (tested with 10 for quick validation)
- **Products per Tenant**: 500 products each
- **Total Data**: 500,000 products across all tenants
- **Queries per Tenant**: 10 queries each
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)

## Multi-Tenancy Approaches

### Qdrant

- **Strategy**: Separate collections per tenant
- **Implementation**: Each tenant gets its own dedicated collection (e.g., `tenant_B0001`)
- **Pros**: Complete physical isolation, predictable performance
- **Cons**: More collections to manage at scale

### Weaviate

- **Strategy**: Native multi-tenancy with single collection
- **Implementation**: One collection with built-in tenant partitioning
- **Pros**: Efficient resource usage, designed for multi-tenancy
- **Cons**: Requires multi-tenancy configuration

### ChromaDB

- **Strategy**: Separate collections per tenant
- **Implementation**: Each tenant gets its own collection (e.g., `tenant_B0001`)
- **Pros**: Simple isolation model
- **Cons**: More collections at scale

## Quick Start

### 1. Generate Multi-Tenant Data

```powershell
cd benchmarks/multi_tenant
python generate_multitenant_data.py
```

This creates:

- 1,000 businesses
- 500,000 products (500 per business)
- Files saved to `data/multitenant_*.csv`

### 2. Run Quick Test (10 tenants)

```powershell
python benchmark_multitenant_test.py
```

### 3. Run Full Benchmark (1,000 tenants)

```powershell
python benchmark_multitenant.py
```

**Note**: Full benchmark with 1,000 tenants will take several hours!

### 4. Generate Visualizations

```powershell
python visualize_results.py
```

## Test Results (10 Tenants, 500 Products Each)

### Performance Summary

| Database     | Insertion Time | Query Latency | Memory Usage | Leakage |
| ------------ | -------------- | ------------- | ------------ | ------- |
| **Qdrant**   | 102.37s        | 10.01ms       | 409 MB       | **0** ✓ |
| **Weaviate** | 153.22s        | **5.14ms** ⚡ | 334 MB       | **0** ✓ |
| **ChromaDB** | 107.45s        | 54.03ms       | 451 MB       | **0** ✓ |

### Key Findings

✅ **Zero Cross-Tenant Leakage**: All three databases achieved perfect isolation

- 0 out of 100 queries per database leaked data between tenants
- Leakage rate: 0.0000% for all databases

⚡ **Fastest Queries**: Weaviate (5.14ms average)

- 49% faster than Qdrant
- 90% faster than ChromaDB

🚀 **Fastest Insertion**: Qdrant (102.37s total)

- 33% faster than Weaviate
- 5% faster than ChromaDB

💾 **Lowest Memory**: Weaviate (334 MB average)

- 18% less than Qdrant
- 26% less than ChromaDB

## Metrics Explained

### Insertion Metrics

- **Total Insertion Time**: Time to insert all products for all tenants
- **Avg Insertion Time/Tenant**: Average time to insert 500 products for one tenant
- **Memory Usage**: Average container memory during insertion

### Query Metrics

- **Avg Query Latency**: Mean time to execute a query (milliseconds)
- **Min/Max Latency**: Range of query response times
- **Std Latency**: Standard deviation of latencies (consistency measure)

### Isolation Metrics

- **Cross-Tenant Leakage**: Number of times a query returned data from wrong tenant
- **Leakage Rate**: Percentage of queries that leaked data
- **Target**: 0.0000% (perfect isolation)

## Generated Files

### Data Files

- `data/multitenant_businesses.csv` - 1,000 businesses
- `data/multitenant_products.csv` - 500,000 products
- `data/multitenant_summary.json` - Data generation summary

### Results Files

- `results/multitenant_benchmark_results.json` - Detailed JSON results
- `results/multitenant_benchmark_summary.csv` - Summary table
- `results/multitenant_benchmark_charts.png` - 6-panel visualization

## Visualization Charts

The benchmark generates 6 charts:

1. **Total Insertion Time** - Time to load all tenant data
2. **Average Query Latency** - Query response time comparison
3. **Average Memory Usage** - Container memory consumption
4. **Latency Distribution** - Min/Avg/Max query latencies
5. **Cross-Tenant Leakage** - Isolation verification (should be 0)
6. **Avg Insertion Time/Tenant** - Per-tenant insertion performance

## Use Cases

### When to Use Multi-Tenancy

✅ **SaaS Applications**: Serving multiple customers from single infrastructure
✅ **Enterprise Deployments**: Separate departments with isolated data
✅ **Development/Staging**: Multiple environments on one instance
✅ **Cost Optimization**: Share resources across tenants

### Single vs Multi-Tenant Comparison

| Aspect               | Single-Tenant                  | Multi-Tenant                |
| -------------------- | ------------------------------ | --------------------------- |
| **Isolation**        | Complete (separate instances)  | Logical (same instance)     |
| **Cost**             | High (one instance per tenant) | Low (shared infrastructure) |
| **Management**       | Complex (many instances)       | Simple (one instance)       |
| **Latency Overhead** | None                           | Minimal (tenant routing)    |

## Performance Tips

### For Better Insertion Performance

1. **Batch Size**: Increase batch sizes for bulk inserts
2. **Concurrent Inserts**: Insert multiple tenants in parallel
3. **Pre-compute Embeddings**: Generate embeddings before insertion

### For Better Query Performance

1. **Tenant Routing**: Ensure queries specify correct tenant
2. **Index Optimization**: Configure indexes per workload
3. **Caching**: Cache frequently accessed tenant data

### For Scalability

1. **Monitor Collection Count**: Track number of collections (Qdrant, ChromaDB)
2. **Resource Allocation**: Allocate sufficient CPU/memory for peak tenants
3. **Data Distribution**: Balance tenant sizes for even resource usage

## Troubleshooting

### High Memory Usage

- Reduce batch sizes during insertion
- Increase Docker container memory limits
- Test with fewer tenants first

### Slow Queries

- Check tenant-specific indexes
- Verify correct tenant routing
- Monitor resource contention

### Cross-Tenant Leakage Detected

- Verify tenant ID filtering in queries
- Check collection/tenant names match exactly
- Review isolation implementation

## Security Considerations

1. **Tenant ID Validation**: Always validate tenant IDs in application layer
2. **Access Control**: Implement authentication per tenant
3. **Audit Logging**: Log all cross-tenant access attempts
4. **Data Encryption**: Encrypt sensitive tenant data at rest

## Scaling Beyond 1,000 Tenants

For production deployments with >1,000 tenants:

1. **Qdrant**: Consider collection limits and index organization
2. **Weaviate**: Leverage native multi-tenancy for efficiency
3. **ChromaDB**: Monitor collection count and resource usage
4. **All**: Implement tenant sharding across multiple instances

## Benchmark Limitations

- **Test Data**: Synthetic data may not match production patterns
- **Query Patterns**: Random queries may differ from real workloads
- **Resource Contention**: Single machine limits concurrent performance
- **Network Latency**: Local Docker containers eliminate network overhead

## Contributing

Improvements welcome:

- Test with different embedding models
- Add more vector databases (Pinecone, Milvus)
- Implement concurrent query testing
- Test with unbalanced tenant sizes
- Add data update/delete operations

---

**Last Updated**: October 2025  
**Test Environment**: Windows, Docker Desktop, Python 3.12  
**Embedding Model**: all-MiniLM-L6-v2 (384d)  
**Total Test Records**: 500,000 products across 1,000 tenants
