# Sources Tab Performance Optimization Guide

## Overview

The Sources tab has been experiencing performance issues with queries taking over 30 seconds and timing out on Heroku. This guide provides optimized queries and database changes to reduce query times to under 5 seconds each.

## Key Performance Issues Identified

1. **Multiple Complex JOINs**: Each query joins 3-4 tables multiple times
2. **Large JSONB Operations**: Processing keywords and named entities from JSONB arrays is expensive
3. **Real-time Aggregations**: All statistics are computed on-demand without caching
4. **Full Table Scans**: No sampling used for large datasets
5. **Missing Indexes**: Key foreign keys and filter columns lack indexes

## Optimization Strategy

### 1. Database Optimizations (One-time setup)

Run the SQL commands in `database_sources_optimization.sql`:

```bash
psql -U your_user -d your_database -f database_sources_optimization.sql
```

This creates:
- Optimized indexes on foreign keys and filter columns
- Materialized views for pre-aggregated statistics
- Helper functions for view refresh

### 2. Use Optimized Query Functions

Replace imports in your Sources tab code:

```python
# Old import
from database.data_fetchers_sources import (
    fetch_documents_data,
    fetch_chunks_data,
    fetch_keywords_data,
    fetch_named_entities_data,
    fetch_taxonomy_combinations
)

# New import
from database.data_fetchers_sources_optimized import (
    fetch_documents_data,
    fetch_chunks_data,
    fetch_keywords_data,
    fetch_named_entities_data,
    fetch_taxonomy_combinations
)
```

### 3. Key Optimizations Implemented

#### Documents Data (`fetch_documents_data`)
- **Before**: Multiple passes through data with complex JOINs
- **After**: Single temp table creation, then fast queries against it
- **Expected speedup**: 5-10x

#### Chunks Data (`fetch_chunks_data`)
- **Before**: Separate queries for each statistic
- **After**: Single comprehensive query with conditional aggregation
- **Expected speedup**: 3-5x

#### Keywords Data (`fetch_keywords_data`)
- **Before**: Full JSONB expansion for all chunks
- **After**: Sampling approach (5,000 chunks) with extrapolation
- **Expected speedup**: 10-20x

#### Named Entities Data (`fetch_named_entities_data`)
- **Before**: Full JSONB processing with multiple CTEs
- **After**: Sampling approach (3,000 chunks) with estimation
- **Expected speedup**: 10-20x

#### Taxonomy Combinations (`fetch_taxonomy_combinations`)
- **Before**: Full aggregation of all combinations
- **After**: Top 30 combinations only
- **Expected speedup**: 5-10x

### 4. Materialized View Maintenance

Set up a daily refresh job to keep materialized views current:

```bash
# Add to crontab
0 2 * * * psql -U your_user -d your_database -c "SELECT refresh_sources_materialized_views();"
```

Or manually refresh when needed:
```sql
SELECT refresh_sources_materialized_views();
```

### 5. Monitoring Performance

Track query performance with the built-in logging:

```sql
-- View average query times
SELECT 
    query_name,
    COUNT(*) as execution_count,
    AVG(execution_time_ms) as avg_time_ms,
    MAX(execution_time_ms) as max_time_ms
FROM query_performance_log
WHERE created_at > CURRENT_DATE - INTERVAL '7 days'
GROUP BY query_name
ORDER BY avg_time_ms DESC;
```

## Expected Results

After implementing these optimizations:

1. **Corpus Stats**: < 1 second (from ~5 seconds)
2. **Documents Data**: < 3 seconds (from ~15 seconds)
3. **Chunks Data**: < 2 seconds (from ~10 seconds)
4. **Keywords Data**: < 3 seconds (from ~30+ seconds)
5. **Named Entities Data**: < 3 seconds (from ~30+ seconds)
6. **Taxonomy Combinations**: < 2 seconds (from ~10 seconds)

**Total Sources tab load time**: < 15 seconds (from > 90 seconds)

## Rollback Plan

If issues arise, simply revert to the original import:

```python
# Revert to original
from database.data_fetchers_sources import ...
```

The optimized version maintains the same API, so no other code changes are needed.

## Trade-offs

1. **Sampling for Keywords/Entities**: Results are estimates based on samples, not exact counts
2. **Materialized Views**: Data is slightly stale (up to 24 hours old)
3. **Limited Results**: Taxonomy combinations limited to top 30

These trade-offs are acceptable for a dashboard where approximate statistics are sufficient and performance is critical.

## Next Steps

1. Test the optimized queries in a staging environment
2. Run the database optimization SQL
3. Deploy the optimized query functions
4. Monitor performance improvements
5. Adjust sampling sizes if needed for better accuracy