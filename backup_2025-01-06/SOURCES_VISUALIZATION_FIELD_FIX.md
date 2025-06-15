# Sources Tab Visualization Field Name Fix

## Problem
The error `unsupported format string passed to list.__format__` was occurring because the visualization functions were using incorrect field names that didn't match what the data fetchers returned.

## Root Cause
Mismatch between expected field names in visualization functions and actual field names returned by data fetchers.

## Fixes Applied

### 1. Keywords Visualization (`create_keywords_visualizations`)
Fixed field name mismatches:
- `unique_keywords` → `total_unique_keywords`
- `total_occurrences` → `total_keyword_occurrences`
- `top_keywords['keywords']` → `top_keywords['labels']`
- `top_keywords['counts']` → `top_keywords['values']`
- `coverage_rate` → `keyword_coverage`
- `avg_per_chunk` → `avg_keywords_per_chunk`

### 2. Named Entities Visualization (`create_entities_visualizations`)
Fixed field name mismatches:
- `unique_entities` → `total_unique_entities`
- `total_occurrences` → `total_entity_occurrences`
- `top_entities['entities']` → `top_entities['labels']`
- `top_entities['counts']` → `top_entities['values']`
- Fixed entity_types structure to use arrays instead of dict

### 3. Added Better Error Logging
Added traceback logging to help debug future issues:
```python
import traceback
logging.error(f"Traceback: {traceback.format_exc()}")
```

## Data Structure Reference

### Keywords Data Structure
```python
{
    "total_unique_keywords": int,
    "total_keyword_occurrences": int,
    "chunks_with_keywords": int,
    "keyword_coverage": float,
    "avg_keywords_per_chunk": float,
    "top_keywords": {
        "labels": [...],
        "values": [...]
    }
}
```

### Named Entities Data Structure  
```python
{
    "total_unique_entities": int,
    "total_entity_occurrences": int,
    "entity_types": {
        "labels": [...],
        "counts": [...],
        "unique_entities": [...]
    },
    "top_entities": {
        "labels": [...],
        "values": [...],
        "types": [...]
    }
}
```

## Result
All visualizations now display correctly without format string errors.