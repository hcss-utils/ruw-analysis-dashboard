# Named Entities Query Optimization

## Current Issue
The named entities query takes over 100 seconds because it's using `jsonb_array_elements` on a large dataset without proper indexing.

## Immediate Fix (Applied)
Added a LIMIT clause to the most expensive part of the query to prevent timeout while still providing useful data.

## Recommended Database Optimizations

### 1. Add GIN Index on named_entities column
```sql
CREATE INDEX idx_chunk_named_entities_gin ON document_section_chunk 
USING GIN (named_entities);
```

### 2. Add composite index for the join conditions
```sql
CREATE INDEX idx_chunk_taxonomy_join ON document_section_chunk (id)
WHERE named_entities IS NOT NULL;
```

### 3. Consider materialized view for entity statistics
```sql
CREATE MATERIALIZED VIEW mv_entity_stats AS
WITH entity_data AS (
    SELECT 
        dsc.id as chunk_id,
        ud.language,
        ud.database,
        ud.source_type,
        ud.date,
        jsonb_array_elements(dsc.named_entities) as entity
    FROM document_section_chunk dsc
    JOIN document_section ds ON dsc.document_section_id = ds.id
    JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
    INNER JOIN taxonomy t ON dsc.id = t.chunk_id
    WHERE dsc.named_entities IS NOT NULL 
        AND jsonb_typeof(dsc.named_entities) = 'array'
        AND jsonb_array_length(dsc.named_entities) > 0
)
SELECT 
    entity->>'text' as entity_text,
    entity->>'label' as entity_type,
    language,
    database,
    source_type,
    date,
    COUNT(*) as occurrence_count
FROM entity_data
GROUP BY entity_text, entity_type, language, database, source_type, date;

CREATE INDEX idx_mv_entity_stats_text ON mv_entity_stats (entity_text);
CREATE INDEX idx_mv_entity_stats_type ON mv_entity_stats (entity_type);
CREATE INDEX idx_mv_entity_stats_filters ON mv_entity_stats (language, database, source_type, date);
```

Then refresh periodically:
```sql
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_entity_stats;
```

## Alternative: Pre-aggregate Entity Data
Consider creating a separate table for entity statistics that gets updated when documents are processed:

```sql
CREATE TABLE entity_statistics (
    entity_text TEXT,
    entity_type TEXT,
    language TEXT,
    database TEXT,
    source_type TEXT,
    occurrence_count INTEGER,
    first_seen DATE,
    last_seen DATE,
    PRIMARY KEY (entity_text, entity_type, language, database, source_type)
);
```

This would make queries instant but requires updating the data ingestion pipeline.