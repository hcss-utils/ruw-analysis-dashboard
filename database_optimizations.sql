-- Database Optimization Script for Named Entities Performance
-- Run these commands with database admin privileges
-- Estimated time: 10-30 minutes depending on data size

-- 1. Add GIN index on named_entities JSONB column
-- This speeds up queries that search within the JSON structure
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunk_named_entities_gin 
ON document_section_chunk USING GIN (named_entities);

-- 2. Add partial index for chunks with named entities
-- This helps quickly filter chunks that have entity data
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunk_has_entities 
ON document_section_chunk (id) 
WHERE named_entities IS NOT NULL 
  AND jsonb_typeof(named_entities) = 'array' 
  AND jsonb_array_length(named_entities) > 0;

-- 3. Add composite index for the join with taxonomy
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_taxonomy_chunk_id 
ON taxonomy (chunk_id);

-- 4. Create materialized view for entity statistics
-- This pre-computes expensive aggregations
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_entity_stats AS
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
    DATE_TRUNC('month', date) as month,
    COUNT(*) as occurrence_count
FROM entity_data
GROUP BY entity_text, entity_type, language, database, source_type, DATE_TRUNC('month', date);

-- 5. Add indexes on the materialized view
CREATE INDEX IF NOT EXISTS idx_mv_entity_stats_text 
ON mv_entity_stats (entity_text);

CREATE INDEX IF NOT EXISTS idx_mv_entity_stats_type 
ON mv_entity_stats (entity_type);

CREATE INDEX IF NOT EXISTS idx_mv_entity_stats_filters 
ON mv_entity_stats (language, database, source_type, month);

CREATE INDEX IF NOT EXISTS idx_mv_entity_stats_count 
ON mv_entity_stats (occurrence_count DESC);

-- 6. Create a refresh function for the materialized view
CREATE OR REPLACE FUNCTION refresh_entity_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_entity_stats;
END;
$$ LANGUAGE plpgsql;

-- 7. Analyze tables to update statistics
ANALYZE document_section_chunk;
ANALYZE taxonomy;
ANALYZE mv_entity_stats;

-- To refresh the materialized view (run periodically, e.g., daily):
-- SELECT refresh_entity_stats();

-- To check index usage:
-- SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
-- FROM pg_stat_user_indexes
-- WHERE tablename IN ('document_section_chunk', 'taxonomy', 'mv_entity_stats')
-- ORDER BY idx_scan DESC;