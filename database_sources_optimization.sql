-- Database Optimization Script for Sources Tab Performance
-- Run these commands with database admin privileges
-- Estimated time: 15-45 minutes depending on data size

-- =====================================================
-- 1. INDEXES FOR BETTER JOIN PERFORMANCE
-- =====================================================

-- Index for taxonomy table chunk_id (critical for relevance checks)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_taxonomy_chunk_id 
ON taxonomy (chunk_id);

-- Composite index for uploaded_document filtering
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_uploaded_document_filters 
ON uploaded_document (language, database, source_type, date);

-- Index for document_section foreign key
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_section_doc_id 
ON document_section (uploaded_document_id);

-- Index for document_section_chunk foreign key
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_section_chunk_section_id 
ON document_section_chunk (document_section_id);

-- Partial indexes for JSONB columns (keywords and entities)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunk_has_keywords 
ON document_section_chunk (id) 
WHERE keywords_llm IS NOT NULL 
  AND jsonb_typeof(keywords_llm) = 'array' 
  AND jsonb_array_length(keywords_llm) > 0;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunk_has_entities 
ON document_section_chunk (id) 
WHERE named_entities IS NOT NULL 
  AND jsonb_typeof(named_entities) = 'array' 
  AND jsonb_array_length(named_entities) > 0;

-- =====================================================
-- 2. MATERIALIZED VIEW FOR DOCUMENT STATISTICS
-- =====================================================

DROP MATERIALIZED VIEW IF EXISTS mv_document_stats CASCADE;

CREATE MATERIALIZED VIEW mv_document_stats AS
WITH doc_relevance AS (
    SELECT 
        ud.id,
        ud.language,
        ud.database,
        ud.source_type,
        ud.date,
        CASE WHEN EXISTS (
            SELECT 1 
            FROM document_section ds
            JOIN document_section_chunk dsc ON ds.id = dsc.document_section_id
            JOIN taxonomy t ON dsc.id = t.chunk_id
            WHERE ds.uploaded_document_id = ud.id
        ) THEN 1 ELSE 0 END as is_relevant
    FROM uploaded_document ud
)
SELECT 
    language,
    database,
    source_type,
    DATE_TRUNC('week', date) as week,
    DATE_TRUNC('month', date) as month,
    COUNT(*) as doc_count,
    SUM(is_relevant) as relevant_count
FROM doc_relevance
GROUP BY language, database, source_type, DATE_TRUNC('week', date), DATE_TRUNC('month', date);

-- Indexes on the materialized view
CREATE INDEX idx_mv_document_stats_language ON mv_document_stats (language);
CREATE INDEX idx_mv_document_stats_database ON mv_document_stats (database);
CREATE INDEX idx_mv_document_stats_source_type ON mv_document_stats (source_type);
CREATE INDEX idx_mv_document_stats_week ON mv_document_stats (week);
CREATE INDEX idx_mv_document_stats_composite ON mv_document_stats (language, database, source_type);

-- =====================================================
-- 3. MATERIALIZED VIEW FOR CHUNK STATISTICS
-- =====================================================

DROP MATERIALIZED VIEW IF EXISTS mv_chunk_stats CASCADE;

CREATE MATERIALIZED VIEW mv_chunk_stats AS
WITH chunk_data AS (
    SELECT 
        dsc.id,
        ud.language,
        ud.database,
        ud.source_type,
        ud.date,
        LENGTH(dsc.content) as chunk_length,
        CASE WHEN EXISTS (
            SELECT 1 FROM taxonomy t WHERE t.chunk_id = dsc.id
        ) THEN 1 ELSE 0 END as is_relevant,
        CASE 
            WHEN dsc.keywords_llm IS NOT NULL 
                AND jsonb_typeof(dsc.keywords_llm) = 'array' 
                AND jsonb_array_length(dsc.keywords_llm) > 0 
            THEN jsonb_array_length(dsc.keywords_llm) 
            ELSE 0 
        END as keyword_count,
        CASE 
            WHEN dsc.named_entities IS NOT NULL 
                AND jsonb_typeof(dsc.named_entities) = 'array' 
                AND jsonb_array_length(dsc.named_entities) > 0 
            THEN jsonb_array_length(dsc.named_entities) 
            ELSE 0 
        END as entity_count
    FROM document_section_chunk dsc
    JOIN document_section ds ON dsc.document_section_id = ds.id
    JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
)
SELECT 
    language,
    database,
    source_type,
    DATE_TRUNC('month', date) as month,
    COUNT(*) as chunk_count,
    SUM(is_relevant) as relevant_count,
    AVG(chunk_length) as avg_chunk_length,
    AVG(CASE WHEN is_relevant = 1 THEN chunk_length END) as avg_relevant_chunk_length,
    SUM(CASE WHEN keyword_count > 0 THEN 1 ELSE 0 END) as chunks_with_keywords,
    SUM(CASE WHEN entity_count > 0 THEN 1 ELSE 0 END) as chunks_with_entities,
    AVG(keyword_count) as avg_keywords_per_chunk,
    AVG(entity_count) as avg_entities_per_chunk
FROM chunk_data
GROUP BY language, database, source_type, DATE_TRUNC('month', date);

-- Indexes on the chunk stats materialized view
CREATE INDEX idx_mv_chunk_stats_language ON mv_chunk_stats (language);
CREATE INDEX idx_mv_chunk_stats_database ON mv_chunk_stats (database);
CREATE INDEX idx_mv_chunk_stats_source_type ON mv_chunk_stats (source_type);
CREATE INDEX idx_mv_chunk_stats_composite ON mv_chunk_stats (language, database, source_type);

-- =====================================================
-- 4. MATERIALIZED VIEW FOR TAXONOMY COMBINATIONS
-- =====================================================

DROP MATERIALIZED VIEW IF EXISTS mv_taxonomy_combinations CASCADE;

CREATE MATERIALIZED VIEW mv_taxonomy_combinations AS
SELECT 
    t.category,
    t.subcategory,
    t.sub_subcategory,
    ud.language,
    ud.database,
    ud.source_type,
    DATE_TRUNC('month', ud.date) as month,
    COUNT(DISTINCT t.chunk_id) as chunk_count,
    COUNT(DISTINCT ud.id) as doc_count
FROM taxonomy t
JOIN document_section_chunk dsc ON t.chunk_id = dsc.id
JOIN document_section ds ON dsc.document_section_id = ds.id
JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
GROUP BY 
    t.category, 
    t.subcategory, 
    t.sub_subcategory,
    ud.language,
    ud.database,
    ud.source_type,
    DATE_TRUNC('month', ud.date);

-- Indexes on taxonomy combinations
CREATE INDEX idx_mv_taxonomy_combinations_category ON mv_taxonomy_combinations (category);
CREATE INDEX idx_mv_taxonomy_combinations_subcategory ON mv_taxonomy_combinations (subcategory);
CREATE INDEX idx_mv_taxonomy_combinations_filters ON mv_taxonomy_combinations (language, database, source_type);
CREATE INDEX idx_mv_taxonomy_combinations_counts ON mv_taxonomy_combinations (chunk_count DESC, doc_count DESC);

-- =====================================================
-- 5. HELPER FUNCTIONS FOR MATERIALIZED VIEW REFRESH
-- =====================================================

-- Function to refresh all materialized views
CREATE OR REPLACE FUNCTION refresh_sources_materialized_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_document_stats;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_chunk_stats;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_taxonomy_combinations;
    
    -- Also refresh the entity stats view if it exists
    IF EXISTS (SELECT 1 FROM pg_matviews WHERE matviewname = 'mv_entity_stats') THEN
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_entity_stats;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- 6. QUERY PERFORMANCE TRACKING
-- =====================================================

-- Create a table to track query performance
CREATE TABLE IF NOT EXISTS query_performance_log (
    id SERIAL PRIMARY KEY,
    query_name VARCHAR(100),
    execution_time_ms INTEGER,
    filters_used JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Function to log query performance
CREATE OR REPLACE FUNCTION log_query_performance(
    p_query_name VARCHAR(100),
    p_start_time TIMESTAMP,
    p_filters JSONB DEFAULT '{}'::JSONB
)
RETURNS void AS $$
DECLARE
    v_execution_time_ms INTEGER;
BEGIN
    v_execution_time_ms := EXTRACT(MILLISECONDS FROM (CURRENT_TIMESTAMP - p_start_time));
    
    INSERT INTO query_performance_log (query_name, execution_time_ms, filters_used)
    VALUES (p_query_name, v_execution_time_ms, p_filters);
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- 7. MAINTENANCE COMMANDS
-- =====================================================

-- Update table statistics for query planner
ANALYZE uploaded_document;
ANALYZE document_section;
ANALYZE document_section_chunk;
ANALYZE taxonomy;

-- Check index usage (run this query to verify indexes are being used)
/*
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as index_scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
    AND tablename IN ('uploaded_document', 'document_section', 'document_section_chunk', 'taxonomy')
ORDER BY idx_scan DESC;
*/

-- Check materialized view sizes
/*
SELECT 
    schemaname,
    matviewname,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||matviewname)) as total_size
FROM pg_matviews
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||matviewname) DESC;
*/

-- =====================================================
-- 8. SCHEDULED REFRESH (Example cron job command)
-- =====================================================

-- To set up automatic refresh (run daily at 2 AM):
-- Add this to your crontab or use pg_cron extension:
-- 0 2 * * * psql -U your_user -d your_database -c "SELECT refresh_sources_materialized_views();"

-- =====================================================
-- 9. PERFORMANCE MONITORING QUERIES
-- =====================================================

-- View average query times by query name
/*
SELECT 
    query_name,
    COUNT(*) as execution_count,
    AVG(execution_time_ms) as avg_time_ms,
    MIN(execution_time_ms) as min_time_ms,
    MAX(execution_time_ms) as max_time_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY execution_time_ms) as p95_time_ms
FROM query_performance_log
WHERE created_at > CURRENT_DATE - INTERVAL '7 days'
GROUP BY query_name
ORDER BY avg_time_ms DESC;
*/

-- View slow queries (over 5 seconds)
/*
SELECT 
    query_name,
    execution_time_ms,
    filters_used,
    created_at
FROM query_performance_log
WHERE execution_time_ms > 5000
ORDER BY created_at DESC
LIMIT 100;
*/