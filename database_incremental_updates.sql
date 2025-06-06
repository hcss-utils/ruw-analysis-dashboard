-- Alternative: Incremental Update Approach for Daily Data
-- This approach updates statistics incrementally instead of full refresh

-- 1. Create entity statistics table (instead of materialized view)
CREATE TABLE IF NOT EXISTS entity_statistics (
    entity_text TEXT,
    entity_type TEXT,
    language TEXT,
    database TEXT,
    source_type TEXT,
    month DATE,
    occurrence_count INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (entity_text, entity_type, language, database, source_type, month)
);

-- 2. Create indexes
CREATE INDEX idx_entity_stats_text ON entity_statistics (entity_text);
CREATE INDEX idx_entity_stats_type ON entity_statistics (entity_type);
CREATE INDEX idx_entity_stats_filters ON entity_statistics (language, database, source_type, month);
CREATE INDEX idx_entity_stats_updated ON entity_statistics (last_updated);

-- 3. Create function to update statistics for new documents
CREATE OR REPLACE FUNCTION update_entity_statistics_for_documents(doc_ids INTEGER[])
RETURNS void AS $$
BEGIN
    -- Insert or update entity statistics for specified documents
    INSERT INTO entity_statistics (
        entity_text, entity_type, language, database, source_type, month, occurrence_count
    )
    SELECT 
        entity->>'text' as entity_text,
        entity->>'label' as entity_type,
        ud.language,
        ud.database,
        ud.source_type,
        DATE_TRUNC('month', ud.date) as month,
        COUNT(*) as occurrence_count
    FROM document_section_chunk dsc
    JOIN document_section ds ON dsc.document_section_id = ds.id
    JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
    CROSS JOIN LATERAL jsonb_array_elements(dsc.named_entities) as entity
    WHERE ud.id = ANY(doc_ids)
        AND dsc.named_entities IS NOT NULL
        AND jsonb_typeof(dsc.named_entities) = 'array'
    GROUP BY entity_text, entity_type, ud.language, ud.database, ud.source_type, DATE_TRUNC('month', ud.date)
    ON CONFLICT (entity_text, entity_type, language, database, source_type, month) 
    DO UPDATE SET 
        occurrence_count = entity_statistics.occurrence_count + EXCLUDED.occurrence_count,
        last_updated = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- 4. Create trigger to auto-update on new documents (optional)
CREATE OR REPLACE FUNCTION trigger_update_entity_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Update statistics for the new document
    PERFORM update_entity_statistics_for_documents(ARRAY[NEW.id]);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Attach trigger to uploaded_document table
CREATE TRIGGER update_entity_stats_on_insert
AFTER INSERT ON uploaded_document
FOR EACH ROW
EXECUTE FUNCTION trigger_update_entity_stats();

-- 5. Initial population (one-time, will take a while)
DO $$
DECLARE
    batch_size INTEGER := 1000;
    offset_val INTEGER := 0;
    doc_batch INTEGER[];
BEGIN
    LOOP
        -- Get next batch of document IDs
        SELECT ARRAY_AGG(id) INTO doc_batch
        FROM (
            SELECT id FROM uploaded_document
            ORDER BY id
            LIMIT batch_size OFFSET offset_val
        ) t;
        
        EXIT WHEN doc_batch IS NULL OR array_length(doc_batch, 1) IS NULL;
        
        -- Process this batch
        PERFORM update_entity_statistics_for_documents(doc_batch);
        
        -- Log progress
        RAISE NOTICE 'Processed % documents', offset_val + array_length(doc_batch, 1);
        
        offset_val := offset_val + batch_size;
    END LOOP;
END $$;

-- 6. Query example (same as materialized view but from table)
SELECT 
    entity_text,
    entity_type,
    SUM(occurrence_count) as total_count
FROM entity_statistics
WHERE language = 'EN'
    AND month >= '2024-01-01'
GROUP BY entity_text, entity_type
ORDER BY total_count DESC
LIMIT 20;