-- SQL queries to analyze the keywords array structure

-- 1. Check data type of keywords column
SELECT 
    column_name,
    data_type,
    udt_name,
    element_type
FROM information_schema.columns 
WHERE table_name = 'document_section_chunk' 
    AND column_name = 'keywords';

-- 2. Get a sample of keywords arrays to see the structure
SELECT 
    id,
    keywords,
    array_length(keywords, 1) as keyword_count,
    pg_typeof(keywords) as array_type
FROM document_section_chunk 
WHERE keywords IS NOT NULL 
    AND array_length(keywords, 1) > 0
LIMIT 10;

-- 3. Check if keywords are simple strings or complex structures
SELECT 
    id,
    keywords[1] as first_keyword,
    pg_typeof(keywords[1]) as element_type
FROM document_section_chunk 
WHERE keywords IS NOT NULL 
    AND array_length(keywords, 1) > 0
LIMIT 5;

-- 4. Get statistics about keyword array lengths
SELECT 
    MIN(array_length(keywords, 1)) as min_keywords,
    MAX(array_length(keywords, 1)) as max_keywords,
    AVG(array_length(keywords, 1))::numeric(10,2) as avg_keywords,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY array_length(keywords, 1)) as median_keywords,
    COUNT(*) as chunks_with_keywords
FROM document_section_chunk 
WHERE keywords IS NOT NULL;

-- 5. Distribution of keyword array lengths
SELECT 
    array_length(keywords, 1) as keyword_count,
    COUNT(*) as chunk_count
FROM document_section_chunk 
WHERE keywords IS NOT NULL
GROUP BY array_length(keywords, 1)
ORDER BY keyword_count
LIMIT 20;

-- 6. Sample some actual keywords to see their format
SELECT DISTINCT unnest(keywords) as keyword
FROM document_section_chunk 
WHERE keywords IS NOT NULL
LIMIT 50;

-- 7. Check for any special characters or patterns in keywords
SELECT 
    unnest(keywords) as keyword,
    LENGTH(unnest(keywords)) as keyword_length
FROM document_section_chunk 
WHERE keywords IS NOT NULL
ORDER BY keyword_length DESC
LIMIT 20;

-- 8. Check if keywords contain any structured data (JSON-like)
SELECT 
    unnest(keywords) as keyword
FROM document_section_chunk 
WHERE keywords IS NOT NULL
    AND (
        unnest(keywords) LIKE '{%}' 
        OR unnest(keywords) LIKE '[%]'
        OR unnest(keywords) LIKE '%:%'
    )
LIMIT 20;

-- 9. Most common keywords
SELECT 
    unnest(keywords) as keyword,
    COUNT(*) as frequency
FROM document_section_chunk 
WHERE keywords IS NOT NULL
GROUP BY keyword
ORDER BY frequency DESC
LIMIT 30;

-- 10. Check for empty strings or null values in arrays
SELECT 
    id,
    keywords,
    array_position(keywords, '') as empty_string_position,
    array_position(keywords, NULL) as null_position
FROM document_section_chunk 
WHERE keywords IS NOT NULL
    AND ('' = ANY(keywords) OR NULL = ANY(keywords))
LIMIT 10;