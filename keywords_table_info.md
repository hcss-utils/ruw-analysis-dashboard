# Keywords Table Information

## Table: `document_section_chunk`

This table contains the keywords column along with other chunk-related data.

### All Columns:

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| **id** | integer | NO | nextval('document_section_chunk_id_seq'::regclass) | Primary key |
| **document_section_id** | integer | NO | None | Foreign key to document_section table |
| **chunk_index** | integer | NO | None | Index of this chunk within its section |
| **content** | text | NO | None | The actual text content of the chunk |
| **embedding** | USER-DEFINED (vector) | YES | None | Vector embedding of the chunk |
| **embedding_model** | text | YES | None | Model used to generate the embedding |
| **named_entities** | jsonb | YES | None | Named entities in JSON format |
| **keywords** | ARRAY | YES | None | Array of keywords extracted from this chunk |

### Key Points:
- **Row count**: 473,734 chunks
- **Keywords column**: Stored as PostgreSQL ARRAY type (likely TEXT[])
- **Primary key**: id
- **Foreign key**: document_section_id references document_section(id)

### Indexes on this table:
- **idx_embedding**: On the embedding column (for similarity searches)
- **idx_document_section_chunk_document_section_id**: On document_section_id (for joins)

### Relationships:
```
uploaded_document (documents)
    ↓
document_section (sections within documents)
    ↓
document_section_chunk (chunks with keywords)
    ↓
taxonomy (classifications of chunks)
```

### Example Query to Get Keywords:
```sql
-- Get keywords from chunks
SELECT 
    dsc.id,
    dsc.keywords,
    array_length(dsc.keywords, 1) as keyword_count,
    dsc.content
FROM document_section_chunk dsc
WHERE dsc.keywords IS NOT NULL
    AND array_length(dsc.keywords, 1) > 0
LIMIT 10;

-- Get unique keywords across all chunks
SELECT DISTINCT unnest(keywords) as keyword
FROM document_section_chunk
WHERE keywords IS NOT NULL
ORDER BY keyword;

-- Count keyword occurrences
SELECT 
    unnest(keywords) as keyword,
    COUNT(*) as occurrence_count
FROM document_section_chunk
WHERE keywords IS NOT NULL
GROUP BY keyword
ORDER BY occurrence_count DESC
LIMIT 20;
```

### To Add an Index on Keywords (if needed):
```sql
-- GIN index for array contains operations
CREATE INDEX idx_chunk_keywords_gin ON document_section_chunk 
USING GIN (keywords);

-- B-tree index for array length operations
CREATE INDEX idx_chunk_keywords_length ON document_section_chunk 
((array_length(keywords, 1)))
WHERE keywords IS NOT NULL;
```