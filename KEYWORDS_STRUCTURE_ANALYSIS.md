# Keywords Array Structure Analysis

## Summary
The `keywords` column in `document_section_chunk` table is a PostgreSQL array of text strings (`TEXT[]`).

## Key Findings

### 1. Data Type
- **Column Type**: `ARRAY` 
- **PostgreSQL Type**: `_text` (array of text)
- **Element Type**: Simple text strings (not JSON or structured data)

### 2. Array Characteristics
- **Chunks with keywords**: 362,302 out of 473,734 total chunks (76.5%)
- **NULL keywords**: 122,357 chunks (25.8%)
- **Empty arrays**: 333 chunks have empty strings in their arrays
- **Array lengths**:
  - Min: 1 keyword
  - Max: 5 keywords
  - Average: 4.05 keywords per chunk
  - Median: 5 keywords per chunk

### 3. Keyword Format
Keywords are typically **multi-word phrases** (not single words):
- 99.9% contain spaces (they're phrases, not individual words)
- 12% contain non-ASCII characters (likely Russian/Ukrainian text)
- Average keyword length: 19.1 characters
- No JSON structures or special formatting

### 4. Content Examples

**Russian keywords** (most common):
- "дата выпуск" (release date) - 8,835 occurrences
- "российский войско" (Russian troops) - 3,910 occurrences
- "министр оборона" (Minister of Defense) - 3,519 occurrences
- "владимир путин" (Vladimir Putin) - 2,630 occurrences

**English keywords**:
- "united states" - 4,722 occurrences
- "russian force" - 2,411 occurrences
- "ukraine" - 1,551 occurrences
- "president vladimir putin" - 1,495 occurrences

### 5. Typical Array Structure
```sql
-- Example from chunk 461587:
['секретарь заместитель', 
 'секретарь заместитель министр', 
 'секретарь заместитель министр оборона', 
 'анна цивилёв', 
 'анна цивилёв']

-- Example from chunk 502988:
['russian force', 
 'russian force', 
 'town.**russian force', 
 'russian force', 
 'ukrainian force']
```

### 6. Notable Patterns
- Keywords often include duplicates within the same array
- Keywords are typically noun phrases or named entities
- Mix of languages (Russian and English) in the dataset
- Keywords appear to be extracted key phrases rather than individual words

## Database Optimization Recommendations

### For Keyword Queries:
```sql
-- Add GIN index for array operations
CREATE INDEX idx_chunk_keywords_gin 
ON document_section_chunk USING GIN (keywords);

-- For finding chunks by specific keyword
SELECT * FROM document_section_chunk 
WHERE keywords @> ARRAY['vladimir putin'];

-- For keyword frequency analysis
SELECT unnest(keywords) as keyword, COUNT(*) 
FROM document_section_chunk 
WHERE keywords IS NOT NULL 
GROUP BY keyword 
ORDER BY COUNT(*) DESC;
```

### Performance Tips:
1. The GIN index dramatically speeds up `@>` (contains) operations
2. For keyword search, use `keywords @> ARRAY['search_term']` instead of unnesting
3. Consider creating a separate keyword frequency table for analytics