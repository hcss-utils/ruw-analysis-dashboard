#!/usr/bin/env python3
"""Check the structure of keywords_llm column"""

from database.connection import get_engine
from sqlalchemy import text
import json

engine = get_engine()

with engine.connect() as conn:
    # Get a sample of keywords_llm
    query = """
    SELECT 
        id,
        keywords,
        keywords_llm,
        array_length(keywords, 1) as keywords_count,
        jsonb_array_length(keywords_llm) as keywords_llm_count
    FROM document_section_chunk 
    WHERE keywords_llm IS NOT NULL 
        AND jsonb_typeof(keywords_llm) = 'array'
    LIMIT 5;
    """
    
    result = conn.execute(text(query))
    print("Sample of keywords vs keywords_llm:")
    for row in result:
        print(f"\nChunk ID: {row[0]}")
        print(f"  keywords ({row[3]} items): {row[1][:3] if row[1] else None}...")
        print(f"  keywords_llm ({row[4]} items): {json.dumps(row[2][:3] if row[2] else None, ensure_ascii=False)[:100]}...")
    
    # Check data types in keywords_llm
    query2 = """
    SELECT DISTINCT jsonb_typeof(keywords_llm) as type, COUNT(*)
    FROM document_section_chunk
    WHERE keywords_llm IS NOT NULL
    GROUP BY jsonb_typeof(keywords_llm);
    """
    
    result2 = conn.execute(text(query2))
    print("\n\nData types in keywords_llm:")
    for row in result2:
        print(f"  {row[0]}: {row[1]} chunks")