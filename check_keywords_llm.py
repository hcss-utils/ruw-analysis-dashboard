#!/usr/bin/env python3
"""Check for keywords_llm column in the database"""

from database.connection import get_engine
from sqlalchemy import text

engine = get_engine()

with engine.connect() as conn:
    # Check columns in document_section_chunk
    query = """
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'document_section_chunk' 
    AND column_name LIKE '%keyword%'
    ORDER BY column_name;
    """
    
    result = conn.execute(text(query))
    print("Keyword-related columns in document_section_chunk:")
    for row in result:
        print(f"  - {row[0]}: {row[1]}")