#!/usr/bin/env python3
"""
Analyze the structure of keywords array in document_section_chunk table
"""

import pandas as pd
import numpy as np
from collections import Counter
import json
from database.connection import get_engine
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_keywords_structure():
    """Analyze the structure and content of keywords arrays"""
    
    engine = get_engine()
    
    with engine.connect() as conn:
        # 1. Basic structure info
        logging.info("1. Checking keywords column data type...")
        query = """
        SELECT 
            data_type,
            udt_name
        FROM information_schema.columns 
        WHERE table_name = 'document_section_chunk' 
            AND column_name = 'keywords'
        """
        result = conn.execute(text(query)).fetchone()
        print(f"\nKeywords column type: {result[0]} ({result[1]})")
        
        # 2. Get sample of keywords
        logging.info("2. Getting sample keywords arrays...")
        query = """
        SELECT 
            id,
            keywords,
            array_length(keywords, 1) as keyword_count
        FROM document_section_chunk 
        WHERE keywords IS NOT NULL 
            AND array_length(keywords, 1) > 0
        LIMIT 5
        """
        df_sample = pd.read_sql(text(query), conn)
        print("\nSample keywords arrays:")
        for _, row in df_sample.iterrows():
            print(f"  Chunk {row['id']}: {row['keyword_count']} keywords")
            print(f"    First 5: {row['keywords'][:5] if row['keywords'] else []}")
        
        # 3. Array length statistics
        logging.info("3. Calculating array length statistics...")
        query = """
        SELECT 
            MIN(array_length(keywords, 1)) as min_keywords,
            MAX(array_length(keywords, 1)) as max_keywords,
            AVG(array_length(keywords, 1))::numeric(10,2) as avg_keywords,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY array_length(keywords, 1)) as median_keywords,
            COUNT(*) as chunks_with_keywords,
            COUNT(CASE WHEN keywords = '{}' THEN 1 END) as empty_arrays
        FROM document_section_chunk 
        WHERE keywords IS NOT NULL
        """
        stats = conn.execute(text(query)).fetchone()
        print(f"\nKeyword array statistics:")
        print(f"  Chunks with keywords: {stats[4]:,}")
        print(f"  Empty arrays: {stats[5]:,}")
        print(f"  Min keywords per chunk: {stats[0]}")
        print(f"  Max keywords per chunk: {stats[1]}")
        print(f"  Average keywords per chunk: {stats[2]}")
        print(f"  Median keywords per chunk: {stats[3]}")
        
        # 4. Keyword format analysis
        logging.info("4. Analyzing keyword formats...")
        query = """
        SELECT unnest(keywords) as keyword
        FROM document_section_chunk 
        WHERE keywords IS NOT NULL
            AND array_length(keywords, 1) > 0
        LIMIT 1000
        """
        df_keywords = pd.read_sql(text(query), conn)
        
        # Analyze keyword characteristics
        keywords = df_keywords['keyword'].tolist()
        
        # Check for various patterns
        patterns = {
            'contains_colon': sum(1 for k in keywords if ':' in k),
            'contains_underscore': sum(1 for k in keywords if '_' in k),
            'contains_dash': sum(1 for k in keywords if '-' in k),
            'contains_space': sum(1 for k in keywords if ' ' in k),
            'all_caps': sum(1 for k in keywords if k.isupper()),
            'starts_with_number': sum(1 for k in keywords if k and k[0].isdigit()),
            'contains_non_ascii': sum(1 for k in keywords if not all(ord(c) < 128 for c in k)),
            'looks_like_json': sum(1 for k in keywords if k.startswith(('{', '['))),
        }
        
        print("\nKeyword format analysis (sample of 1000):")
        for pattern, count in patterns.items():
            print(f"  {pattern}: {count} ({count/len(keywords)*100:.1f}%)")
        
        # 5. Keyword length distribution
        logging.info("5. Analyzing keyword lengths...")
        keyword_lengths = [len(k) for k in keywords]
        print(f"\nKeyword length statistics:")
        print(f"  Min length: {min(keyword_lengths)}")
        print(f"  Max length: {max(keyword_lengths)}")
        print(f"  Average length: {np.mean(keyword_lengths):.1f}")
        print(f"  Median length: {np.median(keyword_lengths):.1f}")
        
        # 6. Most common keywords
        logging.info("6. Finding most common keywords...")
        query = """
        SELECT 
            unnest(keywords) as keyword,
            COUNT(*) as frequency
        FROM document_section_chunk 
        WHERE keywords IS NOT NULL
        GROUP BY keyword
        ORDER BY frequency DESC
        LIMIT 20
        """
        df_common = pd.read_sql(text(query), conn)
        print("\nTop 20 most common keywords:")
        for _, row in df_common.iterrows():
            print(f"  '{row['keyword']}': {row['frequency']:,} occurrences")
        
        # 7. Check for special cases
        logging.info("7. Checking for special cases...")
        query = """
        SELECT 
            COUNT(CASE WHEN '' = ANY(keywords) THEN 1 END) as chunks_with_empty_string,
            COUNT(CASE WHEN keywords @> ARRAY[''] THEN 1 END) as chunks_containing_empty,
            COUNT(CASE WHEN keywords = '{}' THEN 1 END) as empty_arrays,
            COUNT(CASE WHEN keywords IS NULL THEN 1 END) as null_keywords
        FROM document_section_chunk
        """
        special = conn.execute(text(query)).fetchone()
        print(f"\nSpecial cases:")
        print(f"  Chunks with empty string in array: {special[0]:,}")
        print(f"  Empty arrays: {special[2]:,}")
        print(f"  NULL keywords: {special[3]:,}")
        
        # 8. Language detection (sample)
        logging.info("8. Detecting keyword languages...")
        # Sample keywords that contain non-ASCII characters
        query = """
        SELECT DISTINCT unnest(keywords) as keyword
        FROM document_section_chunk 
        WHERE keywords IS NOT NULL
            AND unnest(keywords) ~ '[^\x00-\x7F]'
        LIMIT 50
        """
        try:
            df_non_ascii = pd.read_sql(text(query), conn)
            if not df_non_ascii.empty:
                print(f"\nSample non-ASCII keywords (likely non-English):")
                for keyword in df_non_ascii['keyword'].head(10):
                    print(f"  '{keyword}'")
        except Exception as e:
            print(f"\nCouldn't analyze non-ASCII keywords: {e}")
        
        # 9. Keyword uniqueness
        logging.info("9. Analyzing keyword uniqueness...")
        query = """
        WITH keyword_counts AS (
            SELECT 
                unnest(keywords) as keyword,
                COUNT(*) as frequency
            FROM document_section_chunk 
            WHERE keywords IS NOT NULL
            GROUP BY keyword
        )
        SELECT 
            COUNT(*) as unique_keywords,
            COUNT(CASE WHEN frequency = 1 THEN 1 END) as single_occurrence,
            COUNT(CASE WHEN frequency > 100 THEN 1 END) as very_common
        FROM keyword_counts
        """
        uniqueness = conn.execute(text(query)).fetchone()
        print(f"\nKeyword uniqueness:")
        print(f"  Total unique keywords: {uniqueness[0]:,}")
        print(f"  Keywords appearing only once: {uniqueness[1]:,}")
        print(f"  Keywords appearing >100 times: {uniqueness[2]:,}")

if __name__ == "__main__":
    analyze_keywords_structure()