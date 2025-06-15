#!/usr/bin/env python
# coding: utf-8

"""
Data fetchers specifically for the compare tab, using the same approach as burstiness tab
for keywords (using keywords array field instead of keywords_llm).
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import pandas as pd
from sqlalchemy import text

import sys
import os

# Add the project root to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.connection import get_engine
from utils.cache import cached
from utils.keyword_mapping import remap_and_aggregate_frequencies
from config import SOURCE_TYPE_FILTERS


def _build_source_type_condition(source_type: Optional[str]) -> str:
    """Build SQL condition for source type filtering."""
    if not source_type or source_type == 'ALL':
        return ""
    
    if source_type in SOURCE_TYPE_FILTERS:
        return SOURCE_TYPE_FILTERS[source_type]
    
    return ""


@cached(timeout=300)
def fetch_keywords_data_compare(
    lang_val: Optional[str] = None,
    db_val: Optional[str] = None,
    source_type: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None
):
    """
    Fetch keywords data for comparison using the same approach as burstiness tab.
    Uses the keywords array field with unnest for better performance.
    
    Args:
        lang_val: Language filter value
        db_val: Database filter value
        source_type: Source type filter value
        date_range: Date range filter
        
    Returns:
        dict: Keywords data including statistics and distributions
    """
    start_time = time.time()
    logging.info(f"Fetching keywords data for compare with filters: lang={lang_val}, db={db_val}, source_type={source_type}")
    
    if lang_val == 'ALL':
        lang_val = None
    if db_val == 'ALL':
        db_val = None
    
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Build filter conditions
            filter_conditions = []
            params = {}
            
            if lang_val:
                filter_conditions.append("ud.language = :lang")
                params['lang'] = lang_val
            
            if db_val:
                filter_conditions.append("ud.database = :db")
                params['db'] = db_val
                
            if date_range and len(date_range) == 2:
                filter_conditions.append("ud.date >= :start_date AND ud.date <= :end_date")
                params['start_date'] = date_range[0]
                params['end_date'] = date_range[1]
            
            # Add source type filter
            source_type_condition = _build_source_type_condition(source_type)
            if source_type_condition:
                # Remove the AND from the condition as we'll add it in the filter
                filter_conditions.append(source_type_condition.replace("AND ", ""))
            
            filter_sql = " AND " + " AND ".join(filter_conditions) if filter_conditions else ""
            
            # Get top keywords using unnest (same as burstiness tab)
            top_keywords_query = f"""
            WITH keyword_counts AS (
                SELECT 
                    keyword,
                    COUNT(*) as count
                FROM uploaded_document ud
                JOIN document_section ds ON ud.id = ds.uploaded_document_id
                JOIN document_section_chunk dsc ON ds.id = dsc.document_section_id
                JOIN taxonomy t ON dsc.id = t.chunk_id  -- Only relevant chunks
                CROSS JOIN LATERAL unnest(dsc.keywords) as keyword
                WHERE dsc.keywords IS NOT NULL 
                    AND array_length(dsc.keywords, 1) > 0
                {filter_sql}
                GROUP BY keyword
            ),
            mapped_keywords AS (
                SELECT * FROM keyword_counts
            )
            SELECT 
                keyword,
                count
            FROM mapped_keywords
            ORDER BY count DESC
            LIMIT 100;
            """
            
            # Set timeout for complex queries
            if 'DYNO' in os.environ:
                conn.execute(text("SET statement_timeout = '20000'"))  # 20 seconds for Heroku
            else:
                conn.execute(text("SET statement_timeout = '60000'"))  # 60 seconds locally
                
            top_keywords_df = pd.read_sql(text(top_keywords_query), conn, params=params)
            
            # Apply keyword mapping
            if not top_keywords_df.empty:
                # Create DataFrame for mapping
                keywords_for_mapping = pd.DataFrame({
                    'Keyword': top_keywords_df['keyword'],
                    'Count': top_keywords_df['count']
                })
                
                # Apply remapping and aggregation
                mapped_df = remap_and_aggregate_frequencies(
                    keywords_for_mapping, 
                    keyword_col='Keyword', 
                    freq_col='Count'
                )
                
                # Replace original data with mapped data
                if not mapped_df.empty:
                    top_keywords_df = pd.DataFrame({
                        'keyword': mapped_df['Keyword'],
                        'count': mapped_df['Count']
                    })
                    # Re-sort and limit to top 20
                    top_keywords_df = top_keywords_df.nlargest(20, 'count')
            
            # Get basic statistics
            stats_query = f"""
            SELECT 
                COUNT(DISTINCT keyword) as unique_keywords,
                COUNT(*) as total_occurrences,
                COUNT(DISTINCT dsc.id) as chunks_with_keywords
            FROM uploaded_document ud
            JOIN document_section ds ON ud.id = ds.uploaded_document_id
            JOIN document_section_chunk dsc ON ds.id = dsc.document_section_id
            JOIN taxonomy t ON dsc.id = t.chunk_id  -- Only relevant chunks
            CROSS JOIN LATERAL unnest(dsc.keywords) as keyword
            WHERE dsc.keywords IS NOT NULL 
                AND array_length(dsc.keywords, 1) > 0
            {filter_sql}
            """
            
            stats_df = pd.read_sql(text(stats_query), conn, params=params)
            
            # Extract statistics
            unique_keywords = int(stats_df['unique_keywords'].iloc[0]) if not stats_df.empty else 0
            total_occurrences = int(stats_df['total_occurrences'].iloc[0]) if not stats_df.empty else 0
            chunks_with_keywords = int(stats_df['chunks_with_keywords'].iloc[0]) if not stats_df.empty else 0
            
            # Process top keywords
            top_keywords_labels = []
            top_keywords_values = []
            if not top_keywords_df.empty:
                top_keywords_labels = top_keywords_df['keyword'].tolist()
                top_keywords_values = top_keywords_df['count'].tolist()
            
            keywords_data = {
                "unique_keywords": unique_keywords,
                "total_occurrences": total_occurrences,
                "chunks_with_keywords": chunks_with_keywords,
                "top_keywords": {
                    "labels": top_keywords_labels,
                    "values": top_keywords_values
                }
            }
            
            end_time = time.time()
            logging.info(f"Keywords data for compare fetched in {end_time - start_time:.2f} seconds")
            return keywords_data
            
    except Exception as e:
        logging.error(f"Error fetching keywords data for compare: {e}")
        # Return empty data structure in case of error
        return {
            "unique_keywords": 0,
            "total_occurrences": 0,
            "chunks_with_keywords": 0,
            "top_keywords": {
                "labels": [],
                "values": []
            }
        }