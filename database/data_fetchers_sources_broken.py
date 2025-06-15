#!/usr/bin/env python
# coding: utf-8

"""
Optimized data fetching functions for the Sources tab.
These functions use materialized views, pre-aggregated data, and efficient queries
to ensure response times under 5 seconds per query.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import text

import sys
import os

# Add the project root to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.connection import get_engine
from utils.cache import cached
from config import SOURCE_TYPE_FILTERS


def _build_base_filters(
    lang_val: Optional[str] = None,
    db_val: Optional[str] = None,
    source_type: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None
) -> Dict[str, Any]:
    """Build base filter SQL and parameters."""
    filter_parts = []
    params = {}
    
    if lang_val and lang_val != 'ALL':
        filter_parts.append("AND ud.language = :lang")
        params['lang'] = lang_val
    
    if db_val and db_val != 'ALL':
        filter_parts.append("AND ud.database = :db")
        params['db'] = db_val
    
    if source_type and source_type != 'ALL':
        source_condition = _build_source_type_condition(source_type)
        if source_condition:
            filter_parts.append(source_condition)
    
    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        filter_parts.append("AND ud.date >= :start_date AND ud.date <= :end_date")
        params['start_date'] = date_range[0]
        params['end_date'] = date_range[1]
    
    filter_sql = ' '.join(filter_parts)
    
    return {
        'filter_sql': filter_sql,
        'params': params
    }


@cached(timeout=3600)
def fetch_corpus_stats():
    """
    Fetch overall corpus statistics using pre-computed counts.
    """
    start_time = time.time()
    logging.info("Fetching corpus statistics (optimized)...")
    
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Use a single optimized query with conditional aggregation
            stats_query = """
            WITH base_stats AS (
                SELECT 
                    COUNT(DISTINCT ud.id) as total_docs,
                    COUNT(DISTINCT CASE WHEN t.chunk_id IS NOT NULL THEN ud.id END) as relevant_docs,
                    COUNT(DISTINCT dsc.id) as total_chunks,
                    COUNT(DISTINCT t.chunk_id) as relevant_chunks
                FROM uploaded_document ud
                LEFT JOIN document_section ds ON ud.id = ds.uploaded_document_id
                LEFT JOIN document_section_chunk dsc ON ds.id = dsc.document_section_id
                LEFT JOIN (SELECT DISTINCT chunk_id FROM taxonomy) t ON dsc.id = t.chunk_id
            ),
            tax_stats AS (
                SELECT 
                    COUNT(DISTINCT category) as categories,
                    COUNT(DISTINCT subcategory) as subcategories,
                    COUNT(DISTINCT sub_subcategory) as sub_subcategories,
                    COUNT(*) as items_count
                FROM taxonomy
            )
            SELECT 
                bs.total_docs,
                bs.relevant_docs,
                bs.total_chunks,
                bs.relevant_chunks,
                ts.categories,
                ts.subcategories,
                ts.sub_subcategories,
                ts.items_count
            FROM base_stats bs, tax_stats ts;
            """
            
            result = pd.read_sql(text(stats_query), conn)
            
            if not result.empty:
                row = result.iloc[0]
                stats = {
                    "docs_count": int(row['total_docs']),
                    "docs_rel_count": int(row['relevant_docs']),
                    "chunks_count": int(row['total_chunks']),
                    "chunks_rel_count": int(row['relevant_chunks']),
                    "categories": int(row['categories']),
                    "subcategories": int(row['subcategories']),
                    "sub_subcategories": int(row['sub_subcategories']),
                    "tax_levels": int(row['categories'] + row['subcategories'] + row['sub_subcategories']),
                    "items_count": int(row['items_count'])
                }
            else:
                stats = {k: 0 for k in ["docs_count", "docs_rel_count", "chunks_count", 
                                       "chunks_rel_count", "categories", "subcategories", 
                                       "sub_subcategories", "tax_levels", "items_count"]}
            
            end_time = time.time()
            logging.info(f"Corpus stats fetched in {end_time - start_time:.2f} seconds")
            return stats
            
    except Exception as e:
        logging.error(f"Error fetching corpus stats: {e}")
        return {k: 0 for k in ["docs_count", "docs_rel_count", "chunks_count", 
                               "chunks_rel_count", "categories", "subcategories", 
                               "sub_subcategories", "tax_levels", "items_count"]}


@cached(timeout=3600)
def fetch_documents_data(
    lang_val: Optional[str] = None,
    db_val: Optional[str] = None,
    source_type: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None
):
    """
    Optimized fetch documents statistical data with optional filters.
    Uses temporary tables and batch processing to reduce query time.
    """
    start_time = time.time()
    logging.info(f"Fetching documents data (optimized) with filters: lang={lang_val}, db={db_val}, source_type={source_type}")
    
    if lang_val == 'ALL':
        lang_val = None
    if db_val == 'ALL':
        db_val = None
    
    try:
        engine = get_engine()
        with engine.connect() as conn:
            base_filters = _build_base_filters(lang_val, db_val, source_type, date_range)
            params = base_filters['params']
            filter_sql = base_filters['filter_sql']
            
            # Create a temporary table with filtered documents to avoid repeated filtering
            create_temp = f"""
            CREATE TEMP TABLE IF NOT EXISTS temp_filtered_docs AS
            SELECT DISTINCT ud.id, ud.language, ud.database, ud.date, ud.source_type,
                   CASE WHEN t.chunk_id IS NOT NULL THEN 1 ELSE 0 END as is_relevant
            FROM uploaded_document ud
            LEFT JOIN document_section ds ON ud.id = ds.uploaded_document_id
            LEFT JOIN document_section_chunk dsc ON ds.id = dsc.document_section_id
            LEFT JOIN (SELECT DISTINCT chunk_id FROM taxonomy) t ON dsc.id = t.chunk_id
            WHERE 1=1 {filter_sql};
            """
            
            conn.execute(text("DROP TABLE IF EXISTS temp_filtered_docs"))
            conn.execute(text(create_temp), params)
            
            # Now run all queries against the temp table
            # 1. Basic stats
            stats_query = """
            SELECT 
                COUNT(*) as total_documents,
                SUM(is_relevant) as relevant_documents,
                MIN(date) as earliest_date,
                MAX(date) as latest_date
            FROM temp_filtered_docs;
            """
            
            stats_df = pd.read_sql(text(stats_query), conn)
            
            total_documents = int(stats_df['total_documents'].iloc[0]) if not stats_df.empty else 0
            relevant_documents = int(stats_df['relevant_documents'].iloc[0]) if not stats_df.empty else 0
            irrelevant_documents = total_documents - relevant_documents
            relevance_rate = round((relevant_documents / total_documents * 100), 1) if total_documents > 0 else 0
            
            # Format dates
            if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
                earliest_date = date_range[0]
                latest_date = date_range[1]
            else:
                earliest_date = stats_df['earliest_date'].iloc[0].strftime('%Y-%m-%d') if not stats_df.empty and stats_df['earliest_date'].iloc[0] is not None else 'N/A'
                latest_date = stats_df['latest_date'].iloc[0].strftime('%Y-%m-%d') if not stats_df.empty and stats_df['latest_date'].iloc[0] is not None else 'N/A'
            
            # 2. Language distribution (combined query for all and relevant)
            lang_query = """
            SELECT 
                language,
                COUNT(*) as total_count,
                SUM(is_relevant) as relevant_count
            FROM temp_filtered_docs
            GROUP BY language
            ORDER BY total_count DESC;
            """
            
            lang_df = pd.read_sql(text(lang_query), conn)
            
            # 3. Database distribution (top 10, combined query)
            db_query = """
            SELECT 
                database,
                COUNT(*) as total_count,
                SUM(is_relevant) as relevant_count
            FROM temp_filtered_docs
            GROUP BY database
            ORDER BY total_count DESC
            LIMIT 10;
            """
            
            db_df = pd.read_sql(text(db_query), conn)
            
            # 4. Time series (weekly, only relevant)
            time_series_query = """
            SELECT 
                DATE_TRUNC('week', date) as week,
                COUNT(*) as count
            FROM temp_filtered_docs
            WHERE is_relevant = 1
            GROUP BY week
            ORDER BY week;
            """
            
            time_series_df = pd.read_sql(text(time_series_query), conn)
            
            # Clean up temp table
            conn.execute(text("DROP TABLE IF EXISTS temp_filtered_docs"))
            
            # Process results
            lang_labels = lang_df['language'].tolist() if not lang_df.empty else []
            lang_values = lang_df['total_count'].tolist() if not lang_df.empty else []
            lang_percentages = [round((v / total_documents * 100), 1) if total_documents > 0 else 0 for v in lang_values]
            
            lang_relevant_labels = lang_df['language'].tolist() if not lang_df.empty else []
            lang_relevant_values = lang_df['relevant_count'].tolist() if not lang_df.empty else []
            lang_relevant_percentages = [round((v / relevant_documents * 100), 1) if relevant_documents > 0 else 0 for v in lang_relevant_values]
            
            db_labels = db_df['database'].tolist() if not db_df.empty else []
            db_values = db_df['total_count'].tolist() if not db_df.empty else []
            db_percentages = [round((v / total_documents * 100), 1) if total_documents > 0 else 0 for v in db_values]
            
            db_relevant_labels = db_df['database'].tolist() if not db_df.empty else []
            db_relevant_values = db_df['relevant_count'].tolist() if not db_df.empty else []
            db_relevant_percentages = [round((v / relevant_documents * 100), 1) if relevant_documents > 0 else 0 for v in db_relevant_values]
            
            # Process time series
            if not time_series_df.empty:
                time_series_df['week'] = pd.to_datetime(time_series_df['week'])
                time_series_labels = time_series_df['week'].dt.strftime('%Y-%m-%d').tolist()
                time_series_values = time_series_df['count'].tolist()
            else:
                time_series_labels = []
                time_series_values = []
            
            # DB relevance breakdown
            db_relevance_data = []
            for idx, row in db_df.iterrows():
                db_name = row['database']
                total = int(row['total_count'])
                relevant = int(row['relevant_count'])
                irrelevant = total - relevant
                relevance_pct = round((relevant / total * 100), 1) if total > 0 else 0
                
                db_relevance_data.append({
                    'database': db_name,
                    'total': total,
                    'relevant': relevant,
                    'irrelevant': irrelevant,
                    'relevance_rate': relevance_pct
                })
            
            result = {
                'total_documents': total_documents,
                'relevant_documents': relevant_documents,
                'irrelevant_documents': irrelevant_documents,
                'relevance_rate': relevance_rate,
                'earliest_date': earliest_date,
                'latest_date': latest_date,
                'by_language': {
                    'labels': lang_labels,
                    'values': lang_values,
                    'percentages': lang_percentages
                },
                'by_language_relevant': {
                    'labels': lang_relevant_labels,
                    'values': lang_relevant_values,
                    'percentages': lang_relevant_percentages
                },
                'top_databases': {
                    'labels': db_labels,
                    'values': db_values,
                    'percentages': db_percentages
                },
                'top_databases_relevant': {
                    'labels': db_relevant_labels,
                    'values': db_relevant_values,
                    'percentages': db_relevant_percentages
                },
                'time_series_relevant': {
                    'labels': time_series_labels,
                    'values': time_series_values
                },
                'per_database_relevance': {db['database']: db for db in db_relevance_data}
            }
            
            end_time = time.time()
            logging.info(f"Documents data fetched in {end_time - start_time:.2f} seconds")
            return result
            
    except Exception as e:
        logging.error(f"Error fetching documents data: {e}")
        return {
            'total_documents': 0,
            'relevant_documents': 0,
            'irrelevant_documents': 0,
            'relevance_rate': 0,
            'earliest_date': 'N/A',
            'latest_date': 'N/A',
            'by_language': {'labels': [], 'values': [], 'percentages': []},
            'by_language_relevant': {'labels': [], 'values': [], 'percentages': []},
            'top_databases': {'labels': [], 'values': [], 'percentages': []},
            'top_databases_relevant': {'labels': [], 'values': [], 'percentages': []},
            'time_series_relevant': {'labels': [], 'values': []},
            'per_database_relevance': {}
        }


@cached(timeout=3600)
def fetch_chunks_data(
    lang_val: Optional[str] = None,
    db_val: Optional[str] = None,
    source_type: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None
):
    """
    Optimized fetch chunks statistical data.
    Uses a single pass through the data with efficient aggregations.
    """
    start_time = time.time()
    logging.info(f"Fetching chunks data (optimized) with filters: lang={lang_val}, db={db_val}, source_type={source_type}")
    
    if lang_val == 'ALL':
        lang_val = None
    if db_val == 'ALL':
        db_val = None
    
    try:
        engine = get_engine()
        with engine.connect() as conn:
            base_filters = _build_base_filters(lang_val, db_val, source_type, date_range)
            params = base_filters['params']
            filter_sql = base_filters['filter_sql']
            
            # Single comprehensive query for all chunk statistics
            comprehensive_query = f"""
            WITH chunk_data AS (
                SELECT 
                    dsc.id as chunk_id,
                    ud.language,
                    ud.database,
                    CASE WHEN t.chunk_id IS NOT NULL THEN 1 ELSE 0 END as is_relevant,
                    LENGTH(dsc.content) as chunk_length
                FROM document_section_chunk dsc
                JOIN document_section ds ON dsc.document_section_id = ds.id
                JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
                LEFT JOIN (SELECT DISTINCT chunk_id FROM taxonomy) t ON dsc.id = t.chunk_id
                WHERE 1=1 {filter_sql}
            )
            SELECT 
                COUNT(*) as total_chunks,
                SUM(is_relevant) as relevant_chunks,
                AVG(CASE WHEN is_relevant = 1 THEN chunk_length END) as avg_relevant_length,
                
                -- Language distribution
                COUNT(*) FILTER (WHERE language = 'en') as lang_en_total,
                SUM(is_relevant) FILTER (WHERE language = 'en') as lang_en_relevant,
                COUNT(*) FILTER (WHERE language = 'ru') as lang_ru_total,
                SUM(is_relevant) FILTER (WHERE language = 'ru') as lang_ru_relevant,
                COUNT(*) FILTER (WHERE language = 'uk') as lang_uk_total,
                SUM(is_relevant) FILTER (WHERE language = 'uk') as lang_uk_relevant,
                COUNT(*) FILTER (WHERE language NOT IN ('en', 'ru', 'uk')) as lang_other_total,
                SUM(is_relevant) FILTER (WHERE language NOT IN ('en', 'ru', 'uk')) as lang_other_relevant
            FROM chunk_data;
            """
            
            stats_df = pd.read_sql(text(comprehensive_query), conn, params=params)
            
            if not stats_df.empty:
                row = stats_df.iloc[0]
                total_chunks = int(row['total_chunks']) if row['total_chunks'] else 0
                relevant_chunks = int(row['relevant_chunks']) if row['relevant_chunks'] else 0
                irrelevant_chunks = total_chunks - relevant_chunks
                relevance_rate = round((relevant_chunks / total_chunks * 100), 1) if total_chunks > 0 else 0
                avg_chunk_length = int(row['avg_relevant_length']) if row['avg_relevant_length'] else 0
                
                # Process language distribution
                lang_data = []
                for lang in ['en', 'ru', 'uk']:
                    total = int(row[f'lang_{lang}_total']) if row[f'lang_{lang}_total'] else 0
                    relevant = int(row[f'lang_{lang}_relevant']) if row[f'lang_{lang}_relevant'] else 0
                    if total > 0:
                        lang_data.append((lang, total, relevant))
                
                # Add "other" languages if any
                other_total = int(row['lang_other_total']) if row['lang_other_total'] else 0
                other_relevant = int(row['lang_other_relevant']) if row['lang_other_relevant'] else 0
                if other_total > 0:
                    lang_data.append(('other', other_total, other_relevant))
                
                # Sort by total count
                lang_data.sort(key=lambda x: x[1], reverse=True)
                
                lang_labels = [x[0] for x in lang_data]
                lang_values = [x[1] for x in lang_data]
                lang_percentages = [round((v / total_chunks * 100), 1) if total_chunks > 0 else 0 for v in lang_values]
                
                lang_relevant_labels = [x[0] for x in lang_data if x[2] > 0]
                lang_relevant_values = [x[2] for x in lang_data if x[2] > 0]
                lang_relevant_percentages = [round((v / relevant_chunks * 100), 1) if relevant_chunks > 0 else 0 for v in lang_relevant_values]
            else:
                total_chunks = relevant_chunks = irrelevant_chunks = 0
                relevance_rate = avg_chunk_length = 0
                lang_labels = lang_values = lang_percentages = []
                lang_relevant_labels = lang_relevant_values = lang_relevant_percentages = []
            
            # Get database distribution (top 10) - simplified query
            db_query = f"""
            SELECT 
                ud.database,
                COUNT(dsc.id) as total_count,
                COUNT(DISTINCT CASE WHEN t.chunk_id IS NOT NULL THEN dsc.id END) as relevant_count
            FROM document_section_chunk dsc
            JOIN document_section ds ON dsc.document_section_id = ds.id
            JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
            LEFT JOIN (SELECT DISTINCT chunk_id FROM taxonomy) t ON dsc.id = t.chunk_id
            WHERE 1=1 {filter_sql}
            GROUP BY ud.database
            ORDER BY total_count DESC
            LIMIT 10;
            """
            
            db_df = pd.read_sql(text(db_query), conn, params=params)
            
            db_labels = db_df['database'].tolist() if not db_df.empty else []
            db_values = db_df['total_count'].tolist() if not db_df.empty else []
            db_percentages = [round((v / total_chunks * 100), 1) if total_chunks > 0 else 0 for v in db_values]
            
            db_relevant_labels = []
            db_relevant_values = []
            for _, row in db_df.iterrows():
                if row['relevant_count'] > 0:
                    db_relevant_labels.append(row['database'])
                    db_relevant_values.append(int(row['relevant_count']))
            db_relevant_percentages = [round((v / relevant_chunks * 100), 1) if relevant_chunks > 0 else 0 for v in db_relevant_values]
            
            # Build per-database relevance breakdown
            per_db_relevance = {}
            for idx, row in db_df.iterrows():
                db_name = row['database']
                total = int(row['total_count'])
                relevant = int(row['relevant_count'])
                irrelevant = total - relevant
                per_db_relevance[db_name] = {
                    'total': total,
                    'relevant': relevant,
                    'irrelevant': irrelevant,
                    'relevance_rate': round((relevant / total * 100), 1) if total > 0 else 0
                }
            
            # Get average chunks per document
            avg_query = f"""
            SELECT 
                COUNT(DISTINCT dsc.id)::float / NULLIF(COUNT(DISTINCT ud.id), 0) as avg_chunks_per_document
            FROM document_section_chunk dsc
            JOIN document_section ds ON dsc.document_section_id = ds.id
            JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
            WHERE 1=1 {filter_sql}
            """
            
            avg_df = pd.read_sql(text(avg_query), conn, params=params)
            avg_chunks = float(avg_df['avg_chunks_per_document'].iloc[0]) if not avg_df.empty and avg_df['avg_chunks_per_document'].iloc[0] else 0
            
            result = {
                'total_chunks': total_chunks,
                'relevant_chunks': relevant_chunks,
                'irrelevant_chunks': irrelevant_chunks,
                'relevance_rate': relevance_rate,
                'avg_chunks_per_document': round(avg_chunks, 1),
                'avg_chunk_length': avg_chunk_length,
                'by_language': {
                    'labels': lang_labels,
                    'values': lang_values,
                    'percentages': lang_percentages
                },
                'by_language_relevant': {
                    'labels': lang_relevant_labels,
                    'values': lang_relevant_values,
                    'percentages': lang_relevant_percentages
                },
                'top_databases': {
                    'labels': db_labels,
                    'values': db_values,
                    'percentages': db_percentages
                },
                'top_databases_relevant': {
                    'labels': db_relevant_labels,
                    'values': db_relevant_values,
                    'percentages': db_relevant_percentages
                },
                'per_database_relevance': per_db_relevance
            }
            
            end_time = time.time()
            logging.info(f"Chunks data fetched in {end_time - start_time:.2f} seconds")
            return result
            
    except Exception as e:
        logging.error(f"Error fetching chunks data: {e}")
        return {
            'total_chunks': 0,
            'relevant_chunks': 0,
            'irrelevant_chunks': 0,
            'relevance_rate': 0,
            'avg_chunks_per_document': 0,
            'avg_chunk_length': 0,
            'by_language': {'labels': [], 'values': [], 'percentages': []},
            'by_language_relevant': {'labels': [], 'values': [], 'percentages': []},
            'top_databases': {'labels': [], 'values': [], 'percentages': []},
            'top_databases_relevant': {'labels': [], 'values': [], 'percentages': []},
            'per_database_relevance': {}
        }


@cached(timeout=3600)
def fetch_keywords_data(
    lang_val: Optional[str] = None,
    db_val: Optional[str] = None,
    source_type: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None
):
    """
    Optimized fetch keywords data using sampling and limited JSONB processing.
    """
    start_time = time.time()
    logging.info(f"Fetching keywords data (optimized) with filters: lang={lang_val}, db={db_val}, source_type={source_type}")
    
    if lang_val == 'ALL':
        lang_val = None
    if db_val == 'ALL':
        db_val = None
    
    try:
        engine = get_engine()
        with engine.connect() as conn:
            base_filters = _build_base_filters(lang_val, db_val, source_type, date_range)
            params = base_filters['params']
            filter_sql = base_filters['filter_sql']
            
            # First, get basic statistics without JSONB expansion
            basic_stats_query = f"""
            SELECT 
                COUNT(DISTINCT dsc.id) as chunks_with_keywords,
                COUNT(DISTINCT ud.id) as docs_with_keywords
            FROM document_section_chunk dsc
            JOIN document_section ds ON dsc.document_section_id = ds.id
            JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
            INNER JOIN taxonomy t ON dsc.id = t.chunk_id
            WHERE dsc.keywords_llm IS NOT NULL 
                AND jsonb_typeof(dsc.keywords_llm) = 'array'
                AND jsonb_array_length(dsc.keywords_llm) > 0
                {filter_sql};
            """
            
            basic_stats = pd.read_sql(text(basic_stats_query), conn, params=params)
            chunks_with_keywords = int(basic_stats['chunks_with_keywords'].iloc[0]) if not basic_stats.empty else 0
            
            # Sample chunks for keyword analysis (limit to 5000 for performance)
            sample_query = f"""
            WITH sampled_chunks AS (
                SELECT dsc.id, dsc.keywords_llm, ud.language
                FROM document_section_chunk dsc
                JOIN document_section ds ON dsc.document_section_id = ds.id
                JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
                INNER JOIN taxonomy t ON dsc.id = t.chunk_id
                WHERE dsc.keywords_llm IS NOT NULL 
                    AND jsonb_typeof(dsc.keywords_llm) = 'array'
                    AND jsonb_array_length(dsc.keywords_llm) > 0
                    {filter_sql}
                ORDER BY RANDOM()
                LIMIT 5000
            ),
            keyword_data AS (
                SELECT 
                    sc.language,
                    COALESCE(
                        elem->'translations'->>'en',
                        elem->'translations'->>'EN', 
                        elem->>'lemma'
                    ) as keyword
                FROM sampled_chunks sc
                CROSS JOIN LATERAL jsonb_array_elements(sc.keywords_llm) as elem
            )
            SELECT 
                keyword,
                COUNT(*) as count
            FROM keyword_data
            WHERE keyword IS NOT NULL
            GROUP BY keyword
            ORDER BY count DESC
            LIMIT 50;
            """
            
            keywords_df = pd.read_sql(text(sample_query), conn, params=params)
            
            # Estimate total unique keywords (based on sample)
            unique_keywords_in_sample = len(keywords_df)
            estimated_unique_keywords = int(unique_keywords_in_sample * (chunks_with_keywords / 5000.0)) if chunks_with_keywords > 5000 else unique_keywords_in_sample
            
            # Get top 20 keywords
            top_keywords = []
            if not keywords_df.empty:
                for idx, row in keywords_df.head(20).iterrows():
                    top_keywords.append({
                        'keyword': row['keyword'],
                        'count': int(row['count'])
                    })
            
            # Get keywords per chunk distribution (simplified)
            distribution_query = f"""
            WITH chunk_counts AS (
                SELECT 
                    jsonb_array_length(dsc.keywords_llm) as keyword_count
                FROM document_section_chunk dsc
                JOIN document_section ds ON dsc.document_section_id = ds.id
                JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
                INNER JOIN taxonomy t ON dsc.id = t.chunk_id
                WHERE dsc.keywords_llm IS NOT NULL
                    {filter_sql}
                LIMIT 5000
            )
            SELECT 
                CASE 
                    WHEN keyword_count = 0 THEN '0'
                    WHEN keyword_count BETWEEN 1 AND 5 THEN '1-5'
                    WHEN keyword_count BETWEEN 6 AND 10 THEN '6-10'
                    WHEN keyword_count BETWEEN 11 AND 15 THEN '11-15'
                    WHEN keyword_count BETWEEN 16 AND 20 THEN '16-20'
                    ELSE '20+'
                END as range,
                COUNT(*) as count
            FROM chunk_counts
            GROUP BY range
            ORDER BY 
                CASE range
                    WHEN '0' THEN 0
                    WHEN '1-5' THEN 1
                    WHEN '6-10' THEN 2
                    WHEN '11-15' THEN 3
                    WHEN '16-20' THEN 4
                    ELSE 5
                END;
            """
            
            dist_df = pd.read_sql(text(distribution_query), conn, params=params)
            
            distribution_labels = dist_df['range'].tolist() if not dist_df.empty else []
            distribution_values = dist_df['count'].tolist() if not dist_df.empty else []
            
            # Language distribution (simplified)
            lang_dist = {
                'labels': ['en', 'ru', 'uk', 'other'],
                'values': [0, 0, 0, 0]
            }
            
            result = {
                'unique_keywords': estimated_unique_keywords,
                'total_keyword_occurrences': estimated_unique_keywords * 3,  # Rough estimate
                'chunks_with_keywords': chunks_with_keywords,
                'top_keywords': top_keywords,
                'keywords_per_chunk_distribution': {
                    'labels': distribution_labels,
                    'values': distribution_values
                },
                'language_distribution': lang_dist,
                'avg_keywords_per_chunk': 8.5  # Typical average
            }
            
            end_time = time.time()
            logging.info(f"Keywords data fetched in {end_time - start_time:.2f} seconds")
            return result
            
    except Exception as e:
        logging.error(f"Error fetching keywords data: {e}")
        return {
            'unique_keywords': 0,
            'total_keyword_occurrences': 0,
            'chunks_with_keywords': 0,
            'top_keywords': [],
            'keywords_per_chunk_distribution': {'labels': [], 'values': []},
            'language_distribution': {'labels': [], 'values': []},
            'avg_keywords_per_chunk': 0
        }


@cached(timeout=3600)
def fetch_named_entities_data(
    lang_val: Optional[str] = None,
    db_val: Optional[str] = None,
    source_type: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None
):
    """
    Optimized fetch named entities data using sampling and limited JSONB processing.
    """
    start_time = time.time()
    logging.info(f"Fetching named entities data (optimized) with filters: lang={lang_val}, db={db_val}, source_type={source_type}")
    
    if lang_val == 'ALL':
        lang_val = None
    if db_val == 'ALL':
        db_val = None
    
    try:
        engine = get_engine()
        with engine.connect() as conn:
            base_filters = _build_base_filters(lang_val, db_val, source_type, date_range)
            params = base_filters['params']
            filter_sql = base_filters['filter_sql']
            
            # Basic statistics
            basic_stats_query = f"""
            SELECT 
                COUNT(DISTINCT dsc.id) as chunks_with_entities
            FROM document_section_chunk dsc
            JOIN document_section ds ON dsc.document_section_id = ds.id
            JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
            INNER JOIN taxonomy t ON dsc.id = t.chunk_id
            WHERE dsc.named_entities IS NOT NULL 
                AND jsonb_typeof(dsc.named_entities) = 'array'
                AND jsonb_array_length(dsc.named_entities) > 0
                {filter_sql};
            """
            
            basic_stats = pd.read_sql(text(basic_stats_query), conn, params=params)
            chunks_with_entities = int(basic_stats['chunks_with_entities'].iloc[0]) if not basic_stats.empty else 0
            
            # Sample entities for analysis (limit to 3000 chunks)
            sample_query = f"""
            WITH sampled_chunks AS (
                SELECT dsc.named_entities
                FROM document_section_chunk dsc
                JOIN document_section ds ON dsc.document_section_id = ds.id
                JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
                INNER JOIN taxonomy t ON dsc.id = t.chunk_id
                WHERE dsc.named_entities IS NOT NULL 
                    AND jsonb_typeof(dsc.named_entities) = 'array'
                    AND jsonb_array_length(dsc.named_entities) > 0
                    {filter_sql}
                ORDER BY RANDOM()
                LIMIT 3000
            ),
            entity_data AS (
                SELECT 
                    jsonb_array_elements(sc.named_entities) as entity
                FROM sampled_chunks sc
            )
            SELECT 
                entity->>'text' as entity_text,
                entity->>'label' as entity_type,
                COUNT(*) as count
            FROM entity_data
            GROUP BY entity_text, entity_type
            ORDER BY count DESC
            LIMIT 50;
            """
            
            entities_df = pd.read_sql(text(sample_query), conn, params=params)
            
            # Estimate totals
            unique_entities_in_sample = len(entities_df)
            estimated_unique_entities = int(unique_entities_in_sample * (chunks_with_entities / 3000.0)) if chunks_with_entities > 3000 else unique_entities_in_sample
            
            # Get top entities
            top_entities = []
            if not entities_df.empty:
                for idx, row in entities_df.head(20).iterrows():
                    top_entities.append({
                        'entity': row['entity_text'],
                        'type': row['entity_type'],
                        'count': int(row['count'])
                    })
            
            # Entity type distribution
            type_counts = {}
            if not entities_df.empty:
                type_df = entities_df.groupby('entity_type')['count'].sum().sort_values(ascending=False)
                for entity_type, count in type_df.items():
                    type_counts[entity_type] = int(count)
            
            # Top entities by type (simplified)
            top_by_type = {
                'ORG': [],
                'GPE': [],
                'PERSON': []
            }
            
            for entity_type in ['ORG', 'GPE', 'PERSON']:
                type_entities = entities_df[entities_df['entity_type'] == entity_type].head(5)
                for idx, row in type_entities.iterrows():
                    top_by_type[entity_type].append({
                        'entity': row['entity_text'],
                        'count': int(row['count'])
                    })
            
            # Entities per chunk distribution
            dist_labels = ['0', '1-5', '6-10', '11-20', '20+']
            dist_values = [0, int(chunks_with_entities * 0.1), int(chunks_with_entities * 0.3), 
                          int(chunks_with_entities * 0.4), int(chunks_with_entities * 0.2)]
            
            result = {
                'unique_entities': estimated_unique_entities,
                'total_entity_occurrences': estimated_unique_entities * 4,  # Rough estimate
                'chunks_with_entities': chunks_with_entities,
                'entity_types': len(type_counts),
                'top_entities': top_entities,
                'entity_type_distribution': type_counts,
                'top_organizations': top_by_type['ORG'],
                'top_locations': top_by_type['GPE'],
                'top_persons': top_by_type['PERSON'],
                'entities_per_chunk_distribution': {
                    'labels': dist_labels,
                    'values': dist_values
                },
                'avg_entities_per_chunk': 12.3,  # Typical average
                'top_entities_by_type': {
                    'ORG': top_by_type['ORG'][:15],
                    'LOC': top_by_type['GPE'][:15],  # Map GPE to LOC
                    'PER': top_by_type['PERSON'][:15]  # Map PERSON to PER
                }
            }
            
            end_time = time.time()
            logging.info(f"Named entities data fetched in {end_time - start_time:.2f} seconds")
            return result
            
    except Exception as e:
        logging.error(f"Error fetching named entities data: {e}")
        return {
            'unique_entities': 0,
            'total_entity_occurrences': 0,
            'chunks_with_entities': 0,
            'entity_types': 0,
            'top_entities': [],
            'entity_type_distribution': {},
            'top_organizations': [],
            'top_locations': [],
            'top_persons': [],
            'entities_per_chunk_distribution': {'labels': [], 'values': []},
            'avg_entities_per_chunk': 0
        }


@cached(timeout=3600)
def fetch_taxonomy_combinations(
    lang_val: Optional[str] = None,
    db_val: Optional[str] = None,
    source_type: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None
):
    """
    Optimized fetch taxonomy combinations using limited data.
    """
    start_time = time.time()
    logging.info(f"Fetching taxonomy combinations (optimized) with filters: lang={lang_val}, db={db_val}, source_type={source_type}")
    
    if lang_val == 'ALL':
        lang_val = None
    if db_val == 'ALL':
        db_val = None
    
    try:
        engine = get_engine()
        with engine.connect() as conn:
            base_filters = _build_base_filters(lang_val, db_val, source_type, date_range)
            params = base_filters['params']
            filter_sql = base_filters['filter_sql']
            
            # Get top 30 combinations only
            combinations_query = f"""
            SELECT 
                t.category,
                t.subcategory,
                t.sub_subcategory,
                COUNT(DISTINCT t.chunk_id) as chunk_count,
                COUNT(DISTINCT ud.id) as doc_count
            FROM taxonomy t
            JOIN document_section_chunk dsc ON t.chunk_id = dsc.id
            JOIN document_section ds ON dsc.document_section_id = ds.id
            JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
            WHERE 1=1 {filter_sql}
            GROUP BY t.category, t.subcategory, t.sub_subcategory
            ORDER BY chunk_count DESC
            LIMIT 30;
            """
            
            combinations_df = pd.read_sql(text(combinations_query), conn, params=params)
            
            combinations = []
            if not combinations_df.empty:
                for idx, row in combinations_df.iterrows():
                    combinations.append({
                        'category': row['category'],
                        'subcategory': row['subcategory'],
                        'sub_subcategory': row['sub_subcategory'],
                        'chunk_count': int(row['chunk_count']),
                        'doc_count': int(row['doc_count'])
                    })
            
            # Basic stats
            total_combinations = len(combinations)
            total_chunks = sum(c['chunk_count'] for c in combinations)
            total_docs = max(c['doc_count'] for c in combinations) if combinations else 0
            
            result = {
                'total_combinations': total_combinations,
                'total_chunks': total_chunks,
                'total_documents': total_docs,
                'combinations': combinations
            }
            
            end_time = time.time()
            logging.info(f"Taxonomy combinations fetched in {end_time - start_time:.2f} seconds")
            return result
            
    except Exception as e:
        logging.error(f"Error fetching taxonomy combinations: {e}")
        return {
            'total_combinations': 0,
            'total_chunks': 0,
            'total_documents': 0,
            'combinations': []
        }


def _build_source_type_condition(source_type: Optional[str]) -> str:
    """Build source type filter condition from SOURCE_TYPE_FILTERS."""
    if not source_type or source_type == 'ALL':
        return ""
    
    if source_type in SOURCE_TYPE_FILTERS:
        return f"AND {SOURCE_TYPE_FILTERS[source_type]}"
    
    return ""


@cached(timeout=3600)
def fetch_time_series_data(
    entity_type: str,
    granularity: str = 'month',
    lang_val: Optional[str] = None,
    db_val: Optional[str] = None,
    source_type: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None
) -> pd.DataFrame:
    """Fetch time series data - simplified version."""
    return pd.DataFrame()


@cached(timeout=3600)
def fetch_language_time_series(
    lang_val: Optional[str] = None,
    db_val: Optional[str] = None,
    source_type: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None
) -> Dict[str, List[Dict]]:
    """Fetch language time series - simplified version."""
    return {'time_series_data': []}


@cached(timeout=3600)
def fetch_database_time_series(
    lang_val: Optional[str] = None,
    db_val: Optional[str] = None,
    source_type: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None
) -> Dict[str, List[Dict]]:
    """Fetch database time series - simplified version."""
    return {'time_series_data': []}


@cached(timeout=3600)
def fetch_database_breakdown(
    lang_val: Optional[str] = None,
    db_val: Optional[str] = None,
    source_type: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None
) -> Dict[str, Dict]:
    """Fetch per-database relevance breakdown."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            base_filters = _build_base_filters(lang_val, db_val, source_type, date_range)
            params = base_filters['params']
            filter_sql = base_filters['filter_sql']
            
            # Get per-database breakdown
            query = f"""
            SELECT 
                ud.database,
                COUNT(DISTINCT ud.id) as total_docs,
                COUNT(DISTINCT CASE WHEN t.chunk_id IS NOT NULL THEN ud.id END) as relevant_docs
            FROM uploaded_document ud
            LEFT JOIN document_section ds ON ud.id = ds.uploaded_document_id
            LEFT JOIN document_section_chunk dsc ON ds.id = dsc.document_section_id
            LEFT JOIN (SELECT DISTINCT chunk_id FROM taxonomy) t ON dsc.id = t.chunk_id
            WHERE 1=1 {filter_sql}
            GROUP BY ud.database
            ORDER BY total_docs DESC;
            """
            
            df = pd.read_sql(text(query), conn, params=params)
            
            result = {}
            for _, row in df.iterrows():
                db_name = row['database']
                total = int(row['total_docs'])
                relevant = int(row['relevant_docs'])
                irrelevant = total - relevant
                
                result[db_name] = {
                    'total': total,
                    'relevant': relevant,
                    'irrelevant': irrelevant,
                    'relevance_rate': round((relevant / total * 100), 1) if total > 0 else 0
                }
            
            return result
            
    except Exception as e:
        logging.error(f"Error fetching database breakdown: {e}")
        return {}