"""
Optimized version of named entities data fetchers using materialized view.
Replace the fetch_named_entities_data function in data_fetchers_sources.py with this version
after running the database_optimizations.sql script.
"""

import logging
import time
import pandas as pd
from typing import Optional, Tuple
from sqlalchemy import text
from database.connection import get_engine


def fetch_named_entities_data_optimized(
    lang_val: Optional[str] = None,
    db_val: Optional[str] = None,
    source_type: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None
):
    """
    Fetch named entities statistical data using materialized view for performance.
    
    Args:
        lang_val: Language filter value
        db_val: Database filter value
        source_type: Source type filter value
        date_range: Date range filter
        
    Returns:
        dict: Named entities data including statistics and distributions
    """
    start_time = time.time()
    logging.info(f"Fetching named entities data with filters: lang={lang_val}, db={db_val}, source_type={source_type}")
    
    if lang_val == 'ALL':
        lang_val = None
    if db_val == 'ALL':
        db_val = None
    
    try:
        engine = get_engine()
        logging.info("Database engine obtained")
        with engine.connect() as conn:
            logging.info("Database connection established")
            
            # Build filter conditions for materialized view
            filters = []
            params = {}
            
            if lang_val:
                filters.append("language = :lang")
                params['lang'] = lang_val
            
            if db_val:
                filters.append("database = :db")
                params['db'] = db_val
                
            if source_type and source_type != 'ALL':
                filters.append("source_type = :source_type")
                params['source_type'] = source_type
                
            if date_range and len(date_range) == 2:
                filters.append("month >= :start_date AND month <= :end_date")
                params['start_date'] = date_range[0]
                params['end_date'] = date_range[1]
            
            filter_sql = " AND " + " AND ".join(filters) if filters else ""
            
            # Get statistics from materialized view (FAST!)
            stats_query = f"""
            SELECT 
                COUNT(DISTINCT entity_text) as unique_entities,
                SUM(occurrence_count) as total_entity_occurrences,
                COUNT(DISTINCT entity_type) as entity_types
            FROM mv_entity_stats
            WHERE 1=1 {filter_sql}
            """
            
            stats_df = pd.read_sql(text(stats_query), conn, params=params)
            
            # Get top entities by frequency from materialized view (FAST!)
            top_entities_query = f"""
            SELECT 
                entity_text,
                entity_type,
                SUM(occurrence_count) as count
            FROM mv_entity_stats
            WHERE 1=1 {filter_sql}
            GROUP BY entity_text, entity_type
            ORDER BY count DESC
            LIMIT 20;
            """
            
            top_entities_df = pd.read_sql(text(top_entities_query), conn, params=params)
            
            # Get entity types distribution from materialized view (FAST!)
            entity_types_query = f"""
            SELECT 
                entity_type,
                SUM(occurrence_count) as count,
                COUNT(DISTINCT entity_text) as unique_entities
            FROM mv_entity_stats
            WHERE 1=1 {filter_sql}
            GROUP BY entity_type
            ORDER BY count DESC;
            """
            
            entity_types_df = pd.read_sql(text(entity_types_query), conn, params=params)
            
            # For chunks with entities count, we still need to query the main table
            # but with the optimized indexes it should be much faster
            chunks_query = f"""
            SELECT COUNT(DISTINCT dsc.id) as chunks_with_entities
            FROM document_section_chunk dsc
            JOIN document_section ds ON dsc.document_section_id = ds.id
            JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
            INNER JOIN taxonomy t ON dsc.id = t.chunk_id
            WHERE dsc.named_entities IS NOT NULL 
                AND jsonb_typeof(dsc.named_entities) = 'array'
                AND jsonb_array_length(dsc.named_entities) > 0
                {filter_sql if not filter_sql else filter_sql.replace('mv_entity_stats.', 'ud.')}
            """
            
            chunks_df = pd.read_sql(text(chunks_query), conn, params=params)
            
            # Extract statistics
            unique_entities = int(stats_df['unique_entities'].iloc[0]) if not stats_df.empty else 0
            total_occurrences = int(stats_df['total_entity_occurrences'].iloc[0]) if not stats_df.empty else 0
            entity_types_count = int(stats_df['entity_types'].iloc[0]) if not stats_df.empty else 0
            chunks_with_entities = int(chunks_df['chunks_with_entities'].iloc[0]) if not chunks_df.empty else 0
            
            # Process top entities
            top_entities = []
            if not top_entities_df.empty:
                for _, row in top_entities_df.head(10).iterrows():
                    top_entities.append({
                        'entity': row['entity_text'],
                        'type': row['entity_type'],
                        'count': int(row['count'])
                    })
            
            # Process entity types distribution
            entity_types_dist = []
            entity_type_labels = []
            entity_type_values = []
            
            if not entity_types_df.empty:
                for _, row in entity_types_df.iterrows():
                    entity_types_dist.append({
                        'type': row['entity_type'],
                        'count': int(row['count']),
                        'unique_entities': int(row['unique_entities'])
                    })
                    entity_type_labels.append(row['entity_type'])
                    entity_type_values.append(int(row['count']))
            
            # Prepare final data structure
            data = {
                'stats': {
                    'unique_entities': unique_entities,
                    'total_occurrences': total_occurrences,
                    'entity_types': entity_types_count,
                    'chunks_with_entities': chunks_with_entities,
                    'avg_entities_per_chunk': round(total_occurrences / chunks_with_entities, 1) if chunks_with_entities > 0 else 0
                },
                'top_entities': top_entities,
                'entity_types': entity_types_dist,
                'entity_type_labels': entity_type_labels,
                'entity_type_values': entity_type_values,
                'distributions': {
                    'entities_per_chunk': {
                        'labels': ['0', '1-5', '6-10', '11-20', '21-50', '50+'],
                        'values': [0, 0, 0, 0, 0, 0]  # Would need separate query for this
                    }
                }
            }
            
            elapsed = time.time() - start_time
            logging.info(f"Named entities data fetched in {elapsed:.2f} seconds (OPTIMIZED)")
            
            return data
            
    except Exception as e:
        logging.error(f"Error fetching named entities data: {e}")
        return {
            'stats': {
                'unique_entities': 0,
                'total_occurrences': 0,
                'entity_types': 0,
                'chunks_with_entities': 0,
                'avg_entities_per_chunk': 0
            },
            'top_entities': [],
            'entity_types': [],
            'entity_type_labels': [],
            'entity_type_values': [],
            'distributions': {
                'entities_per_chunk': {
                    'labels': ['0', '1-5', '6-10', '11-20', '21-50', '50+'],
                    'values': [0, 0, 0, 0, 0, 0]
                }
            }
        }