#!/usr/bin/env python
# coding: utf-8

"""
Data fetchers specifically for the freshness tab, including burst detection for
taxonomic elements, keywords, and named entities.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sqlalchemy import text

import sys
import os

# Add the project root to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import SOURCE_TYPE_FILTERS
from database.connection import get_engine
from utils.cache import cached
from utils.burst_detection import kleinberg_burst_detection, prepare_time_periods, normalize_burst_scores


@cached(timeout=300)
def get_taxonomy_elements_for_burst(
    start_date: datetime, 
    end_date: datetime,
    filter_value: str = 'all',
    group_by: str = 'category',
    top_n: int = 20,
    language: Optional[str] = None,
    database: Optional[str] = None,
    source_type: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None
) -> pd.DataFrame:
    """
    Get taxonomy elements data for burst detection.
    
    Args:
        start_date: Start date for analysis
        end_date: End date for analysis
        filter_value: Filter value for source selection
        group_by: Level to group by (category, subcategory, sub_subcategory)
        top_n: Number of top elements to return
        language: Language filter value
        database: Database filter value
        source_type: Source type filter value
        date_range: Custom date range to override start_date/end_date
        
    Returns:
        pd.DataFrame: Taxonomy elements data with date and counts
    """
    logging.info(f"Fetching taxonomy elements for burst analysis: {start_date} to {end_date}, level={group_by}")
    
    # Build appropriate filter conditions
    filter_condition = ""
    params = {'start_date': start_date, 'end_date': end_date}
    
    # Add source filter based on filter_value
    if filter_value != 'all':
        if filter_value == 'russian':
            filter_condition = "AND ud.language = 'RU'"
        elif filter_value == 'ukrainian':
            filter_condition = "AND (ud.database LIKE '%ukraine%' OR ud.database LIKE '%kyiv%')"
        elif filter_value == 'western':
            filter_condition = "AND (ud.language = 'EN' AND ud.database NOT LIKE '%russia%')"
        elif filter_value == 'military':
            filter_condition = "AND (ud.database LIKE '%military%' OR ud.database LIKE '%mil%')"
        elif filter_value == 'social_media':
            filter_condition = "AND (ud.database LIKE 'telegram%' OR ud.database = 'twitter' OR ud.database = 'vk')"
    
    # Add standard filters if provided
    if language is not None:
        filter_condition += " AND ud.language = :language"
        params['language'] = language
        
    if database is not None:
        filter_condition += " AND ud.database = :database"
        params['database'] = database
        
    if source_type is not None and source_type in SOURCE_TYPE_FILTERS:
        filter_condition += f" AND {SOURCE_TYPE_FILTERS[source_type]}"
    
    # Use date_range if provided, otherwise use start_date/end_date
    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        params['start_date'] = date_range[0]
        params['end_date'] = date_range[1]
    
    # Validate the group_by parameter
    allowed_levels = ['category', 'subcategory', 'sub_subcategory']
    if group_by not in allowed_levels:
        logging.warning(f"Invalid group_by value: {group_by}. Using 'category' instead.")
        group_by = 'category'
    
    # Query for elements by day
    query = f"""
    SELECT 
        ud.date,
        t.{group_by} as element,
        COUNT(*) as count
    FROM taxonomy t
    JOIN document_section_chunk dsc ON t.chunk_id = dsc.id
    JOIN document_section ds ON dsc.document_section_id = ds.id
    JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
    WHERE ud.date BETWEEN :start_date AND :end_date
    {filter_condition}
    GROUP BY ud.date, t.{group_by}
    ORDER BY ud.date, count DESC
    """
    
    try:
        engine = get_engine()
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
        
        # Find the top N elements by total count for analysis
        element_totals = df.groupby('element')['count'].sum().reset_index()
        top_elements = element_totals.sort_values('count', ascending=False).head(top_n)['element'].tolist()
        
        # Filter to only include top elements
        df = df[df['element'].isin(top_elements)]
        
        logging.info(f"Fetched {len(df)} taxonomy element data points for burst analysis")
        return df
    except Exception as e:
        logging.error(f"Error fetching taxonomy elements for burst analysis: {e}")
        return pd.DataFrame()


@cached(timeout=300)
def get_keywords_for_burst(
    start_date: datetime, 
    end_date: datetime,
    filter_value: str = 'all',
    top_n: int = 20,
    language: Optional[str] = None,
    database: Optional[str] = None,
    source_type: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None
) -> pd.DataFrame:
    """
    Get keywords data for burst detection.
    
    Args:
        start_date: Start date for analysis
        end_date: End date for analysis
        filter_value: Filter value for source selection
        top_n: Number of top elements to return
        language: Language filter value
        database: Database filter value
        source_type: Source type filter value
        date_range: Custom date range to override start_date/end_date
        
    Returns:
        pd.DataFrame: Keywords data with date and counts
    """
    logging.info(f"Fetching keywords for burst analysis: {start_date} to {end_date}")
    
    # Build appropriate filter conditions
    filter_condition = ""
    params = {'start_date': start_date, 'end_date': end_date}
    
    # Add source filter based on filter_value
    if filter_value != 'all':
        if filter_value == 'russian':
            filter_condition = "AND ud.language = 'RU'"
        elif filter_value == 'ukrainian':
            filter_condition = "AND (ud.database LIKE '%ukraine%' OR ud.database LIKE '%kyiv%')"
        elif filter_value == 'western':
            filter_condition = "AND (ud.language = 'EN' AND ud.database NOT LIKE '%russia%')"
        elif filter_value == 'military':
            filter_condition = "AND (ud.database LIKE '%military%' OR ud.database LIKE '%mil%')"
        elif filter_value == 'social_media':
            filter_condition = "AND (ud.database LIKE 'telegram%' OR ud.database = 'twitter' OR ud.database = 'vk')"
    
    # Add standard filters if provided
    if language is not None:
        filter_condition += " AND ud.language = :language"
        params['language'] = language
        
    if database is not None:
        filter_condition += " AND ud.database = :database"
        params['database'] = database
        
    if source_type is not None and source_type in SOURCE_TYPE_FILTERS:
        filter_condition += f" AND {SOURCE_TYPE_FILTERS[source_type]}"
    
    # Use date_range if provided, otherwise use start_date/end_date
    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        params['start_date'] = date_range[0]
        params['end_date'] = date_range[1]
    
    # Query for keywords by day using unnest
    query = f"""
    SELECT 
        ud.date,
        keyword as element,
        COUNT(*) as count
    FROM uploaded_document ud
    JOIN document_section ds ON ud.id = ds.uploaded_document_id
    JOIN document_section_chunk dsc ON ds.id = dsc.document_section_id,
    unnest(dsc.keywords) as keyword
    WHERE ud.date BETWEEN :start_date AND :end_date
    AND dsc.keywords IS NOT NULL
    {filter_condition}
    GROUP BY ud.date, keyword
    ORDER BY ud.date, count DESC
    """
    
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Set a longer statement timeout for this query
            conn.execute(text("SET statement_timeout = '120000'"))  # 120 seconds
            df = pd.read_sql(text(query), conn, params=params)
        
        if df.empty:
            logging.warning("No keyword data found for burst analysis")
            return pd.DataFrame()
        
        # Find the top N keywords by total count for analysis
        element_totals = df.groupby('element')['count'].sum().reset_index()
        top_elements = element_totals.sort_values('count', ascending=False).head(top_n)['element'].tolist()
        
        # Filter to only include top elements
        df = df[df['element'].isin(top_elements)]
        
        logging.info(f"Fetched {len(df)} keyword data points for burst analysis")
        return df
    except Exception as e:
        logging.error(f"Error fetching keywords for burst analysis: {e}")
        return pd.DataFrame()


@cached(timeout=300)
def get_named_entities_for_burst(
    start_date: datetime, 
    end_date: datetime,
    filter_value: str = 'all',
    entity_types: List[str] = ['GPE', 'ORG', 'PERSON', 'NORP'],
    top_n: int = 20,
    language: Optional[str] = None,
    database: Optional[str] = None,
    source_type: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None
) -> pd.DataFrame:
    """
    Get named entities data for burst detection.
    
    Args:
        start_date: Start date for analysis
        end_date: End date for analysis
        filter_value: Filter value for source selection
        entity_types: List of entity types to include
        top_n: Number of top elements to return
        language: Language filter value
        database: Database filter value
        source_type: Source type filter value
        date_range: Custom date range to override start_date/end_date
        
    Returns:
        pd.DataFrame: Named entities data with date and counts
    """
    logging.info(f"Fetching named entities for burst analysis: {start_date} to {end_date}")
    
    # Build appropriate filter conditions
    filter_condition = ""
    params = {'start_date': start_date, 'end_date': end_date}
    
    # Add source filter based on filter_value
    if filter_value != 'all':
        if filter_value == 'russian':
            filter_condition = "AND ud.language = 'RU'"
        elif filter_value == 'ukrainian':
            filter_condition = "AND (ud.database LIKE '%ukraine%' OR ud.database LIKE '%kyiv%')"
        elif filter_value == 'western':
            filter_condition = "AND (ud.language = 'EN' AND ud.database NOT LIKE '%russia%')"
        elif filter_value == 'military':
            filter_condition = "AND (ud.database LIKE '%military%' OR ud.database LIKE '%mil%')"
        elif filter_value == 'social_media':
            filter_condition = "AND (ud.database LIKE 'telegram%' OR ud.database = 'twitter' OR ud.database = 'vk')"
    
    # Add standard filters if provided
    if language is not None:
        filter_condition += " AND ud.language = :language"
        params['language'] = language
        
    if database is not None:
        filter_condition += " AND ud.database = :database"
        params['database'] = database
        
    if source_type is not None and source_type in SOURCE_TYPE_FILTERS:
        filter_condition += f" AND {SOURCE_TYPE_FILTERS[source_type]}"
    
    # Use date_range if provided, otherwise use start_date/end_date
    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        params['start_date'] = date_range[0]
        params['end_date'] = date_range[1]
    
    # Create the entity type filter
    entity_type_list = ", ".join([f"'{e_type}'" for e_type in entity_types])
    
    # Query for named entities by day
    # This is a complex query that extracts entities from JSONB arrays and filters by type
    query = f"""
    WITH entity_data AS (
        SELECT 
            ud.date,
            ne->>'text' as entity_text,
            ne->>'label' as entity_type
        FROM 
            uploaded_document ud
        JOIN 
            document_section ds ON ud.id = ds.uploaded_document_id
        JOIN 
            document_section_chunk dsc ON ds.id = dsc.document_section_id,
            jsonb_array_elements(dsc.named_entities) ne
        WHERE 
            ud.date BETWEEN :start_date AND :end_date
            AND dsc.named_entities IS NOT NULL
            AND jsonb_typeof(dsc.named_entities) = 'array'
            AND ne->>'label' IN ({entity_type_list})
            {filter_condition}
    )
    SELECT
        date,
        entity_text || ' (' || entity_type || ')' as element,
        COUNT(*) as count
    FROM
        entity_data
    GROUP BY
        date, entity_text, entity_type
    ORDER BY
        date, count DESC
    """
    
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Set a longer statement timeout for this query
            conn.execute(text("SET statement_timeout = '180000'"))  # 180 seconds
            df = pd.read_sql(text(query), conn, params=params)
        
        if df.empty:
            logging.warning("No named entity data found for burst analysis")
            return pd.DataFrame()
        
        # Find the top N entities by total count for analysis
        element_totals = df.groupby('element')['count'].sum().reset_index()
        top_elements = element_totals.sort_values('count', ascending=False).head(top_n)['element'].tolist()
        
        # Filter to only include top elements
        df = df[df['element'].isin(top_elements)]
        
        logging.info(f"Fetched {len(df)} named entity data points for burst analysis")
        return df
    except Exception as e:
        logging.error(f"Error fetching named entities for burst analysis: {e}")
        return pd.DataFrame()


def get_burst_data_for_periods(
    period_type: str,
    n_periods: int = 10,
    filter_value: str = 'all',
    data_types: List[str] = ['taxonomy', 'keywords', 'named_entities'],
    language: Optional[str] = None,
    database: Optional[str] = None,
    source_type: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None,
    taxonomy_level: str = 'category',
    entity_types: List[str] = ['GPE', 'ORG', 'PERSON', 'NORP'],
    keywords_top_n: int = 20,
    entities_top_n: int = 20
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Get burst detection data for multiple time periods.
    
    Args:
        period_type: Period type ('week', 'month', or 'quarter')
        n_periods: Number of periods to analyze
        filter_value: Filter value for source selection
        data_types: List of data types to analyze
        language: Language filter value
        database: Database filter value
        source_type: Source type filter value
        date_range: Custom date range filter
        taxonomy_level: Level of taxonomy to analyze
        entity_types: Types of named entities to include
        keywords_top_n: Number of top keywords to analyze
        entities_top_n: Number of top entities to analyze
        
    Returns:
        Dict mapping data types to dicts mapping elements to DataFrames with burst data
    """
    logging.info(f"Getting burst data for {n_periods} {period_type}s")
    
    # Get period boundaries
    period_boundaries, period_labels = prepare_time_periods(period_type, n_periods)
    
    # Initialize results
    results = {data_type: {} for data_type in data_types}
    
    # Process taxonomy data if requested
    if 'taxonomy' in data_types:
        taxonomy_elements = {}
        for (start_date, end_date), period_label in zip(period_boundaries, period_labels):
            taxonomy_df = get_taxonomy_elements_for_burst(
                start_date, end_date, filter_value, 
                group_by=taxonomy_level, 
                language=language, database=database, 
                source_type=source_type, date_range=date_range
            )
            if not taxonomy_df.empty:
                # Group by element and calculate total count and apply burst detection
                for element in taxonomy_df['element'].unique():
                    element_df = taxonomy_df[taxonomy_df['element'] == element]
                    element_df = element_df.sort_values('date')
                    
                    # Add to the element's time series
                    if element not in taxonomy_elements:
                        taxonomy_elements[element] = pd.DataFrame()
                    
                    # Apply burst detection to this period's data
                    burst_df = kleinberg_burst_detection(element_df)
                    
                    # Add period label
                    burst_df['period'] = period_label
                    
                    # Add to the element's data
                    taxonomy_elements[element] = pd.concat([taxonomy_elements[element], burst_df])
        
        # Normalize burst scores across all taxonomy elements
        results['taxonomy'] = normalize_burst_scores(taxonomy_elements)
    
    # Process keywords data if requested
    if 'keywords' in data_types:
        keywords = {}
        for (start_date, end_date), period_label in zip(period_boundaries, period_labels):
            keywords_df = get_keywords_for_burst(
                start_date, end_date, filter_value, 
                top_n=keywords_top_n,
                language=language, database=database, 
                source_type=source_type, date_range=date_range
            )
            if not keywords_df.empty:
                # Group by element and calculate total count and apply burst detection
                for element in keywords_df['element'].unique():
                    element_df = keywords_df[keywords_df['element'] == element]
                    element_df = element_df.sort_values('date')
                    
                    # Add to the element's time series
                    if element not in keywords:
                        keywords[element] = pd.DataFrame()
                    
                    # Apply burst detection to this period's data
                    burst_df = kleinberg_burst_detection(element_df)
                    
                    # Add period label
                    burst_df['period'] = period_label
                    
                    # Add to the element's data
                    keywords[element] = pd.concat([keywords[element], burst_df])
        
        # Normalize burst scores across all keywords
        results['keywords'] = normalize_burst_scores(keywords)
    
    # Process named entities data if requested
    if 'named_entities' in data_types:
        entities = {}
        for (start_date, end_date), period_label in zip(period_boundaries, period_labels):
            entities_df = get_named_entities_for_burst(
                start_date, end_date, filter_value, 
                entity_types=entity_types,
                top_n=entities_top_n,
                language=language, database=database, 
                source_type=source_type, date_range=date_range
            )
            if not entities_df.empty:
                # Group by element and calculate total count and apply burst detection
                for element in entities_df['element'].unique():
                    element_df = entities_df[entities_df['element'] == element]
                    element_df = element_df.sort_values('date')
                    
                    # Add to the element's time series
                    if element not in entities:
                        entities[element] = pd.DataFrame()
                    
                    # Apply burst detection to this period's data
                    burst_df = kleinberg_burst_detection(element_df)
                    
                    # Add period label
                    burst_df['period'] = period_label
                    
                    # Add to the element's data
                    entities[element] = pd.concat([entities[element], burst_df])
        
        # Normalize burst scores across all named entities
        results['named_entities'] = normalize_burst_scores(entities)
    
    return results


def calculate_burst_summaries(burst_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """
    Calculate summary statistics for burst data to display in the dashboard.
    
    Args:
        burst_data: Dictionary of dictionaries containing burst data
        
    Returns:
        Dict mapping data types to summary DataFrames
    """
    summaries = {}
    
    for data_type, elements_data in burst_data.items():
        summary_rows = []
        
        for element, df in elements_data.items():
            if df.empty:
                continue
            
            # Group by period and calculate max burst intensity for each period
            period_summaries = df.groupby('period')['burst_intensity'].max().reset_index()
            
            # Find the period with the highest intensity
            max_intensity_period = period_summaries.loc[period_summaries['burst_intensity'].idxmax()]
            
            # Find the date of the maximum burst within that period
            period_df = df[df['period'] == max_intensity_period['period']]
            max_burst_date = period_df.loc[period_df['burst_intensity'].idxmax(), 'date']
            
            # Calculate total occurrences
            total_count = df['count'].sum()
            
            # Calculate the average burst intensity across all periods
            avg_intensity = period_summaries['burst_intensity'].mean()
            
            summary_rows.append({
                'element': element,
                'max_burst_period': max_intensity_period['period'],
                'max_burst_intensity': max_intensity_period['burst_intensity'],
                'max_burst_date': max_burst_date,
                'total_count': total_count,
                'avg_intensity': avg_intensity
            })
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            # Sort by max burst intensity descending
            summary_df = summary_df.sort_values('max_burst_intensity', ascending=False)
            summaries[data_type] = summary_df
        else:
            summaries[data_type] = pd.DataFrame()
    
    return summaries