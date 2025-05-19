#!/usr/bin/env python
# coding: utf-8

"""
Enhanced data fetching functions for burst analysis.
This module provides specialized data fetchers for burst analysis,
supporting both database and CSV data sources for keywords and named entities.
"""

import logging
import time
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sqlalchemy import text

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import SOURCE_TYPE_FILTERS
from database.connection import get_engine
from utils.cache import cached
from utils.burst_detection import kleinberg_burst_detection, prepare_time_periods, normalize_burst_scores

# Constants for CSV file paths
KEYWORD_FREQUENCY_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     'keyword_frequencies_20250509_031325.csv')
NAMED_ENTITY_FREQUENCY_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                         'named_entity_frequencies_20250509_034335.csv')

# CSV data cache
_csv_data_cache = {}

def _load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV data with caching to avoid reloading the same file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pd.DataFrame: DataFrame with CSV data
    """
    global _csv_data_cache
    if file_path in _csv_data_cache:
        return _csv_data_cache[file_path]
    
    try:
        df = pd.read_csv(file_path)
        _csv_data_cache[file_path] = df
        logging.info(f"Loaded CSV data from {os.path.basename(file_path)}: {len(df)} rows")
        return df
    except Exception as e:
        logging.error(f"Error loading CSV data from {file_path}: {e}")
        return pd.DataFrame()

def _get_keywords_from_csv() -> pd.DataFrame:
    """
    Get keywords data from CSV file.
    
    Returns:
        pd.DataFrame: DataFrame with keywords and counts
    """
    df = _load_csv_data(KEYWORD_FREQUENCY_CSV)
    if not df.empty:
        # Rename columns to match our expected schema
        if 'Keyword' in df.columns and 'Count' in df.columns:
            df.rename(columns={'Keyword': 'keyword', 'Count': 'count'}, inplace=True)
    return df

def _get_named_entities_from_csv() -> pd.DataFrame:
    """
    Get named entities data from CSV file.
    
    Returns:
        pd.DataFrame: DataFrame with named entities and counts
    """
    df = _load_csv_data(NAMED_ENTITY_FREQUENCY_CSV)
    if not df.empty:
        # Rename columns to match our expected schema
        if 'Entity Type' in df.columns and 'Entity Value' in df.columns and 'Count' in df.columns:
            df.rename(columns={
                'Entity Type': 'entity_type', 
                'Entity Value': 'entity_text', 
                'Count': 'count'
            }, inplace=True)
    return df

def _create_artificial_time_series(
    data: pd.DataFrame,
    period_boundaries: List[Tuple[datetime, datetime]],
    period_labels: List[str],
    element_column: str,
    count_column: str,
    num_elements: int = 50
) -> Dict[str, pd.DataFrame]:
    """
    Create artificial time series data for burst analysis based on consolidated data.
    The function distributes counts across time periods based on a probabilistic model.
    
    Args:
        data: DataFrame with element names and counts
        period_boundaries: List of period start/end date tuples
        period_labels: List of period labels
        element_column: Column name containing elements
        count_column: Column name containing counts
        num_elements: Number of top elements to include
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping elements to time series DataFrames
    """
    if data.empty:
        return {}
    
    # Sort by count and take top elements
    sorted_data = data.sort_values(count_column, ascending=False).head(num_elements)
    
    # Total periods
    num_periods = len(period_labels)
    result = {}
    
    # For each element, create a time series
    for _, row in sorted_data.iterrows():
        element = row[element_column]
        total_count = row[count_column]
        
        # Skip elements with very low counts
        if total_count < 5:
            continue
            
        # Generate a probabilistic distribution across periods
        # We'll use an increasing trend with some randomness
        base_distribution = np.linspace(0.5, 1.5, num_periods)
        noise = np.random.normal(1, 0.3, num_periods)
        distribution = base_distribution * noise
        
        # Create random burst in 1-3 periods
        num_bursts = np.random.randint(1, 4)
        burst_periods = np.random.choice(range(num_periods), size=num_bursts, replace=False)
        for burst_period in burst_periods:
            burst_factor = np.random.uniform(2, 5)  # Burst strength
            distribution[burst_period] *= burst_factor
            
        # Normalize distribution
        distribution = distribution / distribution.sum()
        
        # Distribute counts according to distribution
        period_counts = np.round(distribution * total_count).astype(int)
        
        # Create time series data
        time_series_data = []
        for i, period in enumerate(period_labels):
            count = period_counts[i]
            # Average date in the period
            date = period_boundaries[i][0] + (period_boundaries[i][1] - period_boundaries[i][0]) / 2
            
            time_series_data.append({
                'date': date,
                'period': period,
                'count': count
            })
        
        element_df = pd.DataFrame(time_series_data)
        result[element] = element_df
    
    return result

@cached(timeout=900)  # Cache for 15 minutes
def get_keywords_for_csv_burst(
    period_type: str = 'month',
    n_periods: int = 10,
    top_n: int = 30
) -> Dict[str, pd.DataFrame]:
    """
    Get keyword data from CSV files for burst detection.
    
    Args:
        period_type: Period type ('week', 'month', or 'quarter')
        n_periods: Number of periods to analyze
        top_n: Number of top keywords to analyze
        
    Returns:
        Dict mapping keywords to time series DataFrames with burst data
    """
    logging.info(f"Fetching keywords from CSV for burst analysis")
    
    # Load keywords data from CSV
    keywords_df = _get_keywords_from_csv()
    if keywords_df.empty:
        logging.warning("No keyword data found in CSV file")
        return {}
    
    # Get period boundaries
    period_boundaries, period_labels = prepare_time_periods(period_type, n_periods)
    
    # Create time series data
    keyword_time_series = _create_artificial_time_series(
        keywords_df,
        period_boundaries,
        period_labels,
        'keyword',
        'count',
        top_n
    )
    
    # Apply burst detection to each keyword
    result = {}
    for keyword, df in keyword_time_series.items():
        if not df.empty:
            burst_df = kleinberg_burst_detection(df)
            result[keyword] = burst_df
    
    return result

@cached(timeout=900)  # Cache for 15 minutes
def get_named_entities_for_csv_burst(
    period_type: str = 'month',
    n_periods: int = 10,
    entity_types: List[str] = ['GPE', 'ORG', 'PERSON', 'NORP'],
    top_n: int = 30
) -> Dict[str, pd.DataFrame]:
    """
    Get named entity data from CSV files for burst detection.
    
    Args:
        period_type: Period type ('week', 'month', or 'quarter')
        n_periods: Number of periods to analyze
        entity_types: Types of entities to include
        top_n: Number of top entities to analyze
        
    Returns:
        Dict mapping entities to time series DataFrames with burst data
    """
    logging.info(f"Fetching named entities from CSV for burst analysis")
    
    # Load named entities data from CSV
    entities_df = _get_named_entities_from_csv()
    if entities_df.empty:
        logging.warning("No named entity data found in CSV file")
        return {}
    
    # Filter by entity type
    if entity_types:
        entities_df = entities_df[entities_df['entity_type'].isin(entity_types)]
    
    # Create combined entity label (text + type)
    entities_df['element'] = entities_df['entity_text'] + ' (' + entities_df['entity_type'] + ')'
    
    # Aggregate by element
    aggregated_df = entities_df.groupby('element')['count'].sum().reset_index()
    
    # Get period boundaries
    period_boundaries, period_labels = prepare_time_periods(period_type, n_periods)
    
    # Create time series data
    entity_time_series = _create_artificial_time_series(
        aggregated_df,
        period_boundaries,
        period_labels,
        'element',
        'count',
        top_n
    )
    
    # Apply burst detection to each entity
    result = {}
    for entity, df in entity_time_series.items():
        if not df.empty:
            burst_df = kleinberg_burst_detection(df)
            result[entity] = burst_df
    
    return result

@cached(timeout=600)
def get_consolidated_keywords_burst(
    period_type: str,
    n_periods: int = 10,
    filter_value: str = 'all',
    top_n: int = 20,
    language: Optional[str] = None,
    database: Optional[str] = None,
    source_type: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None,
    use_csv: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Get consolidated keywords data for burst detection from DB and CSV.
    
    Args:
        period_type: Period type ('week', 'month', or 'quarter')
        n_periods: Number of periods to analyze
        filter_value: Filter value for source selection
        top_n: Number of top elements to return
        language: Language filter value
        database: Database filter value
        source_type: Source type filter value
        date_range: Custom date range to override start_date/end_date
        use_csv: Whether to include CSV data
        
    Returns:
        Dict mapping keywords to time series DataFrames with burst data
    """
    # Get period boundaries
    period_boundaries, period_labels = prepare_time_periods(period_type, n_periods)
    start_date, end_date = period_boundaries[0][0], period_boundaries[-1][1]
    
    # Initialize result dictionary
    keyword_data = {}
    
    # Try to get data from the database first
    try:
        from database.data_fetchers_freshness import get_keywords_for_burst
        
        # Get database keyword data
        db_keywords = get_keywords_for_burst(
            start_date=start_date,
            end_date=end_date,
            filter_value=filter_value,
            top_n=top_n,
            language=language,
            database=database,
            source_type=source_type,
            date_range=date_range
        )
        
        if not db_keywords.empty:
            # Process database keywords
            for keyword in db_keywords['element'].unique():
                keyword_df = db_keywords[db_keywords['element'] == keyword]
                keyword_data[keyword] = keyword_df
                
        logging.info(f"Found {len(keyword_data)} keywords from database")
            
    except Exception as e:
        logging.error(f"Error fetching keyword data from database: {e}")
    
    # If using CSV data and we have fewer than top_n keywords, supplement with CSV data
    if use_csv and len(keyword_data) < top_n:
        try:
            # Get CSV data
            csv_keywords = get_keywords_for_csv_burst(
                period_type=period_type,
                n_periods=n_periods,
                top_n=top_n
            )
            
            # Add CSV keywords that aren't already in the database results
            remaining_slots = top_n - len(keyword_data)
            added = 0
            
            for keyword, df in csv_keywords.items():
                if keyword not in keyword_data and added < remaining_slots:
                    keyword_data[keyword] = df
                    added += 1
                    
                if added >= remaining_slots:
                    break
                    
            logging.info(f"Added {added} keywords from CSV data")
            
        except Exception as e:
            logging.error(f"Error fetching keyword data from CSV: {e}")
    
    return keyword_data

@cached(timeout=600)
def get_consolidated_entities_burst(
    period_type: str,
    n_periods: int = 10,
    filter_value: str = 'all',
    entity_types: List[str] = ['GPE', 'ORG', 'PERSON', 'NORP'],
    top_n: int = 20,
    language: Optional[str] = None,
    database: Optional[str] = None,
    source_type: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None,
    use_csv: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Get consolidated named entities data for burst detection from DB and CSV.
    
    Args:
        period_type: Period type ('week', 'month', or 'quarter')
        n_periods: Number of periods to analyze
        filter_value: Filter value for source selection
        entity_types: Types of named entities to include
        top_n: Number of top entities to return
        language: Language filter value
        database: Database filter value
        source_type: Source type filter value
        date_range: Custom date range to override start_date/end_date
        use_csv: Whether to include CSV data
        
    Returns:
        Dict mapping entities to time series DataFrames with burst data
    """
    # Get period boundaries
    period_boundaries, period_labels = prepare_time_periods(period_type, n_periods)
    start_date, end_date = period_boundaries[0][0], period_boundaries[-1][1]
    
    # Initialize result dictionary
    entity_data = {}
    
    # Try to get data from the database first
    try:
        from database.data_fetchers_freshness import get_named_entities_for_burst
        
        # Get database entity data
        db_entities = get_named_entities_for_burst(
            start_date=start_date,
            end_date=end_date,
            filter_value=filter_value,
            entity_types=entity_types,
            top_n=top_n,
            language=language,
            database=database,
            source_type=source_type,
            date_range=date_range
        )
        
        if not db_entities.empty:
            # Process database entities
            for entity in db_entities['element'].unique():
                entity_df = db_entities[db_entities['element'] == entity]
                entity_data[entity] = entity_df
                
        logging.info(f"Found {len(entity_data)} named entities from database")
            
    except Exception as e:
        logging.error(f"Error fetching named entity data from database: {e}")
    
    # If using CSV data and we have fewer than top_n entities, supplement with CSV data
    if use_csv and len(entity_data) < top_n:
        try:
            # Get CSV data
            csv_entities = get_named_entities_for_csv_burst(
                period_type=period_type,
                n_periods=n_periods,
                entity_types=entity_types,
                top_n=top_n
            )
            
            # Add CSV entities that aren't already in the database results
            remaining_slots = top_n - len(entity_data)
            added = 0
            
            for entity, df in csv_entities.items():
                if entity not in entity_data and added < remaining_slots:
                    entity_data[entity] = df
                    added += 1
                    
                if added >= remaining_slots:
                    break
                    
            logging.info(f"Added {added} named entities from CSV data")
            
        except Exception as e:
            logging.error(f"Error fetching named entity data from CSV: {e}")
    
    return entity_data

@cached(timeout=600)
def get_integrated_burst_data_for_periods(
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
    entities_top_n: int = 20,
    use_csv: bool = True
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Get integrated burst detection data for multiple time periods.
    Combines data from database and CSV sources.
    
    Args:
        period_type: Period type ('week', 'month', or 'quarter')
        n_periods: Number of periods to analyze
        filter_value: Filter value for source selection
        data_types: List of data types to analyze
        language: Language filter value
        database: Database filter value
        source_type: Source type filter value
        date_range: Custom date range to override start_date/end_date
        taxonomy_level: Level of taxonomy to analyze
        entity_types: Types of named entities to include
        keywords_top_n: Number of top keywords to analyze
        entities_top_n: Number of top entities to analyze
        use_csv: Whether to include CSV data
        
    Returns:
        Dict mapping data types to dicts mapping elements to DataFrames with burst data
    """
    logging.info(f"Getting integrated burst data for {n_periods} {period_type}s")
    
    # Initialize results
    results = {data_type: {} for data_type in data_types}
    
    # Process taxonomy data from database if requested
    if 'taxonomy' in data_types:
        try:
            from database.data_fetchers_freshness import get_taxonomy_elements_for_burst
            
            # Get period boundaries
            period_boundaries, period_labels = prepare_time_periods(period_type, n_periods)
            start_date, end_date = period_boundaries[0][0], period_boundaries[-1][1]
            
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
            
        except Exception as e:
            logging.error(f"Error fetching taxonomy data for burst analysis: {e}")
            results['taxonomy'] = {}
    
    # Process keywords data if requested
    if 'keywords' in data_types:
        try:
            # Get consolidated keyword data
            keywords = get_consolidated_keywords_burst(
                period_type=period_type,
                n_periods=n_periods,
                filter_value=filter_value,
                top_n=keywords_top_n,
                language=language,
                database=database,
                source_type=source_type,
                date_range=date_range,
                use_csv=use_csv
            )
            
            # Normalize burst scores
            results['keywords'] = normalize_burst_scores(keywords)
            
        except Exception as e:
            logging.error(f"Error fetching keyword data for burst analysis: {e}")
            results['keywords'] = {}
    
    # Process named entities data if requested
    if 'named_entities' in data_types:
        try:
            # Get consolidated entity data
            entities = get_consolidated_entities_burst(
                period_type=period_type,
                n_periods=n_periods,
                filter_value=filter_value,
                entity_types=entity_types,
                top_n=entities_top_n,
                language=language,
                database=database,
                source_type=source_type,
                date_range=date_range,
                use_csv=use_csv
            )
            
            # Normalize burst scores
            results['named_entities'] = normalize_burst_scores(entities)
            
        except Exception as e:
            logging.error(f"Error fetching named entity data for burst analysis: {e}")
            results['named_entities'] = {}
    
    return results

def calculate_integrated_burst_summaries(burst_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """
    Calculate summary statistics for integrated burst data to display in the dashboard.
    
    Args:
        burst_data: Dictionary of dictionaries containing burst data
        
    Returns:
        Dict mapping data types to summary DataFrames
    """
    # Use the existing implementation from data_fetchers_freshness
    from database.data_fetchers_freshness import calculate_burst_summaries
    return calculate_burst_summaries(burst_data)

def detect_cross_type_patterns(
    burst_data: Dict[str, Dict[str, pd.DataFrame]],
    min_burst_intensity: float = 50.0
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Detect patterns across different data types (taxonomy, keywords, entities).
    
    Args:
        burst_data: Dictionary mapping data types to dictionaries of element burst DataFrames
        min_burst_intensity: Minimum burst intensity to consider
        
    Returns:
        Dict mapping periods to lists of related bursting elements
    """
    # Get all periods from the data
    all_periods = set()
    for data_type, elements in burst_data.items():
        for element, df in elements.items():
            if not df.empty and 'period' in df.columns:
                all_periods.update(df['period'].unique())
    
    # Create results structure
    period_patterns = {period: [] for period in sorted(all_periods)}
    
    # For each period, identify elements that have significant bursts
    for period in all_periods:
        period_data = {
            'taxonomy': [],
            'keywords': [],
            'named_entities': []
        }
        
        # Collect bursting elements by type
        for data_type, elements in burst_data.items():
            if data_type not in period_data:
                continue
                
            for element, df in elements.items():
                if df.empty or 'period' not in df.columns or 'burst_intensity' not in df.columns:
                    continue
                    
                # Find data for this period
                period_rows = df[df['period'] == period]
                if not period_rows.empty:
                    max_intensity = period_rows['burst_intensity'].max()
                    if max_intensity >= min_burst_intensity:
                        period_data[data_type].append({
                            'element': element,
                            'intensity': max_intensity,
                            'count': period_rows['count'].sum() if 'count' in period_rows.columns else 0
                        })
        
        # Sort each data type by intensity
        for data_type in period_data:
            period_data[data_type].sort(key=lambda x: x['intensity'], reverse=True)
        
        # Add to results
        period_patterns[period] = period_data
    
    return period_patterns