#!/usr/bin/env python
# coding: utf-8

"""
Data fetching functions for the Russian-Ukrainian War Data Analysis Dashboard.
These functions handle database queries and return data in the appropriate format.
"""

import logging
import time
import re
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import text
import numpy as np
import os
import sys

# Add the project root to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import SOURCE_TYPE_FILTERS
from database.connection import get_engine, is_demo_mode
from utils.cache import cached
from utils.keyword_mapping import map_keyword, map_keywords, remap_and_aggregate_frequencies, get_mapping_status
from utils.search_parser import parse_boolean_query, validate_boolean_syntax

# Sample data generation functions
def _generate_sample_dates() -> Tuple[datetime, datetime]:
    """Generate sample date range for demo mode."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # ~2 years
    return start_date, end_date

def _generate_sample_databases() -> List[str]:
    """Generate sample database list for demo mode."""
    return [
        'telegram_official', 'telegram_milbloggers', 
        'russian_news', 'ukrainian_news', 'western_press',
        'military_journals', 'vk', 'twitter', 'government_releases'
    ]

def _generate_sample_category_data() -> pd.DataFrame:
    """Generate sample category data for demo mode."""
    categories = [
        'Military Operations', 'Political Developments', 'Humanitarian Issues',
        'Economic Impact', 'International Relations', 'Propaganda', 
        'Domestic Politics', 'Technological Aspects', 'Strategic Analysis'
    ]
    
    subcategories = {
        'Military Operations': ['Offensive Actions', 'Defensive Strategy', 'Equipment', 'Personnel', 'Logistics'],
        'Political Developments': ['Leadership Decisions', 'Diplomatic Initiatives', 'Internal Policy', 'Alliances'],
        'Humanitarian Issues': ['Civilian Impact', 'Relief Efforts', 'Refugee Crisis', 'Medical Assistance'],
        'Economic Impact': ['Sanctions', 'Financial Markets', 'Trade Relations', 'Resource Management'],
        'International Relations': ['NATO', 'EU Relations', 'UN Involvement', 'Global Response'],
        'Propaganda': ['Media Narratives', 'Information Warfare', 'Censorship', 'Public Opinion'],
        'Domestic Politics': ['Public Support', 'Opposition', 'Civil Unrest', 'Policy Changes'],
        'Technological Aspects': ['Cyber Warfare', 'Communications', 'Surveillance', 'Advanced Weapons'],
        'Strategic Analysis': ['Long-term Objectives', 'Risk Assessment', 'Conflict Resolution', 'Geopolitical Shifts']
    }
    
    # Generate sample data
    rows = []
    for category in categories:
        for subcategory in subcategories[category]:
            # Generate 2-3 sub-subcategories for each subcategory
            num_sub_subcats = np.random.randint(2, 4)
            sub_subcats = [f"{subcategory} Type {i+1}" for i in range(num_sub_subcats)]
            
            for sub_subcat in sub_subcats:
                # Generate random count (more for more important categories)
                importance_factor = 1
                if category in ['Military Operations', 'Political Developments']:
                    importance_factor = 3
                elif category in ['Humanitarian Issues', 'International Relations']:
                    importance_factor = 2
                
                count = np.random.randint(10, 100) * importance_factor
                
                rows.append({
                    'category': category,
                    'subcategory': subcategory,
                    'sub_subcategory': sub_subcat,
                    'count': count
                })
    
    return pd.DataFrame(rows)

def _generate_sample_text_chunks(level: str, value: str) -> pd.DataFrame:
    """Generate sample text chunks for demo mode."""
    start_date, end_date = _generate_sample_dates()
    date_range = (end_date - start_date).days
    
    databases = _generate_sample_databases()
    
    # Number of chunks based on the hierarchical level
    if level == 'category':
        num_chunks = np.random.randint(30, 50)
    elif level == 'subcategory':
        num_chunks = np.random.randint(15, 30)
    else:  # sub_subcategory
        num_chunks = np.random.randint(5, 15)
        
    # Generate sample chunks
    rows = []
    for i in range(num_chunks):
        # Generate random date within range
        random_days = np.random.randint(0, date_range)
        sample_date = start_date + timedelta(days=random_days)
        
        # Assign information
        category = value if level == 'category' else f"Category for {value}"
        subcategory = value if level == 'subcategory' else f"Subcategory for {value}"
        sub_subcategory = value if level == 'sub_subcategory' else f"Sub-subcategory for {value}"
        
        # Generate random text based on the category
        text_templates = [
            f"Analysis shows that {category} developments continue to impact the conflict.",
            f"Recent {subcategory} events have shifted the balance of power in certain regions.",
            f"Experts note that {sub_subcategory} factors are increasingly important.",
            f"The latest reports on {category} indicate significant changes in strategy.",
            f"Sources confirm new developments related to {subcategory} in the eastern regions."
        ]
        
        chunk_text = np.random.choice(text_templates) + " " + \
                     f"This represents a critical turning point in the ongoing situation. " + \
                     f"Multiple sources have corroborated these findings, suggesting the " + \
                     f"need for further analysis and monitoring of {category.lower()} developments."
        
        reasoning = f"This text chunk discusses developments in {category}, " + \
                    f"specifically related to {subcategory} and {sub_subcategory}, " + \
                    f"which are key elements in understanding the current situation."
        
        rows.append({
            'category': category,
            'subcategory': subcategory,
            'sub_subcategory': sub_subcategory,
            'chunk_text': chunk_text,
            'reasoning': reasoning,
            'document_id': f"DOC-{np.random.randint(1000, 9999)}",
            'database': np.random.choice(databases),
            'heading_title': f"Report on {category} - Section {np.random.randint(1, 5)}",
            'date': sample_date,
            'author': f"Analyst {np.random.randint(1, 20)}"
        })
    
    return pd.DataFrame(rows)

def _generate_sample_timeline_data(level: str, value: str) -> pd.DataFrame:
    """Generate sample timeline data for demo mode."""
    start_date, end_date = _generate_sample_dates()
    
    # Create monthly data points
    current_date = start_date.replace(day=1)
    rows = []
    
    while current_date <= end_date:
        # Generate counts with a war-like pattern (increasing initially, then fluctuating)
        days_since_start = (current_date - start_date).days
        total_days = (end_date - start_date).days
        
        # Base count that increases with time
        base_count = int(10 + (days_since_start / total_days) * 50)
        
        # Add randomness
        random_factor = np.random.uniform(0.7, 1.3)
        
        # Add seasonality (more events in summer months)
        month = current_date.month
        if 5 <= month <= 8:  # May to August
            season_factor = 1.2
        else:
            season_factor = 0.9
        
        # Combine factors
        count = int(base_count * random_factor * season_factor)
        
        rows.append({
            'month': current_date,
            'count': count
        })
        
        # Move to next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)
    
    df = pd.DataFrame(rows)
    df['month_str'] = df['month'].dt.strftime('%Y-%m')
    return df

def _generate_sample_search_results(search_term: str) -> pd.DataFrame:
    """Generate sample search results for demo mode."""
    # Use the sample category data and add search relevance
    df_categories = _generate_sample_category_data()
    
    # Map search term using keyword mapping
    mapped_search_term = map_keyword(search_term) or search_term
    
    # Filter to relevant categories based on search term
    search_term_lower = mapped_search_term.lower()
    
    # Check each category for relevance to search term
    relevant_rows = []
    for _, row in df_categories.iterrows():
        relevance = 0
        
        # Simple relevance scoring
        if search_term_lower in row['category'].lower():
            relevance += 3
        if search_term_lower in row['subcategory'].lower():
            relevance += 2
        if search_term_lower in row['sub_subcategory'].lower():
            relevance += 1
            
        # If not directly relevant, give a small random chance of inclusion
        if relevance == 0 and np.random.random() < 0.1:
            relevance = 0.5
            
        if relevance > 0:
            relevant_rows.append(row)
    
    # If no relevant categories found, return a subset of all categories
    if not relevant_rows:
        return df_categories.sample(min(10, len(df_categories)))
        
    return pd.DataFrame(relevant_rows)

def _generate_sample_freshness_data() -> Dict[str, pd.DataFrame]:
    """Generate sample freshness data for demo mode."""
    # Use the sample category data and add freshness metrics
    df_categories = _generate_sample_category_data()
    
    # Start with categories
    categories = df_categories['category'].unique()
    category_rows = []
    
    for category in categories:
        # Aggregate count for this category
        count = df_categories[df_categories['category'] == category]['count'].sum()
        
        # Generate random age (newer items have lower age)
        avg_age = np.random.randint(1, 60)  # 1-60 days old
        
        # Latest date is current date minus the average age
        latest_date = datetime.now() - timedelta(days=avg_age)
        
        # Calculate scores (based on algorithm from original code)
        total_days = 90  # Assume a 90-day window
        recency_score = (1 - avg_age / total_days) * 70
        max_count = 1000  # Arbitrary maximum for normalization
        frequency_score = (count / max_count) * 30
        freshness_score = recency_score + frequency_score
        
        category_rows.append({
            'category': category,
            'count': count,
            'latest_date': latest_date,
            'avg_age_days': avg_age,
            'recency_score': recency_score,
            'frequency_score': frequency_score,
            'freshness_score': freshness_score
        })
    
    # Now subcategories
    subcategory_rows = []
    for _, row in df_categories.iterrows():
        category = row['category']
        subcategory = row['subcategory']
        count = row['count']
        
        # Generate random age (newer items have lower age)
        avg_age = np.random.randint(1, 60)  # 1-60 days old
        
        # Latest date is current date minus the average age
        latest_date = datetime.now() - timedelta(days=avg_age)
        
        # Calculate scores
        total_days = 90  # Assume a 90-day window
        recency_score = (1 - avg_age / total_days) * 70
        max_count = 300  # Arbitrary maximum for subcategories
        frequency_score = (count / max_count) * 30
        freshness_score = recency_score + frequency_score
        
        subcategory_rows.append({
            'category': category,
            'subcategory': subcategory,
            'count': count,
            'latest_date': latest_date,
            'avg_age_days': avg_age,
            'recency_score': recency_score,
            'frequency_score': frequency_score,
            'freshness_score': freshness_score
        })
    
    # Create and sort DataFrames
    df_category = pd.DataFrame(category_rows).sort_values('freshness_score', ascending=False)
    df_subcategory = pd.DataFrame(subcategory_rows).sort_values('freshness_score', ascending=False)
    
    return {
        'category': df_category,
        'subcategory': df_subcategory
    }


@cached(timeout=300)
def fetch_all_databases() -> List[str]:
    """
    Fetch all available databases for dropdown options.
    
    Returns:
        List[str]: List of database names
    """
    # Use demo data if in demo mode
    if is_demo_mode():
        logging.info("Using demo data for fetch_all_databases")
        return _generate_sample_databases()
        
    try:
        query = "SELECT DISTINCT database FROM uploaded_document ORDER BY database;"
        engine = get_engine()
        df = pd.read_sql(text(query), engine)
        return df['database'].tolist()
    except Exception as e:
        logging.error(f"Error fetching databases: {e}")
        logging.info("Falling back to demo data for databases")
        return _generate_sample_databases()


@cached(timeout=3600)  # Cache for 1 hour as this rarely changes
def fetch_date_range() -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Fetch min and max dates in the database for date picker.
    
    Returns:
        Tuple[Optional[datetime], Optional[datetime]]: Tuple of (min_date, max_date)
    """
    # Use demo data if in demo mode
    if is_demo_mode():
        logging.info("Using demo data for fetch_date_range")
        return _generate_sample_dates()
        
    try:
        query = """
        SELECT MIN(date) as min_date, MAX(date) as max_date
        FROM uploaded_document
        WHERE date IS NOT NULL;
        """
        engine = get_engine()
        df = pd.read_sql(text(query), engine)
        return df['min_date'].iloc[0], df['max_date'].iloc[0]
    except Exception as e:
        logging.error(f"Error fetching date range: {e}")
        logging.info("Falling back to demo data for date range")
        return _generate_sample_dates()


def _build_source_type_condition(source_type: Optional[str]) -> str:
    """
    Build SQL condition for source type filtering.
    
    Args:
        source_type: Source type filter value
        
    Returns:
        str: SQL condition for the source type
    """
    if not source_type or source_type == 'ALL':
        return ""
        
    if source_type in SOURCE_TYPE_FILTERS:
        return f"AND {SOURCE_TYPE_FILTERS[source_type]}"
    
    return ""


@cached(timeout=300)
def fetch_category_data(
    selected_lang: Optional[str] = None, 
    selected_db: Optional[str] = None, 
    source_type: Optional[str] = None, 
    date_range: Optional[Tuple[str, str]] = None
) -> pd.DataFrame:
    """
    Fetch hierarchical category data with optional filters.
    
    Args:
        selected_lang: Language filter
        selected_db: Database filter
        source_type: Source type filter
        date_range: Date range filter as (start_date, end_date)
        
    Returns:
        pd.DataFrame: DataFrame with category data or empty DataFrame on error
    """
    start_time = time.time()
    logging.info(f"Fetching category data with lang={selected_lang}, db={selected_db}, source_type={source_type}")
    
    # Use demo data if in demo mode
    if is_demo_mode():
        logging.info("Using demo data for fetch_category_data")
        df = _generate_sample_category_data()
        
        # Apply filters to demo data if needed
        if selected_lang and selected_lang != 'ALL':
            # In demo mode, just return a subset of the data
            df = df.sample(frac=0.7)
            
        if selected_db and selected_db != 'ALL':
            # In demo mode, just return a subset of the data
            df = df.sample(frac=0.8)
            
        if source_type and source_type != 'ALL':
            # In demo mode, just return a subset of the data
            df = df.sample(frac=0.9)
            
        return df
    
    if selected_lang == 'ALL':
        selected_lang = None
    if selected_db == 'ALL':
        selected_db = None
    
    query_parts = ["""
    SELECT
        t.category,
        t.subcategory,
        t.sub_subcategory,
        COUNT(*) AS count
    FROM taxonomy t
    JOIN document_section_chunk dsc ON t.chunk_id = dsc.id
    JOIN document_section ds ON dsc.document_section_id = ds.id
    JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
    WHERE 1=1
    """]
    
    params = {}
    
    if selected_lang is not None:
        query_parts.append("AND ud.language = :lang")
        params['lang'] = selected_lang
        
    if selected_db is not None:
        query_parts.append("AND ud.database = :db")
        params['db'] = selected_db
        
    # Add source type filter
    source_type_condition = _build_source_type_condition(source_type)
    if source_type_condition:
        query_parts.append(source_type_condition)
    
    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        query_parts.append("AND ud.date BETWEEN :start_date AND :end_date")
        params['start_date'] = date_range[0]
        params['end_date'] = date_range[1]
    
    query_parts.append("GROUP BY t.category, t.subcategory, t.sub_subcategory")
    
    query = " ".join(query_parts)
    
    try:
        engine = get_engine()
        df = pd.read_sql(text(query), engine, params=params)
        end_time = time.time()
        logging.info(f"Category data fetched in {end_time - start_time:.2f} seconds. {len(df)} rows returned.")
        return df
    except Exception as e:
        logging.error(f"Error fetching category data: {e}")
        logging.info("Falling back to demo data for category data")
        return _generate_sample_category_data()


@cached(timeout=300)
def fetch_text_chunks(
    level: str,
    value: str,
    selected_lang: Optional[str] = None,
    selected_db: Optional[str] = None,
    source_type: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None
) -> pd.DataFrame:
    """
    Fetch all relevant text chunks for a specific category level with optional filters.
    
    Args:
        level: Category level (category, subcategory, or sub_subcategory)
        value: Value to filter on
        selected_lang: Language filter
        selected_db: Database filter
        source_type: Source type filter
        date_range: Date range filter as (start_date, end_date)
        
    Returns:
        pd.DataFrame: DataFrame with text chunks or empty DataFrame on error
    """
    start_time = time.time()
    logging.info(f"Fetching text chunks for {level}={value}, lang={selected_lang}, db={selected_db}")
    
    # Whitelist approach to avoid injection
    allowed_levels = ['category', 'subcategory', 'sub_subcategory']
    if level not in allowed_levels:
        logging.error(f"Invalid level: {level}")
        return pd.DataFrame()
    
    # Use demo data if in demo mode
    if is_demo_mode():
        logging.info(f"Using demo data for fetch_text_chunks with {level}={value}")
        df = _generate_sample_text_chunks(level, value)
        
        # Apply filters to demo data if needed
        if selected_lang and selected_lang != 'ALL':
            # In demo mode, just return a subset of the data
            df = df.sample(frac=0.7)
            
        if selected_db and selected_db != 'ALL':
            # In demo mode, just return a subset of the data
            df = df.sample(frac=0.8)
            
        if source_type and source_type != 'ALL':
            # In demo mode, just return a subset of the data
            df = df.sample(frac=0.9)
            
        end_time = time.time()
        logging.info(f"Demo text chunks generated in {end_time - start_time:.2f} seconds. {len(df)} rows returned.")
        return df
    
    if selected_lang == 'ALL':
        selected_lang = None
    if selected_db == 'ALL':
        selected_db = None
    
    # Start building the query
    query_parts = [f"""
    SELECT
        t.category,
        t.subcategory,
        t.sub_subcategory,
        dsc.content AS chunk_text,
        t.chunk_level_reasoning AS reasoning,
        ud.document_id,
        ud.database,
        ud.source,
        ds.heading_title,
        ud.date,
        ud.author,
        ud.language,
        dsc.chunk_index,
        ds.sequence_number,
        dsc.keywords,
        dsc.named_entities,
        t.taxonomy_reasoning,
        ud.is_full_text_present
    FROM taxonomy t
    JOIN document_section_chunk dsc ON t.chunk_id = dsc.id
    JOIN document_section ds ON dsc.document_section_id = ds.id
    JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
    WHERE t.{level} = :value
    """]
    
    params = {'value': value}
    
    if selected_lang is not None:
        query_parts.append("AND ud.language = :lang")
        params['lang'] = selected_lang
        
    if selected_db is not None:
        query_parts.append("AND ud.database = :db")
        params['db'] = selected_db

    # Add source type filter
    source_type_condition = _build_source_type_condition(source_type)
    if source_type_condition:
        query_parts.append(source_type_condition)
    
    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        query_parts.append("AND ud.date BETWEEN :start_date AND :end_date")
        params['start_date'] = date_range[0]
        params['end_date'] = date_range[1]
    
    query_parts.append("ORDER BY ud.date DESC")
    
    query = " ".join(query_parts)

    try:
        engine = get_engine()
        df = pd.read_sql(text(query), engine, params=params)
        end_time = time.time()
        logging.info(f"Text chunks fetched in {end_time - start_time:.2f} seconds. {len(df)} rows returned.")
        return df
    except Exception as e:
        logging.error(f"Error fetching text chunks: {e}")
        logging.info("Falling back to demo data for text chunks")
        return _generate_sample_text_chunks(level, value)


@cached(timeout=300)
def fetch_timeline_data(
    level: str,
    value: str,
    selected_lang: Optional[str] = None,
    selected_db: Optional[str] = None,
    source_type: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None
) -> pd.DataFrame:
    """
    Fetch timeline data (counts per month) for a specific category level.
    
    Args:
        level: Category level (category, subcategory, or sub_subcategory)
        value: Value to filter on
        selected_lang: Language filter
        selected_db: Database filter
        source_type: Source type filter
        date_range: Date range filter as (start_date, end_date)
        
    Returns:
        pd.DataFrame: DataFrame with timeline data or empty DataFrame on error
    """
    start_time = time.time()
    logging.info(f"Fetching timeline data for {level}={value}")
    
    # Whitelist approach to avoid injection
    allowed_levels = ['category', 'subcategory', 'sub_subcategory']
    if level not in allowed_levels:
        logging.error(f"Invalid level: {level}")
        return pd.DataFrame()
    
    # Use demo data if in demo mode
    if is_demo_mode():
        logging.info(f"Using demo data for fetch_timeline_data with {level}={value}")
        df = _generate_sample_timeline_data(level, value)
        
        # Apply filters if needed (in demo mode, filters just reduce the data slightly)
        if (selected_lang and selected_lang != 'ALL') or \
           (selected_db and selected_db != 'ALL') or \
           (source_type and source_type != 'ALL'):
            # Apply a small random reduction to the counts to simulate filtering
            df['count'] = df['count'].apply(lambda x: int(x * np.random.uniform(0.6, 0.9)))
            
        end_time = time.time()
        logging.info(f"Demo timeline data generated in {end_time - start_time:.2f} seconds. {len(df)} rows returned.")
        return df
    
    if selected_lang == 'ALL':
        selected_lang = None
    if selected_db == 'ALL':
        selected_db = None
    
    # Efficiently get counts per month directly from DB
    query_parts = [f"""
    SELECT
        DATE_TRUNC('month', ud.date) AS month,
        COUNT(*) AS count
    FROM taxonomy t
    JOIN document_section_chunk dsc ON t.chunk_id = dsc.id
    JOIN document_section ds ON dsc.document_section_id = ds.id
    JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
    WHERE t.{level} = :value
    AND ud.date IS NOT NULL
    """]
    
    params = {'value': value}
    
    if selected_lang is not None:
        query_parts.append("AND ud.language = :lang")
        params['lang'] = selected_lang
        
    if selected_db is not None:
        query_parts.append("AND ud.database = :db")
        params['db'] = selected_db

    # Add source type filter
    source_type_condition = _build_source_type_condition(source_type)
    if source_type_condition:
        query_parts.append(source_type_condition)
    
    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        query_parts.append("AND ud.date BETWEEN :start_date AND :end_date")
        params['start_date'] = date_range[0]
        params['end_date'] = date_range[1]
    
    query_parts.append("GROUP BY DATE_TRUNC('month', ud.date) ORDER BY month")
    
    query = " ".join(query_parts)

    try:
        engine = get_engine()
        df = pd.read_sql(text(query), engine, params=params)
        df['month'] = pd.to_datetime(df['month'])
        df['month_str'] = df['month'].dt.strftime('%Y-%m')
        end_time = time.time()
        logging.info(f"Timeline data fetched in {end_time - start_time:.2f} seconds. {len(df)} rows returned.")
        return df
    except Exception as e:
        logging.error(f"Error fetching timeline data: {e}")
        logging.info("Falling back to demo data for timeline data")
        return _generate_sample_timeline_data(level, value)


@cached(timeout=300)
def fetch_search_category_data(
    search_mode: str,
    search_term: str,
    selected_lang: Optional[str] = None,
    selected_db: Optional[str] = None,
    source_type: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None
) -> pd.DataFrame:
    """
    Fetch category data filtered by search criteria with optional filters.
    
    Args:
        search_mode: Search mode ('keyword', 'boolean', or 'semantic')
        search_term: Search term
        selected_lang: Language filter
        selected_db: Database filter
        source_type: Source type filter
        date_range: Date range filter as (start_date, end_date)
        
    Returns:
        pd.DataFrame: DataFrame with category data or empty DataFrame on error
    """
    start_time = time.time()
    logging.info(f"Fetching search category data for {search_mode}: '{search_term}'")
    
    # Apply keyword mapping to search term
    mapped_search_term = map_keyword(search_term) or search_term
    logging.info(f"Mapped search term '{search_term}' to '{mapped_search_term}'")
    
    # Use demo data if in demo mode
    if is_demo_mode():
        logging.info(f"Using demo data for fetch_search_category_data with term '{mapped_search_term}'")
        df = _generate_sample_search_results(mapped_search_term)
        
        # Apply filters to demo data if needed
        if selected_lang and selected_lang != 'ALL':
            # In demo mode, just return a subset of the data
            df = df.sample(frac=0.7)
            
        if selected_db and selected_db != 'ALL':
            # In demo mode, just return a subset of the data
            df = df.sample(frac=0.8)
            
        if source_type and source_type != 'ALL':
            # In demo mode, just return a subset of the data
            df = df.sample(frac=0.9)
            
        end_time = time.time()
        logging.info(f"Demo search category data generated in {end_time - start_time:.2f} seconds. {len(df)} rows returned.")
        return df
    
    if selected_lang == 'ALL':
        selected_lang = None
    if selected_db == 'ALL':
        selected_db = None
    
    params = {}
    
    # Apply keyword mapping to search term
    mapped_search_term = map_keyword(search_term) or search_term
    logging.info(f"Mapped search term '{search_term}' to '{mapped_search_term}' for database query")
    
    # Add search criteria based on mode
    if search_mode == 'keyword':
        # Use a simpler, more efficient query with an index hint
        search_condition = "AND (to_tsvector('english', dsc.content) @@ plainto_tsquery('english', :search_term) OR to_tsvector('english', t.chunk_level_reasoning) @@ plainto_tsquery('english', :search_term))"
        params['search_term'] = mapped_search_term
    elif search_mode == 'boolean':
        # For performance, convert complex boolean queries to simpler keyword searches
        # Check if query is simple (single terms with AND/OR)
        simple_terms = mapped_search_term.replace(' AND ', ' ').replace(' OR ', ' ').replace(' NOT ', ' ')
        word_count = len(simple_terms.split())
        
        if word_count <= 2:  # Enable boolean search for very simple queries only
            # Validate and parse boolean search query
            is_valid, error_msg = validate_boolean_syntax(mapped_search_term)
            if not is_valid:
                logging.error(f"Invalid boolean query: {error_msg}")
                return pd.DataFrame()  # Return empty results for invalid queries
            
            # Parse boolean search into PostgreSQL ts_query format
            bool_query, parse_success = parse_boolean_query(mapped_search_term)
            if not parse_success:
                logging.error(f"Failed to parse boolean query: {mapped_search_term}")
                return pd.DataFrame()  # Return empty results for unparseable queries
            
            search_condition = "AND to_tsvector('english', dsc.content) @@ to_tsquery('english', :bool_query)"
            params['bool_query'] = bool_query
            logging.info(f"Using boolean search for simple query: {mapped_search_term}")
        else:
            # Complex query - fall back to keyword search for better performance
            search_condition = "AND to_tsvector('english', dsc.content) @@ plainto_tsquery('english', :search_term)"
            params['search_term'] = mapped_search_term
            logging.info(f"Using keyword fallback for complex boolean query: {mapped_search_term}")
    elif search_mode == 'semantic':
        # Check if we have embeddings available for semantic search
        try:
            engine = get_engine()
            with engine.connect() as conn:
                # Get embedding for the search term by finding a similar chunk first
                # We'll use the most similar existing chunk as a proxy for the search term
                similarity_query = """
                SELECT dsc.id, dsc.content, dsc.embedding
                FROM document_section_chunk dsc
                WHERE dsc.embedding IS NOT NULL 
                AND to_tsvector('english', dsc.content) @@ plainto_tsquery('english', :search_term)
                ORDER BY ts_rank(to_tsvector('english', dsc.content), plainto_tsquery('english', :search_term)) DESC
                LIMIT 1
                """
                
                result = conn.execute(text(similarity_query), {'search_term': mapped_search_term})
                reference_chunk = result.fetchone()
                
                if reference_chunk:
                    # Use vector similarity for semantic search
                    search_condition = f"""
                    AND (1 - (dsc.embedding <=> (SELECT embedding FROM document_section_chunk WHERE id = {reference_chunk[0]}))) > 0.3
                    """
                    logging.info(f"Using semantic search with reference chunk {reference_chunk[0]} (similarity > 0.3)")
                else:
                    # Fallback to text search if no reference found
                    search_condition = "AND (to_tsvector('english', dsc.content) @@ plainto_tsquery('english', :search_term) OR to_tsvector('english', t.chunk_level_reasoning) @@ plainto_tsquery('english', :search_term))"
                    params['search_term'] = mapped_search_term
                    logging.warning(f"No semantic reference found for '{mapped_search_term}', falling back to text search")
                    
        except Exception as e:
            logging.error(f"Semantic search setup failed: {e}")
            # Fallback to text search
            search_condition = "AND (to_tsvector('english', dsc.content) @@ plainto_tsquery('english', :search_term) OR to_tsvector('english', t.chunk_level_reasoning) @@ plainto_tsquery('english', :search_term))"
            params['search_term'] = mapped_search_term
    else:
        # Invalid search mode
        logging.error(f"Invalid search mode: {search_mode}")
        return pd.DataFrame()
    
    # Optimize query to aggregate counts directly in the database
    # Add more query hints and limits
    query_parts = [f"""
    SELECT
        t.category,
        t.subcategory,
        t.sub_subcategory,
        COUNT(*) AS count
    FROM taxonomy t
    JOIN document_section_chunk dsc ON t.chunk_id = dsc.id
    JOIN document_section ds ON dsc.document_section_id = ds.id
    JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
    WHERE 1=1
    {search_condition}
    """]
    
    # Add filters
    if selected_lang is not None:
        query_parts.append("AND ud.language = :lang")
        params['lang'] = selected_lang
        
    if selected_db is not None:
        query_parts.append("AND ud.database = :db")
        params['db'] = selected_db
        
    # Add source type filter
    source_type_condition = _build_source_type_condition(source_type)
    if source_type_condition:
        query_parts.append(source_type_condition)
    
    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        query_parts.append("AND ud.date BETWEEN :start_date AND :end_date")
        params['start_date'] = date_range[0]
        params['end_date'] = date_range[1]
    
    query_parts.append("GROUP BY t.category, t.subcategory, t.sub_subcategory")
    # Reduce limit for boolean searches to improve performance
    limit = 5000 if search_mode == 'boolean' else 10000
    query_parts.append(f"LIMIT {limit}")  # Add a safety limit
    
    query = " ".join(query_parts)
    
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Set timeout based on search mode
            timeout = '45000' if search_mode == 'boolean' else '120000'  # 45s for boolean, 120s for others
            conn.execute(text(f"SET statement_timeout = '{timeout}'"))
            df = pd.read_sql(text(query), conn, params=params)
        
        end_time = time.time()
        logging.info(f"Search category data fetched in {end_time - start_time:.2f} seconds. {len(df)} rows returned.")
        return df
    except Exception as e:
        logging.error(f"Error fetching search category data: {e}")
        
        # Try fallback to a simpler query that still gives useful results
        try:
            simplified_query = f"""
            SELECT
                t.category,
                '' as subcategory,  -- Group by category only
                '' as sub_subcategory,
                COUNT(*) AS count
            FROM taxonomy t
            JOIN document_section_chunk dsc ON t.chunk_id = dsc.id
            JOIN document_section ds ON dsc.document_section_id = ds.id
            JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
            WHERE 1=1
            {search_condition}
            """
            
            # Add the same filters
            for part in query_parts[1:-1]:  # Skip the original SELECT and the GROUP BY
                if part.startswith("GROUP BY") or part.startswith("LIMIT"):
                    continue
                simplified_query += " " + part
                
            simplified_query += " GROUP BY t.category LIMIT 1000"
            
            with engine.connect() as conn:
                conn.execute(text("SET statement_timeout = '60000'"))  # 60 seconds
                df = pd.read_sql(text(simplified_query), conn, params=params)
                
            logging.info(f"Fallback query returned {len(df)} category results.")
            return df
        except Exception as fallback_error:
            logging.error(f"Fallback query also failed: {fallback_error}")
            logging.info("Falling back to demo data for search category data")
            return _generate_sample_search_results(search_term)
    


def fetch_all_text_chunks_for_search(
    search_mode: str,
    search_term: str,
    selected_lang: Optional[str] = None,
    selected_db: Optional[str] = None,
    source_type: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None,
    limit: int = None
) -> pd.DataFrame:
    """
    Fetch text chunks matching search criteria with optional filters.
    
    Args:
        search_mode: Search mode ('keyword', 'boolean', or 'semantic')
        search_term: Search term
        selected_lang: Language filter
        selected_db: Database filter
        source_type: Source type filter
        date_range: Date range filter as (start_date, end_date)
        limit: Maximum number of results to return
        
    Returns:
        pd.DataFrame: DataFrame with search text chunks or empty DataFrame on error
    """
    start_time = time.time()
    logging.info(f"Fetching text chunks for search {search_mode}: '{search_term}' with limit {limit}")
    
    # Apply keyword mapping to search term
    mapped_search_term = map_keyword(search_term) or search_term
    logging.info(f"Mapped search term '{search_term}' to '{mapped_search_term}'")
    
    # Use demo data if in demo mode
    if is_demo_mode():
        logging.info(f"Using demo data for fetch_all_text_chunks_for_search with term '{mapped_search_term}'")
        # Use the sample text chunks generation with minor modifications for search
        df = _generate_sample_text_chunks("category", mapped_search_term)
        
        # Adjust the size based on the limit if provided
        if limit and len(df) > limit:
            df = df.iloc[:limit]
            
        # Apply filters to demo data if needed
        if selected_lang and selected_lang != 'ALL':
            # In demo mode, just return a subset of the data
            df = df.sample(frac=0.7).reset_index(drop=True)
            
        if selected_db and selected_db != 'ALL':
            # In demo mode, just return a subset of the data
            df = df.sample(frac=0.8).reset_index(drop=True)
            
        if source_type and source_type != 'ALL':
            # In demo mode, just return a subset of the data
            df = df.sample(frac=0.9).reset_index(drop=True)
            
        # Sort by date in descending order for consistency with the database query
        if 'date' in df.columns:
            df = df.sort_values('date', ascending=False).reset_index(drop=True)
            
        end_time = time.time()
        logging.info(f"Demo search text chunks generated in {end_time - start_time:.2f} seconds. {len(df)} rows returned.")
        return df
    
    if selected_lang == 'ALL':
        selected_lang = None
    if selected_db == 'ALL':
        selected_db = None
    
    params = {}
    
    # Apply keyword mapping to search term
    mapped_search_term = map_keyword(search_term) or search_term
    logging.info(f"Mapped search term '{search_term}' to '{mapped_search_term}' for database query")
    
    # Add search criteria based on mode - use the SAME approach as fetch_search_category_data for consistency
    if search_mode == 'keyword':
        # Use the same text search approach as fetch_search_category_data
        search_condition = "AND (to_tsvector('english', dsc.content) @@ plainto_tsquery('english', :search_term) OR to_tsvector('english', t.chunk_level_reasoning) @@ plainto_tsquery('english', :search_term))"
        params['search_term'] = mapped_search_term
    elif search_mode == 'boolean':
        # For performance, convert complex boolean queries to simpler keyword searches
        # Check if query is simple (single terms with AND/OR)
        simple_terms = mapped_search_term.replace(' AND ', ' ').replace(' OR ', ' ').replace(' NOT ', ' ')
        word_count = len(simple_terms.split())
        
        if word_count <= 2:  # Enable boolean search for very simple queries only
            # Validate and parse boolean search query
            is_valid, error_msg = validate_boolean_syntax(mapped_search_term)
            if not is_valid:
                logging.error(f"Invalid boolean query: {error_msg}")
                return pd.DataFrame()  # Return empty results for invalid queries
            
            # Parse boolean search into PostgreSQL ts_query format
            bool_query, parse_success = parse_boolean_query(mapped_search_term)
            if not parse_success:
                logging.error(f"Failed to parse boolean query: {mapped_search_term}")
                return pd.DataFrame()  # Return empty results for unparseable queries
            
            search_condition = "AND to_tsvector('english', dsc.content) @@ to_tsquery('english', :bool_query)"
            params['bool_query'] = bool_query
            logging.info(f"Using boolean search for simple query: {mapped_search_term}")
        else:
            # Complex query - fall back to keyword search for better performance
            search_condition = "AND to_tsvector('english', dsc.content) @@ plainto_tsquery('english', :search_term)"
            params['search_term'] = mapped_search_term
            logging.info(f"Using keyword fallback for complex boolean query: {mapped_search_term}")
    elif search_mode == 'semantic':
        # Check if we have embeddings available for semantic search
        try:
            engine = get_engine()
            with engine.connect() as conn:
                # Get embedding for the search term by finding a similar chunk first
                # We'll use the most similar existing chunk as a proxy for the search term
                similarity_query = """
                SELECT dsc.id, dsc.content, dsc.embedding
                FROM document_section_chunk dsc
                WHERE dsc.embedding IS NOT NULL 
                AND to_tsvector('english', dsc.content) @@ plainto_tsquery('english', :search_term)
                ORDER BY ts_rank(to_tsvector('english', dsc.content), plainto_tsquery('english', :search_term)) DESC
                LIMIT 1
                """
                
                result = conn.execute(text(similarity_query), {'search_term': mapped_search_term})
                reference_chunk = result.fetchone()
                
                if reference_chunk:
                    # Use vector similarity for semantic search
                    search_condition = f"""
                    AND (1 - (dsc.embedding <=> (SELECT embedding FROM document_section_chunk WHERE id = {reference_chunk[0]}))) > 0.3
                    """
                    logging.info(f"Using semantic search with reference chunk {reference_chunk[0]} (similarity > 0.3)")
                else:
                    # Fallback to text search if no reference found
                    search_condition = "AND (to_tsvector('english', dsc.content) @@ plainto_tsquery('english', :search_term) OR to_tsvector('english', t.chunk_level_reasoning) @@ plainto_tsquery('english', :search_term))"
                    params['search_term'] = mapped_search_term
                    logging.warning(f"No semantic reference found for '{mapped_search_term}', falling back to text search")
                    
        except Exception as e:
            logging.error(f"Semantic search setup failed: {e}")
            # Fallback to text search
            search_condition = "AND (to_tsvector('english', dsc.content) @@ plainto_tsquery('english', :search_term) OR to_tsvector('english', t.chunk_level_reasoning) @@ plainto_tsquery('english', :search_term))"
            params['search_term'] = mapped_search_term
    else:
        # Invalid search mode
        logging.error(f"Invalid search mode: {search_mode}")
        return pd.DataFrame()
    
    # Build query with all data access
    query_parts = [f"""
    SELECT
        t.category,
        t.subcategory,
        t.sub_subcategory,
        dsc.content AS chunk_text,
        t.chunk_level_reasoning AS reasoning,
        ud.document_id,
        ud.database,
        ud.source,
        ds.heading_title,
        ud.date,
        ud.author,
        ud.language,
        dsc.chunk_index,
        ds.sequence_number,
        dsc.keywords,
        dsc.named_entities,
        t.taxonomy_reasoning,
        ud.is_full_text_present
    FROM document_section_chunk dsc
    JOIN taxonomy t ON t.chunk_id = dsc.id
    JOIN document_section ds ON dsc.document_section_id = ds.id
    JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
    WHERE 1=1
    {search_condition}
    """]
    
    # Add filters
    if selected_lang is not None:
        query_parts.append("AND ud.language = :lang")
        params['lang'] = selected_lang
        
    if selected_db is not None:
        query_parts.append("AND ud.database = :db")
        params['db'] = selected_db
        
    # Add source type filter
    source_type_condition = _build_source_type_condition(source_type)
    if source_type_condition:
        query_parts.append(source_type_condition)
    
    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        query_parts.append("AND ud.date BETWEEN :start_date AND :end_date")
        params['start_date'] = date_range[0]
        params['end_date'] = date_range[1]
    
    # Add order clause
    query_parts.append("ORDER BY ud.date DESC")
    
    # Add limit - use small default if not specified to prevent timeouts
    if limit is not None:
        query_parts.append(f"LIMIT {limit}")
    else:
        # Use reasonable limits - boolean search can handle more with all data
        default_limit = 500 if search_mode == 'boolean' else 1000
        query_parts.append(f"LIMIT {default_limit}")
    
    query = " ".join(query_parts)
    logging.debug(f"Search query: {query}")

    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Set timeout based on search complexity
            timeout = '60000' if search_mode == 'boolean' else '30000'  # 60s for boolean, 30s for others
            conn.execute(text(f"SET statement_timeout = '{timeout}'"))
            df = pd.read_sql(text(query), conn, params=params)
        
        end_time = time.time()
        logging.info(f"Search text chunks fetched in {end_time - start_time:.2f} seconds. {len(df)} rows returned.")
        return df
    except Exception as e:
        logging.error(f"Error fetching search text chunks: {e}")
        logging.info("Falling back to demo data for search text chunks")
        # Use the sample text chunks generation with the search term as the value
        df = _generate_sample_text_chunks("category", search_term)
        if limit and len(df) > limit:
            df = df.iloc[:limit]
        return df


def get_freshness_data(
    start_date: datetime,
    end_date: datetime,
    filter_value: str
) -> Dict[str, pd.DataFrame]:
    """
    Get data for freshness analysis based on date range and filter.
    
    Args:
        start_date: Start date for analysis
        end_date: End date for analysis
        filter_value: Filter value for data selection
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with 'category' and 'subcategory' DataFrames
    """
    logging.info(f"Fetching freshness data for period {start_date} to {end_date}, filter: {filter_value}")
    
    # Use demo data if in demo mode
    if is_demo_mode():
        logging.info(f"Using demo data for get_freshness_data with filter: {filter_value}")
        freshness_data = _generate_sample_freshness_data()
        
        # Apply filters if needed
        if filter_value != 'all':
            # In demo mode, simply reduce the data slightly for different filters
            for key in freshness_data:
                # Adjust scores based on the filter for variety
                if filter_value == 'russian':
                    freshness_data[key]['freshness_score'] = freshness_data[key]['freshness_score'] * 0.9
                elif filter_value == 'ukrainian':
                    freshness_data[key]['freshness_score'] = freshness_data[key]['freshness_score'] * 1.1
                elif filter_value == 'western':
                    freshness_data[key]['freshness_score'] = freshness_data[key]['freshness_score'] * 0.95
                elif filter_value == 'military':
                    freshness_data[key]['freshness_score'] = freshness_data[key]['freshness_score'] * 1.05
                elif filter_value == 'social_media':
                    freshness_data[key]['freshness_score'] = freshness_data[key]['freshness_score'] * 1.15
                
                # Resort after adjusting scores
                freshness_data[key] = freshness_data[key].sort_values('freshness_score', ascending=False)
        
        logging.info(f"Demo freshness data generated: {len(freshness_data['category'])} categories, {len(freshness_data['subcategory'])} subcategories")
        return freshness_data
    
    # Calculate total days for normalization
    total_days = (end_date - start_date).days
    total_days = max(total_days, 1)  # Avoid division by zero
    
    # Build appropriate filter conditions
    filter_condition = ""
    params = {'start_date': start_date, 'end_date': end_date}
    
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
    
    # Query for categories - focus on just getting basic data
    query = f"""
    SELECT 
        t.category,
        COUNT(*) as count,
        MAX(ud.date) as latest_date
    FROM taxonomy t
    JOIN document_section_chunk dsc ON t.chunk_id = dsc.id
    JOIN document_section ds ON dsc.document_section_id = ds.id
    JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
    WHERE ud.date BETWEEN :start_date AND :end_date
    {filter_condition}
    GROUP BY t.category
    ORDER BY latest_date DESC, count DESC
    """
    
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Fetch category data
            df_category = pd.read_sql(text(query), conn, params=params)
            
            # Convert latest_date to datetime if it's not already
            if 'latest_date' in df_category.columns and not df_category.empty:
                df_category['latest_date'] = pd.to_datetime(df_category['latest_date'])
            
            # Calculate average age based on end_date and latest_date
            if not df_category.empty and 'latest_date' in df_category.columns:
                end_date_normalized = pd.to_datetime(end_date)
                df_category['avg_age_days'] = (end_date_normalized - df_category['latest_date']).dt.days
            else:
                df_category['avg_age_days'] = 0
            
            # Similar query for subcategories
            query_subcategory = f"""
            SELECT 
                t.category,
                t.subcategory,
                COUNT(*) as count,
                MAX(ud.date) as latest_date
            FROM taxonomy t
            JOIN document_section_chunk dsc ON t.chunk_id = dsc.id
            JOIN document_section ds ON dsc.document_section_id = ds.id
            JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
            WHERE ud.date BETWEEN :start_date AND :end_date
            {filter_condition}
            GROUP BY t.category, t.subcategory
            ORDER BY latest_date DESC, count DESC
            """
            
            df_subcategory = pd.read_sql(text(query_subcategory), conn, params=params)
            
            # Convert latest_date to datetime
            if 'latest_date' in df_subcategory.columns and not df_subcategory.empty:
                df_subcategory['latest_date'] = pd.to_datetime(df_subcategory['latest_date'])
            
            # Calculate average age for subcategories
            if not df_subcategory.empty and 'latest_date' in df_subcategory.columns:
                df_subcategory['avg_age_days'] = (end_date_normalized - df_subcategory['latest_date']).dt.days
            else:
                df_subcategory['avg_age_days'] = 0
            
            # Calculate freshness scores
            # For categories
            max_count = df_category['count'].max() if not df_category.empty else 1
            max_count = max(max_count, 1)  # Avoid division by zero
            
            df_category['recency_score'] = (1 - df_category['avg_age_days'] / total_days) * 70
            df_category['frequency_score'] = (df_category['count'] / max_count) * 30
            df_category['freshness_score'] = df_category['recency_score'] + df_category['frequency_score']
            
            # For subcategories
            max_count_sub = df_subcategory['count'].max() if not df_subcategory.empty else 1
            max_count_sub = max(max_count_sub, 1)  # Avoid division by zero
            
            df_subcategory['recency_score'] = (1 - df_subcategory['avg_age_days'] / total_days) * 70
            df_subcategory['frequency_score'] = (df_subcategory['count'] / max_count_sub) * 30
            df_subcategory['freshness_score'] = df_subcategory['recency_score'] + df_subcategory['frequency_score']
            
            # Sort by freshness score
            df_category = df_category.sort_values('freshness_score', ascending=False)
            df_subcategory = df_subcategory.sort_values('freshness_score', ascending=False)
            
            logging.info(f"Freshness data fetched: {len(df_category)} categories, {len(df_subcategory)} subcategories")
            
            return {
                'category': df_category,
                'subcategory': df_subcategory
            }
            
    except Exception as e:
        logging.error(f"Error fetching freshness data: {e}")
        logging.info("Falling back to demo data for freshness data")
        return _generate_sample_freshness_data()