"""
Utility functions for working with concordance tables.
This module provides functions to load, process, and search the consolidated CSV files,
as well as match elements between the dashboard and the concordance tables.
"""

import pandas as pd
import os
import re
from typing import List, Dict, Union, Optional, Tuple
import numpy as np


def load_concordance_table(file_path: str) -> pd.DataFrame:
    """
    Load a concordance table from a CSV file.
    
    Args:
        file_path: Path to the CSV file containing the concordance data
        
    Returns:
        DataFrame containing the concordance data
        
    Raises:
        FileNotFoundError: If the specified file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Concordance file not found: {file_path}")
    
    return pd.read_csv(file_path)


def load_keyword_concordance() -> pd.DataFrame:
    """
    Load the consolidated keywords concordance table.
    
    Returns:
        DataFrame containing the keyword concordance data
    """
    return load_concordance_table("consolidated_keywords.csv")


def load_entity_concordance() -> pd.DataFrame:
    """
    Load the cross-type entities concordance table.
    
    Returns:
        DataFrame containing the entity concordance data
    """
    return load_concordance_table("cross_type_entities.csv")


def search_concordance(df: pd.DataFrame, search_term: str, 
                      columns: Optional[List[str]] = None, 
                      case_sensitive: bool = False) -> pd.DataFrame:
    """
    Search for a term across specified columns in a concordance table.
    
    Args:
        df: The concordance DataFrame to search
        search_term: The term to search for
        columns: Specific columns to search (if None, searches all columns)
        case_sensitive: Whether to perform a case-sensitive search
        
    Returns:
        DataFrame with rows matching the search criteria
    """
    if columns is None:
        columns = df.columns
    
    # Create a mask for each column
    masks = []
    for col in columns:
        if pd.api.types.is_string_dtype(df[col]):
            if case_sensitive:
                masks.append(df[col].str.contains(search_term, na=False))
            else:
                masks.append(df[col].str.contains(search_term, case=False, na=False))
    
    # Combine masks with OR operation
    if masks:
        final_mask = masks[0]
        for mask in masks[1:]:
            final_mask = final_mask | mask
        return df[final_mask]
    
    return pd.DataFrame()


def match_entity_to_concordance(entity: str, entity_type: str, 
                               concordance_df: Optional[pd.DataFrame] = None) -> List[Dict]:
    """
    Match a named entity to the concordance table.
    
    Args:
        entity: The entity string to match
        entity_type: The type of the entity (e.g., 'PERSON', 'ORG', 'LOC')
        concordance_df: Optional pre-loaded concordance DataFrame
        
    Returns:
        List of matching records from the concordance table as dictionaries
    """
    if concordance_df is None:
        concordance_df = load_entity_concordance()
    
    # Try exact match first
    matches = concordance_df[(concordance_df['entity'] == entity) & 
                            (concordance_df['type'] == entity_type)]
    
    # If no exact matches, try case-insensitive
    if len(matches) == 0:
        matches = concordance_df[(concordance_df['entity'].str.lower() == entity.lower()) & 
                                (concordance_df['type'] == entity_type)]
    
    # If still no matches, try fuzzy matching
    if len(matches) == 0:
        # Simple fuzzy match - entities containing the search term
        matches = concordance_df[(concordance_df['entity'].str.contains(entity, case=False)) & 
                                (concordance_df['type'] == entity_type)]
    
    return matches.to_dict('records')


def match_keyword_to_concordance(keyword: str, 
                                concordance_df: Optional[pd.DataFrame] = None) -> List[Dict]:
    """
    Match a keyword to the concordance table.
    
    Args:
        keyword: The keyword string to match
        concordance_df: Optional pre-loaded concordance DataFrame
        
    Returns:
        List of matching records from the concordance table as dictionaries
    """
    if concordance_df is None:
        concordance_df = load_keyword_concordance()
    
    # Try exact match first
    matches = concordance_df[concordance_df['keyword'] == keyword]
    
    # If no exact matches, try case-insensitive
    if len(matches) == 0:
        matches = concordance_df[concordance_df['keyword'].str.lower() == keyword.lower()]
    
    # If still no matches, try fuzzy matching
    if len(matches) == 0:
        # Simple fuzzy match - keywords containing the search term
        matches = concordance_df[concordance_df['keyword'].str.contains(keyword, case=False)]
    
    return matches.to_dict('records')


def get_concordance_group(item: str, item_type: str, 
                         concordance_dfs: Optional[Dict[str, pd.DataFrame]] = None) -> List[str]:
    """
    Get all items in the same concordance group as the provided item.
    
    Args:
        item: The keyword or entity to find groups for
        item_type: Type of the item ('keyword' or entity type like 'PERSON', 'ORG', etc.)
        concordance_dfs: Optional dict with pre-loaded concordance DataFrames
        
    Returns:
        List of all items in the same group(s)
    """
    if concordance_dfs is None:
        concordance_dfs = {
            'keyword': load_keyword_concordance(),
            'entity': load_entity_concordance()
        }
    
    if item_type.lower() == 'keyword':
        matches = match_keyword_to_concordance(item, concordance_dfs['keyword'])
        if not matches:
            return [item]  # Return the original item if no matches
        
        # Get all keywords in the same group(s)
        group_ids = [match.get('group_id') for match in matches if 'group_id' in match]
        if not group_ids:
            return [item]
        
        grouped_items = concordance_dfs['keyword'][
            concordance_dfs['keyword']['group_id'].isin(group_ids)
        ]['keyword'].unique().tolist()
        
        return grouped_items
    else:
        # For entities
        matches = match_entity_to_concordance(item, item_type, concordance_dfs['entity'])
        if not matches:
            return [item]  # Return the original item if no matches
        
        # Get all entities in the same group(s)
        group_ids = [match.get('group_id') for match in matches if 'group_id' in match]
        if not group_ids:
            return [item]
        
        grouped_items = concordance_dfs['entity'][
            (concordance_dfs['entity']['group_id'].isin(group_ids)) &
            (concordance_dfs['entity']['type'] == item_type)
        ]['entity'].unique().tolist()
        
        return grouped_items


def normalize_concordance_item(item: str, item_type: str,
                             concordance_dfs: Optional[Dict[str, pd.DataFrame]] = None) -> str:
    """
    Get the canonical/normalized form of an item from the concordance table.
    
    Args:
        item: The keyword or entity to normalize
        item_type: Type of the item ('keyword' or entity type like 'PERSON', 'ORG', etc.)
        concordance_dfs: Optional dict with pre-loaded concordance DataFrames
        
    Returns:
        The canonical form of the item, or the original item if not found
    """
    if concordance_dfs is None:
        concordance_dfs = {
            'keyword': load_keyword_concordance(),
            'entity': load_entity_concordance()
        }
    
    if item_type.lower() == 'keyword':
        matches = match_keyword_to_concordance(item, concordance_dfs['keyword'])
        if not matches:
            return item  # Return the original item if no matches
        
        # Look for the canonical form in the matches
        for match in matches:
            if match.get('is_canonical', False):
                return match.get('keyword', item)
        
        # If no canonical form found, return the first match or the original
        return matches[0].get('keyword', item) if matches else item
    else:
        # For entities
        matches = match_entity_to_concordance(item, item_type, concordance_dfs['entity'])
        if not matches:
            return item  # Return the original item if no matches
        
        # Look for the canonical form in the matches
        for match in matches:
            if match.get('is_canonical', False):
                return match.get('entity', item)
        
        # If no canonical form found, return the first match or the original
        return matches[0].get('entity', item) if matches else item


def aggregate_frequencies(items: List[Dict], 
                        use_concordance: bool = True) -> List[Dict]:
    """
    Aggregate frequency data for items using concordance tables.
    
    Args:
        items: List of dictionaries containing item data with frequencies
        use_concordance: Whether to use concordance tables for grouping
        
    Returns:
        List of dictionaries with aggregated frequencies
    """
    if not use_concordance:
        return items
    
    # Load concordance tables
    concordance_dfs = {
        'keyword': load_keyword_concordance(),
        'entity': load_entity_concordance()
    }
    
    # Group items by their canonical form
    grouped_items = {}
    
    for item in items:
        item_text = item.get('text', '')
        item_type = item.get('type', 'keyword')
        
        # Get canonical form
        canonical = normalize_concordance_item(item_text, item_type, concordance_dfs)
        
        if canonical not in grouped_items:
            grouped_items[canonical] = {
                'text': canonical,
                'type': item_type,
                'frequency': 0,
                'documents': set(),
                'original_items': []
            }
        
        # Aggregate frequencies and document counts
        grouped_items[canonical]['frequency'] += item.get('frequency', 0)
        grouped_items[canonical]['documents'].update(item.get('documents', set()))
        grouped_items[canonical]['original_items'].append(item_text)
    
    # Convert back to list format
    result = []
    for canonical, data in grouped_items.items():
        result.append({
            'text': canonical,
            'type': data['type'],
            'frequency': data['frequency'],
            'document_count': len(data['documents']),
            'documents': list(data['documents']),
            'variants': list(set(data['original_items']))
        })
    
    # Sort by frequency descending
    result.sort(key=lambda x: x['frequency'], reverse=True)
    
    return result