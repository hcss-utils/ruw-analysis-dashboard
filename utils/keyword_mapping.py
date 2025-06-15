"""
Keyword mapping utility module.

This module handles the loading and application of keyword mapping rules
from concordance tables for keyword consolidation and standardization.
"""

import os
import pandas as pd
import logging
from typing import Dict, List, Optional, Set, Tuple
import pathlib

# Default paths to mapping files
DEFAULT_CONSOLIDATED_KEYWORDS_PATH = os.path.join('data', 'consolidated_keywords.csv')
DEFAULT_CROSS_TYPE_ENTITIES_PATH = os.path.join('data', 'cross_type_entities.csv')

# List of keywords to exclude (noisy or non-meaningful terms)
DEFAULT_EXCLUDE_KEYWORDS = {
    'https', 'http', 'www', 'com', 'org', 'bi', 'li', 'rt', 
    'amp', 'html', 'btn', 'src', 'rel', 'alt', 'jpg', 'png',
    'img', 'div', 'span', 'href', 'class', 'script', 'style'
}

# Global variables to store mappings
keyword_to_canonical: Dict[str, str] = {}
canonical_keywords: Set[str] = set()
excluded_keywords: Set[str] = DEFAULT_EXCLUDE_KEYWORDS

def load_mapping_files(
    consolidated_path: Optional[str] = None,
    cross_type_path: Optional[str] = None,
    exclude_keywords: Optional[Set[str]] = None,
    max_memory_mb: int = 100
) -> Tuple[bool, str]:
    """
    Load keyword mapping files with memory limit.
    
    Args:
        consolidated_path: Path to consolidated keywords CSV
        cross_type_path: Path to cross-type entities CSV
        exclude_keywords: Set of keywords to exclude
        max_memory_mb: Maximum memory to use in MB (default 100MB)
        
    Returns:
        Tuple of (success, message)
    """
    global keyword_to_canonical, canonical_keywords, excluded_keywords
    
    # Set paths or use defaults
    consolidated_path = consolidated_path or DEFAULT_CONSOLIDATED_KEYWORDS_PATH
    cross_type_path = cross_type_path or DEFAULT_CROSS_TYPE_ENTITIES_PATH
    
    # Override excluded keywords if provided
    if exclude_keywords is not None:
        excluded_keywords = exclude_keywords
    
    # Reset mappings
    keyword_to_canonical = {}
    canonical_keywords = set()
    
    # Initialize memory tracking
    import sys
    current_memory = 0
    max_entries = 0
    max_memory_bytes = max_memory_mb * 1024 * 1024
    
    try:
        # Check if files exist
        consolidated_exists = os.path.exists(consolidated_path)
        cross_type_exists = os.path.exists(cross_type_path)
        
        if not consolidated_exists and not cross_type_exists:
            logging.warning(f"Mapping files not found: {consolidated_path}, {cross_type_path}")
            return False, "Mapping files not found"
        
        # Load consolidated keywords if available
        if consolidated_exists:
            try:
                df_consolidated = pd.read_csv(consolidated_path)
                logging.info(f"Loaded consolidated keywords from {consolidated_path}")
                
                # Sort by frequency if available to keep most important keywords
                if 'frequency' in df_consolidated.columns:
                    df_consolidated = df_consolidated.sort_values('frequency', ascending=False)
                
                # Process consolidated keywords: canonical_keyword,frequency
                if 'canonical_keyword' in df_consolidated.columns:
                    for idx, row in df_consolidated.iterrows():
                        # Handle the case where canonical_keyword might be a float or int
                        if isinstance(row['canonical_keyword'], (int, float)):
                            canonical = str(row['canonical_keyword']).strip().lower()
                        else:
                            canonical = str(row['canonical_keyword']).strip().lower()
                        
                        # Estimate memory usage (rough approximation)
                        entry_size = sys.getsizeof(canonical) * 2  # For both key and value
                        if current_memory + entry_size > max_memory_bytes:
                            logging.info(f"Memory limit reached. Loaded {max_entries} keywords.")
                            break
                        
                        canonical_keywords.add(canonical)
                        # Map canonical to itself
                        keyword_to_canonical[canonical] = canonical
                        current_memory += entry_size
                        max_entries += 1
                else:
                    logging.warning(f"Invalid format in {consolidated_path}, missing 'canonical_keyword' column")
            except Exception as e:
                logging.error(f"Error loading consolidated keywords: {str(e)}")
                return False, f"Error loading consolidated keywords: {str(e)}"
        
        # Load cross-type entities if available
        if cross_type_exists and current_memory < max_memory_bytes:
            try:
                df_cross_type = pd.read_csv(cross_type_path)
                logging.info(f"Loaded cross-type entities from {cross_type_path}")
                
                # Sort by frequency if available
                if 'frequency' in df_cross_type.columns:
                    df_cross_type = df_cross_type.sort_values('frequency', ascending=False)
                
                # Extract mapping data from cross-type entities
                # Assuming format: variant,canonical or similar
                # Adapt this based on the actual format of the file
                if len(df_cross_type.columns) >= 2:
                    variant_col = df_cross_type.columns[0]
                    canonical_col = df_cross_type.columns[1]
                    
                    for idx, row in df_cross_type.iterrows():
                        # Handle the case where values might be numeric
                        variant = str(row[variant_col]).strip().lower()
                        canonical = str(row[canonical_col]).strip().lower()
                        
                        if variant and canonical:
                            # Check memory limit
                            entry_size = sys.getsizeof(variant) + sys.getsizeof(canonical)
                            if current_memory + entry_size > max_memory_bytes:
                                logging.info(f"Memory limit reached. Total loaded: {max_entries} entries.")
                                break
                            
                            keyword_to_canonical[variant] = canonical
                            canonical_keywords.add(canonical)
                            current_memory += entry_size
                            max_entries += 1
                else:
                    logging.warning(f"Invalid format in {cross_type_path}, expected at least 2 columns")
            except Exception as e:
                logging.error(f"Error loading cross-type entities: {str(e)}")
                return False, f"Error loading cross-type entities: {str(e)}"
        
        memory_used_mb = current_memory / (1024 * 1024)
        logging.info(f"Loaded {len(keyword_to_canonical)} keyword mappings and {len(canonical_keywords)} canonical terms")
        logging.info(f"Memory used: {memory_used_mb:.1f}MB of {max_memory_mb}MB limit")
        return True, f"Successfully loaded keyword mappings (using {memory_used_mb:.1f}MB)"
    
    except Exception as e:
        logging.error(f"Error loading mapping files: {str(e)}")
        return False, f"Error loading mapping files: {str(e)}"

def map_keyword(keyword: str) -> Optional[str]:
    """
    Map a keyword to its canonical form.
    
    Args:
        keyword: The keyword to map
        
    Returns:
        The canonical form of the keyword, or None if it should be excluded
    """
    if not keyword:
        return None
    
    # Handle non-string inputs
    if not isinstance(keyword, str):
        keyword = str(keyword)
    
    # Normalize
    norm_keyword = keyword.strip().lower()
    
    # Check if keyword should be excluded
    if norm_keyword in excluded_keywords:
        return None
    
    # Return the canonical form if it exists, otherwise return the original
    return keyword_to_canonical.get(norm_keyword, norm_keyword)

def map_keywords(keywords: List[str]) -> List[str]:
    """
    Map a list of keywords to their canonical forms and filter out excluded keywords.
    
    Args:
        keywords: List of keywords to map
        
    Returns:
        List of mapped keywords with excluded keywords removed
    """
    if not keywords:
        return []
        
    result = []
    for kw in keywords:
        mapped = map_keyword(kw)
        if mapped:
            result.append(mapped)
    return result

def get_canonical_keywords() -> Set[str]:
    """
    Get the set of canonical keywords.
    
    Returns:
        Set of canonical keywords
    """
    return canonical_keywords.copy()

def get_mapping_status() -> Dict:
    """
    Get the status of keyword mappings.
    
    Returns:
        Dictionary with mapping statistics
    """
    return {
        'total_mappings': len(keyword_to_canonical),
        'canonical_keywords': len(canonical_keywords),
        'excluded_keywords': len(excluded_keywords),
    }

def remap_and_aggregate_frequencies(df: pd.DataFrame, keyword_col: str = 'Keyword', 
                                   freq_col: str = 'Count') -> pd.DataFrame:
    """
    Remap keywords in a DataFrame and aggregate their frequencies.
    
    Args:
        df: DataFrame containing keywords and frequencies
        keyword_col: Name of the keyword column
        freq_col: Name of the frequency column
        
    Returns:
        DataFrame with remapped keywords and aggregated frequencies
    """
    if df.empty or keyword_col not in df.columns or freq_col not in df.columns:
        return df
    
    # Create a copy of the input DataFrame
    result_df = pd.DataFrame()
    
    # Map each keyword and create a new DataFrame
    mapped_data = []
    for _, row in df.iterrows():
        keyword = row[keyword_col]
        freq = row[freq_col]
        
        mapped = map_keyword(keyword)
        if mapped:  # Only include non-excluded keywords
            mapped_data.append({
                keyword_col: mapped,
                freq_col: freq
            })
    
    if not mapped_data:
        return pd.DataFrame(columns=df.columns)
    
    # Create DataFrame from mapped data
    result_df = pd.DataFrame(mapped_data)
    
    # Aggregate frequencies by mapped keyword
    result_df = result_df.groupby(keyword_col).sum().reset_index()
    
    # Sort by frequency (descending)
    result_df = result_df.sort_values(by=freq_col, ascending=False)
    
    return result_df

# Try to load mapping files when module is imported
try:
    # Check if we're running from a script or imported as a module
    if __name__ != "__main__":
        success, message = load_mapping_files()
        if not success:
            logging.warning(f"Failed to load keyword mappings on import: {message}")
            logging.warning("Keyword mapping will use identity mapping only.")
except Exception as e:
    logging.error(f"Error initializing keyword mapping module: {str(e)}")