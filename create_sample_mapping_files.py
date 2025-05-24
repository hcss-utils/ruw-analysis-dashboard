"""
Create sample mapping files for keyword consolidation.

This script generates sample mapping files based on the existing frequency files
and creates the necessary files in the 'data' directory.
"""

import os
import pandas as pd
import logging
from typing import Dict, List, Set
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Input files
KEYWORD_FREQ_FILE = 'keyword_frequencies_20250509_031325.csv'
ENTITY_FREQ_FILE1 = 'named_entity_frequencies_20250509_033837.csv'
ENTITY_FREQ_FILE2 = 'named_entity_frequencies_20250509_034335.csv'

# Output files
DATA_DIR = 'data'
CONSOLIDATED_KEYWORDS_FILE = os.path.join(DATA_DIR, 'consolidated_keywords.csv')
CROSS_TYPE_ENTITIES_FILE = os.path.join(DATA_DIR, 'cross_type_entities.csv')

# Words to exclude (noise terms)
EXCLUDE_WORDS = {
    'https', 'http', 'www', 'com', 'org', 'bi', 'li', 'rt', 
    'amp', 'html', 'btn', 'src', 'rel', 'alt', 'jpg', 'png',
    'img', 'div', 'span', 'href', 'class', 'script', 'style'
}

# Sample mappings of entity variants to canonical forms
SAMPLE_ENTITY_MAPPINGS = {
    'us': 'united states',
    'usa': 'united states',
    'america': 'united states',
    'united states of america': 'united states',
    'u.s.': 'united states',
    'u.s.a.': 'united states',
    
    'uk': 'united kingdom',
    'britain': 'united kingdom',
    'great britain': 'united kingdom',
    
    'ru': 'russia',
    'russian federation': 'russia',
    'russia federation': 'russia',
    'russians': 'russia',
    
    'ua': 'ukraine',
    'ukrainians': 'ukraine',
    
    'putin': 'vladimir putin',
    'vladimir vladimirovich putin': 'vladimir putin',
    'v. putin': 'vladimir putin',
    
    'zelensky': 'volodymyr zelensky',
    'zelenskyy': 'volodymyr zelensky',
    'volodymyr zelenskyy': 'volodymyr zelensky',
    'v. zelensky': 'volodymyr zelensky',
    
    'nato': 'nato',
    'north atlantic treaty organization': 'nato',
    
    'eu': 'european union',
    'european commission': 'european union',
    
    'un': 'united nations',
    'united nation': 'united nations',
    
    'china': 'china',
    'prc': 'china',
    'peoples republic of china': 'china',
    "people's republic of china": 'china',
    'chinese': 'china',
    
    'donbas': 'donbas region',
    'donbass': 'donbas region',
    
    'crimea': 'crimea',
    'crimean peninsula': 'crimea',
    
    'kiev': 'kyiv',
    'kyiv': 'kyiv',
    
    'mod_russia': 'russian ministry of defense',
    'russian mod': 'russian ministry of defense',
    'ministry of defense of russia': 'russian ministry of defense',
    
    'mod_ukraine': 'ukrainian ministry of defense',
    'ukrainian mod': 'ukrainian ministry of defense',
    'ministry of defense of ukraine': 'ukrainian ministry of defense',
}

def load_frequency_data() -> Dict[str, pd.DataFrame]:
    """Load all frequency data files."""
    dfs = {}
    
    # Check if files exist
    if os.path.exists(KEYWORD_FREQ_FILE):
        dfs['keyword'] = pd.read_csv(KEYWORD_FREQ_FILE)
        logging.info(f"Loaded {len(dfs['keyword'])} keywords from {KEYWORD_FREQ_FILE}")
    else:
        logging.warning(f"File not found: {KEYWORD_FREQ_FILE}")
        dfs['keyword'] = pd.DataFrame(columns=['Keyword', 'Count', 'Relative Frequency (%)'])
    
    # Load entity files if they exist
    entity_dfs = []
    for entity_file in [ENTITY_FREQ_FILE1, ENTITY_FREQ_FILE2]:
        if os.path.exists(entity_file):
            df = pd.read_csv(entity_file)
            entity_dfs.append(df)
            logging.info(f"Loaded {len(df)} entities from {entity_file}")
        else:
            logging.warning(f"File not found: {entity_file}")
    
    # Combine entity dataframes if any exist
    if entity_dfs:
        dfs['entity'] = pd.concat(entity_dfs, ignore_index=True)
        # Deduplicate entities
        if 'Entity' in dfs['entity'].columns and 'Count' in dfs['entity'].columns:
            dfs['entity'] = dfs['entity'].groupby('Entity').sum().reset_index()
    else:
        dfs['entity'] = pd.DataFrame(columns=['Entity', 'Count', 'Relative Frequency (%)'])
    
    return dfs

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase and trim
    text = text.lower().strip()
    
    # Remove non-alphanumeric characters (except spaces)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    return text

def create_consolidated_keywords(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create consolidated keywords DataFrame."""
    
    # Extract keywords and entities
    keywords = {}
    
    # Process keywords
    if 'keyword' in dfs and 'Keyword' in dfs['keyword'].columns and 'Count' in dfs['keyword'].columns:
        for _, row in dfs['keyword'].iterrows():
            keyword = str(row['Keyword']).strip().lower()
            count = row['Count']
            
            # Skip excluded words
            if keyword in EXCLUDE_WORDS or any(keyword.startswith(ex) for ex in EXCLUDE_WORDS):
                continue
                
            # Apply mapping if exists
            canonical = SAMPLE_ENTITY_MAPPINGS.get(keyword, keyword)
            
            if canonical in keywords:
                keywords[canonical] += count
            else:
                keywords[canonical] = count
    
    # Process entities
    if 'entity' in dfs and 'Entity' in dfs['entity'].columns and 'Count' in dfs['entity'].columns:
        for _, row in dfs['entity'].iterrows():
            entity = str(row['Entity']).strip().lower()
            count = row['Count']
            
            # Skip excluded words
            if entity in EXCLUDE_WORDS or any(entity.startswith(ex) for ex in EXCLUDE_WORDS):
                continue
                
            # Apply mapping if exists
            canonical = SAMPLE_ENTITY_MAPPINGS.get(entity, entity)
            
            if canonical in keywords:
                keywords[canonical] += count
            else:
                keywords[canonical] = count
    
    # Create DataFrame
    result = []
    for canonical, count in keywords.items():
        result.append({
            'canonical_keyword': canonical,
            'frequency': count
        })
    
    # Convert to DataFrame and sort by frequency
    df_result = pd.DataFrame(result)
    if not df_result.empty:
        df_result = df_result.sort_values(by='frequency', ascending=False)
    
    return df_result

def create_cross_type_entities() -> pd.DataFrame:
    """Create cross-type entities DataFrame based on sample mappings."""
    result = []
    
    for variant, canonical in SAMPLE_ENTITY_MAPPINGS.items():
        if variant != canonical:  # Only include actual mappings, not self-mappings
            result.append({
                'variant': variant,
                'canonical_form': canonical
            })
    
    return pd.DataFrame(result)

def main():
    """Main function to create sample mapping files."""
    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logging.info(f"Created directory: {DATA_DIR}")
    
    # Load frequency data
    dfs = load_frequency_data()
    
    # Create consolidated keywords
    df_consolidated = create_consolidated_keywords(dfs)
    df_consolidated.to_csv(CONSOLIDATED_KEYWORDS_FILE, index=False)
    logging.info(f"Created consolidated keywords file with {len(df_consolidated)} entries: {CONSOLIDATED_KEYWORDS_FILE}")
    
    # Create cross-type entities
    df_cross_type = create_cross_type_entities()
    df_cross_type.to_csv(CROSS_TYPE_ENTITIES_FILE, index=False)
    logging.info(f"Created cross-type entities file with {len(df_cross_type)} entries: {CROSS_TYPE_ENTITIES_FILE}")

if __name__ == "__main__":
    main()