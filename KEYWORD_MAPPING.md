# Keyword Mapping and Consolidation

This document explains the keyword mapping and consolidation functionality implemented in the dashboard.

## Overview

The keyword mapping system standardizes variations of keywords and entities, ensuring consistent analysis across the dashboard. It also filters out noise terms that don't provide meaningful insight.

## Features

1. **Keyword Standardization**: Maps variations of the same entity to a canonical form
   - Example: "US", "USA", "United States" → "united states"
   - Example: "Vladimir Putin", "Putin", "V. Putin" → "vladimir putin"

2. **Noise Term Filtering**: Excludes common noise terms and technical artifacts
   - Filters out terms like "https", "www", "html", "div", "btn", etc.
   - Prevents these non-meaningful terms from appearing in analysis results

3. **Frequency Aggregation**: Combines frequency counts for mapped terms
   - After mapping, frequencies of all variants are aggregated to their canonical form
   - Improves trend detection by considering all variations together

## Implementation Details

The implementation consists of:

1. **Mapping Files**: 
   - `data/consolidated_keywords.csv` - Contains canonical keyword forms and frequencies
   - `data/cross_type_entities.csv` - Maps variant terms to canonical forms

2. **Utility Module**: 
   - `utils/keyword_mapping.py` - Core logic for keyword mapping
   - Provides functions for mapping individual keywords and DataFrames

3. **Integration Points**:
   - Keyword mapping is applied in search functionality
   - Burst detection uses mapped keywords
   - All visualizations display the mapped, canonical forms

## API

Key functions in the `utils/keyword_mapping.py` module:

- `map_keyword(keyword)` - Maps a single keyword to its canonical form
- `map_keywords(keywords)` - Maps a list of keywords to canonical forms
- `remap_and_aggregate_frequencies(df)` - Remaps and aggregates frequencies in a DataFrame
- `load_mapping_files()` - Loads mapping files from the data directory
- `get_mapping_status()` - Returns statistics about loaded mappings

## Sample Mappings

The system includes mappings for key entities in the Russia-Ukraine conflict:

- Countries: Russia, Ukraine, United States, United Kingdom, China
- Organizations: NATO, European Union, United Nations
- People: Vladimir Putin, Volodymyr Zelensky
- Regions: Donbas, Crimea, Kyiv

## Customization

To add new mappings:

1. Edit the sample mapping files generation in `create_sample_mapping_files.py`
2. Run the script to generate updated mapping files
3. The system will automatically use the updated mappings

## Heroku Deployment

The mapping files are part of the repository and will be included in the Heroku deployment. The system works in both demo mode and with a connected database.