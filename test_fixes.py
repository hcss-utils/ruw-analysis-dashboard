#!/usr/bin/env python
# coding: utf-8

"""
Test script to verify all fixes are working correctly
"""

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Test imports
print("Testing imports...")
try:
    from tabs.search import create_search_tab_layout
    print("✓ Search tab imported successfully")
except Exception as e:
    print(f"✗ Search tab import failed: {e}")

try:
    from tabs.compare import convert_keywords_to_comparison_format, convert_entities_to_comparison_format
    print("✓ Compare tab functions imported successfully")
except Exception as e:
    print(f"✗ Compare tab import failed: {e}")

try:
    from tabs.burstiness import create_burstiness_tab_layout
    print("✓ Burstiness tab imported successfully")
except Exception as e:
    print(f"✗ Burstiness tab import failed: {e}")

try:
    from tabs.sources import create_sources_tab_layout
    print("✓ Sources tab imported successfully")
except Exception as e:
    print(f"✗ Sources tab import failed: {e}")

# Test keyword conversion
print("\nTesting keyword conversion to hierarchical format...")
test_keywords = {
    'top_keywords': {
        'labels': ['NATO', 'sanctions', 'artillery', 'negotiations', 'humanitarian'],
        'values': [100, 80, 60, 40, 20]
    }
}

try:
    df = convert_keywords_to_comparison_format(test_keywords)
    print(f"✓ Converted {len(df)} keywords")
    print("Sample structure:")
    for i, row in df.head(3).iterrows():
        print(f"  - {row['category']} > {row['subcategory']} > {row['sub_subcategory']}")
except Exception as e:
    print(f"✗ Keyword conversion failed: {e}")

# Test entity conversion
print("\nTesting entity conversion to hierarchical format...")
test_entities = {
    'top_entities': {
        'labels': ['Ukraine', 'NATO', 'Putin', 'EU', 'Moscow'],
        'types': ['GPE', 'ORG', 'PERSON', 'ORG', 'GPE'],
        'values': [150, 120, 90, 60, 30]
    }
}

try:
    df = convert_entities_to_comparison_format(test_entities)
    print(f"✓ Converted {len(df)} entities")
    print("Sample structure:")
    for i, row in df.head(3).iterrows():
        print(f"  - {row['category']} > {row['subcategory']} > {row['sub_subcategory']}")
except Exception as e:
    print(f"✗ Entity conversion failed: {e}")

# Test layout creation
print("\nTesting layout creation...")
try:
    search_layout = create_search_tab_layout([{'label': 'All', 'value': 'ALL'}])
    # Check if search-stats-container has no initial children
    stats_container = None
    for component in search_layout.children:
        if hasattr(component, 'id') and component.id == 'search-stats-container':
            stats_container = component
            break
    
    if stats_container and hasattr(stats_container, 'children'):
        if stats_container.children:
            print("✗ Search stats container has initial children (should be empty)")
        else:
            print("✓ Search stats container is properly empty")
    else:
        print("✓ Search layout created successfully")
except Exception as e:
    print(f"✗ Search layout creation failed: {e}")

print("\nAll tests completed!")