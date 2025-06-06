#!/usr/bin/env python
# coding: utf-8

"""
Test script to check if the app tabs are working correctly.
"""

import logging
logging.basicConfig(level=logging.INFO)

# Test imports
try:
    from tabs.sources import create_sources_tab_layout, register_sources_tab_callbacks
    print("✓ Sources tab imports successfully")
except Exception as e:
    print(f"✗ Sources tab import error: {e}")

try:
    from tabs.compare import create_compare_tab_layout, register_compare_tab_callbacks
    print("✓ Compare tab imports successfully")
except Exception as e:
    print(f"✗ Compare tab import error: {e}")

try:
    from tabs.burstiness import create_burstiness_tab_layout, register_burstiness_tab_callbacks
    print("✓ Burstiness tab imports successfully")
except Exception as e:
    print(f"✗ Burstiness tab import error: {e}")

# Test data fetchers
try:
    from database.data_fetchers_sources import fetch_corpus_stats
    stats = fetch_corpus_stats()
    print(f"✓ Corpus stats fetched: {stats}")
except Exception as e:
    print(f"✗ Corpus stats error: {e}")

# Test layout creation
try:
    layout = create_sources_tab_layout([], None, None)
    print("✓ Sources layout created successfully")
except Exception as e:
    print(f"✗ Sources layout error: {e}")

print("\nAll tests completed!")