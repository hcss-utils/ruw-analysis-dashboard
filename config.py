#!/usr/bin/env python
# coding: utf-8

"""
Configuration for Russian-Ukrainian War Data Analysis Dashboard
"""

import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# App information
APP_VERSION = "1.1.0"  # For display in About modal
REPORT_PDF_URL = "https://example.com/report.pdf"  # Replace with actual URL
STATIC_HTML_URL = "https://example.com/static_analysis.html"  # Replace with actual URL

# Authentication configuration
VALID_USERNAME_PASSWORD_PAIRS = {
    'RuBase': 'StratBase$25'
}

# Database configuration - in production, use environment variables
DB_CONFIG = {
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD'),
    'host': os.environ.get('DB_HOST'),
    'port': os.environ.get('DB_PORT'),
    'database': os.environ.get('DB_NAME'),
    'pool_size': 5,
    'max_overflow': 10,
    'pool_timeout': 30,
    'pool_recycle': 1800  # Recycle connections after 30 minutes
}

# Colors for consistent visualization
THEME_COLORS = {
    'russian': '#E41A1C',  # Red
    'western': '#377EB8',  # Blue
    'neutral': '#4DAF4A',  # Green
    'background': '#f8f9fa',
    'text': '#343a40',
    'EN': '#1f77b4',       # English 
    'RU': '#ff7f0e',       # Russian
    'UK': '#2ca02c',       # Ukrainian
    'DE': '#d62728',       # German
    'FR': '#9467bd',       # French
    'ES': '#8c564b',       # Spanish
    'IT': '#e377c2',       # Italian
    'NL': '#7f7f7f',       # Dutch
    'PT': '#bcbd22',       # Portuguese
    'AR': '#17becf',       # Arabic
    'ZH': '#aec7e8',       # Chinese
    'OTHER': '#c7c7c7'     # Other languages
}

# Default dates
DEFAULT_START_DATE = datetime(2022, 1, 1)
DEFAULT_END_DATE = datetime.now()

# Pagination settings
DEFAULT_PAGE_SIZE = 10
SEARCH_RESULT_LIMIT = 500

# Mapping for more friendly display names
DATABASE_DISPLAY_MAP = {
    'telegram_War correspondent': 'telegram_milbloggers',
    'telegram_Official': 'telegram_official'
}

# Source type filter definitions
SOURCE_TYPE_FILTERS = {
    'Primary': "(ud.database LIKE 'telegram%' OR ud.database = 'vk')",
    'Military': "(ud.database LIKE '%military%' OR ud.database LIKE '%mil%')",
    'Scholarly': "(ud.database LIKE '%journal%' OR ud.database LIKE '%academic%')",
    'Social Media': "(ud.database LIKE 'telegram%' OR ud.database = 'twitter' OR ud.database = 'vk')"
}

# Cache settings
CACHE_TIMEOUT = 300  # 5 minutes
CACHE_ENABLED = True

# Default dropdown options
LANGUAGE_OPTIONS = [
    {'label': 'Both English and Russian', 'value': 'ALL'},
    {'label': 'Russian', 'value': 'RU'},
    {'label': 'English', 'value': 'EN'}
]

SOURCE_TYPE_OPTIONS = [
    {'label': 'All Sources', 'value': 'ALL'},
    {'label': 'Primary Sources', 'value': 'Primary'},
    {'label': 'Military Publications', 'value': 'Military'},
    {'label': 'Scholarly Sources', 'value': 'Scholarly'},
    {'label': 'Social Media', 'value': 'Social Media'}
]

FRESHNESS_PERIOD_OPTIONS = [
    {'label': 'Last Week', 'value': 'week'},
    {'label': 'Last Month', 'value': 'month'},
    {'label': 'Last Quarter', 'value': 'quarter'}
]

FRESHNESS_DATATYPE_OPTIONS = [
    {'label': 'Taxonomy Elements', 'value': 'taxonomy'},
    {'label': 'Keywords', 'value': 'keywords'},
    {'label': 'Named Entities', 'value': 'named_entities'}
]

FRESHNESS_FILTER_OPTIONS = [
    {'label': 'All Sources', 'value': 'all'},
    {'label': 'Russian Sources', 'value': 'russian'},
    {'label': 'Ukrainian Sources', 'value': 'ukrainian'},
    {'label': 'Western Sources', 'value': 'western'},
    {'label': 'Military Publications', 'value': 'military'},
    {'label': 'Social Media', 'value': 'social_media'}
]

COMPARISON_VISUALIZATION_OPTIONS = [
    {'label': 'Difference in Means', 'value': 'diff_means'},
    {'label': 'Parallel Stacked Bars', 'value': 'parallel'},
    {'label': 'Radar Chart', 'value': 'radar'},
    {'label': 'Sankey Diagram', 'value': 'sankey'},
    {'label': 'Heatmap Comparison', 'value': 'heatmap'},
    {'label': 'Sunburst Charts', 'value': 'sunburst'}
]