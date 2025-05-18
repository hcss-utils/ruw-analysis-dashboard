#!/usr/bin/env python
# coding: utf-8

"""
Layout components for the dashboard.
Provides functions to create reusable UI components.
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

import sys
import os

# Add the project root to the path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import APP_VERSION, LANGUAGE_OPTIONS, SOURCE_TYPE_OPTIONS, REPORT_PDF_URL, STATIC_HTML_URL


# Define paths for logos
LOGO_PATH = "assets/logo.png"
TAB_LOGOS = {
    'explore': "assets/explore_icon.png",
    'search': "assets/search_icon.png",
    'compare': "assets/compare_icon.png",
    'freshness': "assets/freshness_icon.png",
    'sources': "assets/sources_icon.png",
    'russian': "assets/russian_icon.png",
    'western': "assets/western_icon.png"
}

def create_header() -> html.Div:
    """
    Create the dashboard header with title and buttons.
    
    Returns:
        html.Div: Header component
    """
    # Blue color for text and buttons with white background
    blue_color = "#13376f"
    
    return html.Div([
        # Main container with grid layout
        html.Div([
            # Left logo
            html.Div([
                html.A(
                    html.Img(src='/static/rubase_logo_4.svg', height='50px'),
                    href='https://hcss.nl/rubase/',
                    target='_blank'
                ),
            ], style={"grid-area": "logo1"}),
            
            # Title and buttons as one unit
            html.Div([
                html.Span([
                    html.H2("Russian-Ukrainian War Corpus Analysis Dashboard", 
                          style={"display": "inline", "margin": 0, "color": blue_color,
                                "font-size": "1.5rem", "margin-right": "15px"}),
                    dbc.Button(
                        "About", 
                        id="open-about-main", 
                        color="secondary", 
                        size="sm", 
                        style={"margin-right": "10px", "background-color": blue_color, 
                              "border-color": blue_color, "position": "relative", "top": "-4px"}
                    ),
                    dbc.Button(
                        "Clear Cache", 
                        id="clear-cache-button", 
                        color="secondary", 
                        size="sm",
                        style={"background-color": blue_color, "border-color": blue_color, 
                              "position": "relative", "top": "-4px"}
                    ),
                ]),
            ], style={"grid-area": "title", "text-align": "center"}),
            
            # Right logo
            html.Div([
                html.A(
                    html.Img(src='/static/HCSS_Beeldmerk_Blauw_RGB.svg', height='50px'),
                    href='https://hcss.nl/',
                    target='_blank'
                ),
            ], style={"grid-area": "logo2", "text-align": "right"})
        ], style={
            "display": "grid",
            "grid-template-areas": "'logo1 title logo2'",
            "grid-template-columns": "200px 1fr 200px",
            "align-items": "center",
            "width": "100%"
        })
    ], className="app-header", style={
        "background-color": "white",  # Changed to white
        "padding": "15px 20px",
        "width": "100%",
        "border-bottom": f"2px solid {blue_color}"  # Added border for definition
    })

def create_about_modal() -> dbc.Modal:
    """
    Create the About modal with dashboard information.
    
    Returns:
        dbc.Modal: About modal component
    """
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("About this Dashboard")),
        dbc.ModalBody([
            html.P([
                "This dashboard provides tools to explore, search, and compare data related to the Russian-Ukrainian War. ",
                "It analyzes content from various sources to extract taxonomic elements (categories, subcategories, and sub-subcategories) ",
                "that describe different aspects of the conflict."
            ]),
            
            html.Div([
                html.H5("Dashboard Features"),
                html.Ul([
                    html.Li([
                        html.Strong("Explore: "), 
                        "Visualize taxonomic elements using a sunburst chart and explore text chunks"
                    ]),
                    html.Li([
                        html.Strong("Search: "), 
                        "Search through the database with keyword, boolean, or semantic search"
                    ]),
                    html.Li([
                        html.Strong("Compare: "), 
                        "Compare taxonomic distributions between different data slices"
                    ]),
                    html.Li([
                        html.Strong("Freshness: "), 
                        "Analyze how recently different taxonomic elements have been discussed"
                    ]),
                    html.Li([
                        html.Strong("Sources: "), 
                        "View statistical information about the corpus"
                    ])
                ])
            ], className="about-box"),
            
            html.Div([
                html.H5("Data Sources"),
                html.P([
                    "This dashboard analyzes content from various sources including government statements, ",
                    "military publications, news articles, academic papers, and social media posts related to the conflict."
                ])
            ], className="about-box"),
            
            html.Div([
                html.H5("Analysis Process"),
                html.P([
                    "Text data is processed using natural language processing (NLP) techniques to identify taxonomic elements. ",
                    "The dashboard provides multiple visualizations to explore and analyze these elements."
                ])
            ], className="about-box"),
            
            html.Div([
                html.H5("Additional Resources"),
                html.Ul([
                    html.Li([
                        html.A("Download Full Report (PDF)", href=REPORT_PDF_URL, target="_blank")
                    ]),
                    html.Li([
                        html.A("Static Analysis Dashboard", href=STATIC_HTML_URL, target="_blank")
                    ])
                ])
            ], className="about-box"),
            
            html.P([
                f"Version: {APP_VERSION} | Last updated: May 2023"
            ], className="text-muted mt-3 text-center")
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-about", className="ms-auto")
        ),
    ], id="about-modal", size="lg", is_open=False)


def create_filter_card(id_prefix: str, db_options: List, min_date: datetime = None, max_date: datetime = None) -> dbc.Card:
    """
    Create a filter card with dropdowns and date picker.
    
    Args:
        id_prefix: Prefix for component IDs
        db_options: Database options for dropdown
        min_date: Minimum date for date picker
        max_date: Maximum date for date picker
        
    Returns:
        dbc.Card: Filter card component
    """
    return dbc.Card([
        dbc.CardHeader("Data Filters"),
        dbc.CardBody([
            dbc.Row([
                # Language
                dbc.Col([
                    html.Label("Language:"),
                    dcc.Dropdown(
                        id=f'{id_prefix}-language-dropdown',
                        options=LANGUAGE_OPTIONS,
                        value='ALL',
                        placeholder='Select Language',
                        className="language-dropdown",
                        style={"width": "120px"}
                    ),
                ], width="auto", className="pe-2"),
                
                # Database
                dbc.Col([
                    html.Label("Database:"),
                    dcc.Dropdown(
                        id=f'{id_prefix}-database-dropdown',
                        options=db_options,
                        value='ALL',
                        placeholder='Select Database',
                        className="database-dropdown",
                        style={"width": "140px"}
                    ),
                ], width="auto", className="pe-2"),
                
                # Source Type
                dbc.Col([
                    html.Label("Source Type:"),
                    dcc.Dropdown(
                        id=f'{id_prefix}-source-type-dropdown',
                        options=SOURCE_TYPE_OPTIONS,
                        value='ALL',
                        placeholder='Select Source Type',
                        className="source-type-dropdown",
                        style={"width": "140px"}
                    ),
                ], width="auto", className="pe-2"),
                
                # Date Range
                dbc.Col([
                    html.Label("Date Range:"),
                    dcc.DatePickerRange(
                        id=f'{id_prefix}-date-range-picker',
                        start_date=min_date,
                        end_date=max_date,
                        min_date_allowed=min_date,
                        max_date_allowed=max_date,
                        display_format='YY-MM-DD',
                        style={"width": "180px"}
                    ),
                ], width="auto", className="pe-2"),
                
                # Empty column that will expand to push the button to the right
                dbc.Col(className="flex-grow-1"),
                
                # Apply button (right-aligned)
                dbc.Col([
                    # The br tag creates vertical alignment with other elements
                    html.Br(),
                    dbc.Button(
                        'Apply Filters', 
                        id=f'{id_prefix}-filter-button', 
                        color="primary",
                        className="mt-1"
                    ),
                ], width="auto"),
            ], className="g-0 align-items-end"),  # g-0 removes gutters, align-items-end aligns at bottom
            
            html.Div(id=f'{id_prefix}-result-stats', className="mt-2")
        ], className="p-3")
    ], className="mb-4 filter-card")


def create_pagination_controls(id_prefix: str) -> Dict[str, html.Div]:
    """
    Create pagination controls (previous/next buttons and page indicator).
    
    Args:
        id_prefix: Prefix for component IDs
        
    Returns:
        Dict[str, html.Div]: Dictionary with 'top' and 'bottom' pagination controls
    """
    # Create pagination controls for top position
    pagination_top = html.Div([
        dbc.Button('Previous Page', id=f'{id_prefix}-prev-page-button', color="secondary", className="me-2"),
        html.Span("Page 1", id=f'{id_prefix}-page-indicator', style={'margin': '0 10px'}),
        dbc.Button('Next Page', id=f'{id_prefix}-next-page-button', color="secondary"),
    ], id=f'{id_prefix}-top-pagination-controls', style={'margin-bottom': '20px', 'display': 'none', 'justify-content': 'center', 'align-items': 'center'})
    
    # Create pagination controls for bottom position
    pagination_bottom = html.Div([
        dbc.Button('Previous Page', id=f'{id_prefix}-prev-page-button-bottom', color="secondary", className="me-2"),
        html.Span("Page 1", id=f'{id_prefix}-page-indicator-bottom', style={'margin': '0 10px'}),
        dbc.Button('Next Page', id=f'{id_prefix}-next-page-button-bottom', color="secondary"),
    ], id=f'{id_prefix}-bottom-pagination-controls', style={'margin-top': '20px', 'margin-bottom': '20px', 'display': 'none', 'justify-content': 'center', 'align-items': 'center'})
    
    return {'top': pagination_top, 'bottom': pagination_bottom}


def create_download_buttons(id_prefix: str) -> html.Div:
    """
    Create download buttons for exporting data.
    
    Args:
        id_prefix: Prefix for component IDs
        
    Returns:
        html.Div: Download buttons component
    """
    return html.Div([
        dbc.Button("Download CSV", id=f"{id_prefix}-btn-csv", color="success", className="me-2"),
        dbc.Button("Download JSON", id=f"{id_prefix}-btn-json", color="success"),
    ], id=f'{id_prefix}-download-buttons', style={'margin-top': '20px', 'text-align': 'center', 'display': 'none'})


def create_tab_label(tab_name: str) -> html.Div:
    """
    Create a tab label with an icon.
    
    Args:
        tab_name: Name of the tab
        
    Returns:
        html.Div: Tab label component
    """
    return html.Div([
        html.Img(src=TAB_LOGOS.get(tab_name.lower(), ""), className="tab-logo") if tab_name.lower() in TAB_LOGOS else None,
        tab_name
    ], className="d-flex align-items-center")


def create_comparison_layout(
    slice_a_name: str = "Russian",
    slice_b_name: str = "Western",
    db_options: List = None,
    min_date: datetime = None,
    max_date: datetime = None
) -> html.Div:
    """
    Create a comprehensive comparison layout with two filter panels.
    
    Args:
        slice_a_name: Name for Slice A
        slice_b_name: Name for Slice B
        db_options: Database options for dropdowns
        min_date: Minimum date for date pickers
        max_date: Maximum date for date pickers
        
    Returns:
        html.Div: Comparison layout component
    """
    if db_options is None:
        db_options = [{'label': 'All Databases', 'value': 'ALL'}]
    
    # Use default layout color
    blue_color = "#13376f"
    
    return html.Div([
        # Header with title and About button
        html.Div([
            html.H3([
                html.Img(src=TAB_LOGOS.get('compare', ""), className="tab-logo"),
                "Compare Datasets"
            ], style={"display": "inline-block", "margin-right": "20px"}),
            dbc.Button(
                "About", 
                id="open-about-compare", 
                color="secondary", 
                size="sm",
                className="ml-auto",
                style={"display": "inline-block"}
            ),
        ], style={"display": "flex", "align-items": "center", "margin-bottom": "20px"}),
        
        dbc.Row([
            # Filters for Slice A (Russian)
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Img(src=TAB_LOGOS.get('russian', ""), className="tab-logo"),
                        f"{slice_a_name} Data Filters"
                    ], className="d-flex align-items-center"),
                    dbc.CardBody([
                        html.Label("Language:"),
                        dcc.Dropdown(
                            id='compare-language-dropdown-a',
                            options=LANGUAGE_OPTIONS,
                            placeholder='Select Language',
                            value='RU',  # Default to Russian
                            className="mb-2"
                        ),
                        html.Label("Database:"),
                        dcc.Dropdown(
                            id='compare-database-dropdown-a',
                            options=db_options,
                            placeholder='Select Database',
                            value='ALL',
                            className="mb-2"
                        ),
                        html.Label("Source Type:"),
                        dcc.Dropdown(
                            id='compare-source-type-a',
                            options=SOURCE_TYPE_OPTIONS,
                            placeholder='Select Source Type',
                            value='ALL',
                            className="mb-2"
                        ),
                        html.Label("Date Range:"),
                        dcc.DatePickerRange(
                            id='compare-date-picker-a',
                            start_date=min_date,
                            end_date=max_date,
                            min_date_allowed=min_date,
                            max_date_allowed=max_date,
                            with_portal=True,
                            updatemode='singledate',
                            className="mb-2"
                        ),
                        html.Div(id='slice-a-stats', className="mt-2")
                    ])
                ])
            ], width=6),
            
            # Filters for Slice B (Western)
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Img(src=TAB_LOGOS.get('western', ""), className="tab-logo"),
                        f"{slice_b_name} Data Filters"
                    ], className="d-flex align-items-center"),
                    dbc.CardBody([
                        html.Label("Language:"),
                        dcc.Dropdown(
                            id='compare-language-dropdown-b',
                            options=LANGUAGE_OPTIONS,
                            placeholder='Select Language',
                            value='EN',  # Default to English
                            className="mb-2"
                        ),
                        html.Label("Database:"),
                        dcc.Dropdown(
                            id='compare-database-dropdown-b',
                            options=db_options,
                            placeholder='Select Database',
                            value='ALL',
                            className="mb-2"
                        ),
                        html.Label("Source Type:"),
                        dcc.Dropdown(
                            id='compare-source-type-b',
                            options=SOURCE_TYPE_OPTIONS,
                            placeholder='Select Source Type',
                            value='ALL',
                            className="mb-2"
                        ),
                        html.Label("Date Range:"),
                        dcc.DatePickerRange(
                            id='compare-date-picker-b',
                            start_date=min_date,
                            end_date=max_date,
                            min_date_allowed=min_date,
                            max_date_allowed=max_date,
                            with_portal=True,
                            updatemode='singledate',
                            className="mb-2"
                        ),
                        html.Div(id='slice-b-stats', className="mt-2")
                    ])
                ])
            ], width=6)
        ]),
        
        # Comparison controls
        dbc.Card([
            dbc.CardHeader("Comparison Controls"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Comparison Type:"),
                        dcc.Dropdown(
                            id='compare-viz-type',
                            options=[
                                {'label': 'Difference in Means', 'value': 'diff_means'},
                                {'label': 'Parallel Stacked Bars', 'value': 'parallel'},
                                {'label': 'Radar Chart', 'value': 'radar'},
                                {'label': 'Sankey Diagram', 'value': 'sankey'},
                                {'label': 'Heatmap Comparison', 'value': 'heatmap'},
                                {'label': 'Sunburst Charts', 'value': 'sunburst'}
                            ],
                            value='diff_means',
                            className="mb-2"
                        ),
                    ], width=8),
                    dbc.Col([
                        html.Br(),
                        dbc.Button(
                            'Compare', 
                            id='compare-button', 
                            color="primary", 
                            className="mt-2"
                        ),
                    ], width=4, style={"text-align": "right"})
                ]),
            ])
        ], className="my-4"),
        
        # Results section
        html.Div([
            # Side-by-side charts
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        id="loading-compare-a",
                        type="default",
                        children=[dcc.Graph(id='compare-chart-a')]
                    )
                ], width=6, id="compare-col-a"),
                
                dbc.Col([
                    dcc.Loading(
                        id="loading-compare-b",
                        type="default",
                        children=[dcc.Graph(id='compare-chart-b')]
                    )
                ], width=6, id="compare-col-b")
            ], id="side-by-side-charts"),
            
            # Full-width chart
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        id="loading-compare-full",
                        type="default",
                        children=[dcc.Graph(id='compare-chart-full')]
                    )
                ], width=12, id="compare-col-full")
            ], id="full-width-chart"),
            
            # Analysis text
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        id="loading-compare-text",
                        type="default",
                        children=[
                            dbc.Card([
                                dbc.CardHeader("Comparison Analysis"),
                                dbc.CardBody(id='compare-text-comparison')
                            ])
                        ]
                    )
                ], width=12)
            ], className="mt-4")
        ], id="comparison-results", style={"display": "none"}),
        
        # Storage components
        dcc.Store(id='compare-data-a'),
        dcc.Store(id='compare-data-b'),
        dcc.Store(id='current-taxonomy-level'),
        dcc.Store(id='current-parent-category'),
        
        # Modal with comparison guide
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Comparison Visualizations Guide"), 
                           style={"background-color": blue_color, "color": "white"}),
            dbc.ModalBody([
                html.P([
                    "This tab allows you to compare taxonomic elements between two data slices. ",
                    "Choose filters for each slice and select a visualization method."
                ]),
                
                html.H5("Visualization Options:"),
                html.Ul([
                    html.Li([
                        html.Strong("Difference in Means: "), 
                        "Shows the percentage point difference between slices for each category"
                    ]),
                    html.Li([
                        html.Strong("Parallel Stacked Bars: "), 
                        "Shows category proportions side by side with connecting lines"
                    ]),
                    html.Li([
                        html.Strong("Radar Chart: "), 
                        "Displays category distributions in a circular format"
                    ]),
                    html.Li([
                        html.Strong("Sankey Diagram: "), 
                        "Shows the flow of data from slices to categories"
                    ]),
                    html.Li([
                        html.Strong("Heatmap Comparison: "), 
                        "Compares categories using color intensity"
                    ]),
                    html.Li([
                        html.Strong("Sunburst Charts: "), 
                        "Displays two separate sunburst charts for direct comparison"
                    ])
                ]),
                
                html.P([
                    html.Strong("Tip: "),
                    "For the Difference in Means and Parallel Bars visualizations, you can click on a category to drill down into its subcategories."
                ], className="alert alert-info")
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-about-compare", className="ms-auto", 
                          style={"background-color": blue_color, "border": "none"})
            ),
        ], id="compare-about-modal", size="lg", is_open=False)
    ])


def create_search_layout(
    db_options: List = None,
    min_date: datetime = None,
    max_date: datetime = None
) -> html.Div:
    """
    Create a comprehensive search layout with search options.
    
    Args:
        db_options: Database options for dropdown
        min_date: Minimum date for date picker
        max_date: Maximum date for date picker
        
    Returns:
        html.Div: Search layout component
    """
    if db_options is None:
        db_options = [{'label': 'All Databases', 'value': 'ALL'}]
    
    # Use default layout color
    blue_color = "#13376f"
    
    return html.Div([
        # Search header
        html.Div([
            html.H3([
                html.Img(src=TAB_LOGOS.get('search', ""), className="tab-logo"),
                "Search Data"
            ], style={"display": "inline-block", "margin-right": "20px"}),
            dbc.Button(
                "About", 
                id="open-about-search", 
                color="secondary", 
                size="sm",
                className="ml-auto",
                style={"display": "inline-block"}
            ),
        ], style={"display": "flex", "align-items": "center", "margin-bottom": "20px"}),
        
        # Search card
        dbc.Card([
            dbc.CardHeader("Search Options"),
            dbc.CardBody([
                # Search mode selection
                dbc.Row([
                    dbc.Col([
                        html.Label("Search Mode:"),
                        dcc.RadioItems(
                            id='search-mode',
                            options=[
                                {'label': 'Keyword Search', 'value': 'keyword'},
                                {'label': 'Boolean Search', 'value': 'boolean'},
                                {'label': 'Semantic Search (Beta)', 'value': 'semantic'}
                            ],
                            value='keyword',
                            style={'display': 'flex', 'justify-content': 'space-between'}
                        ),
                        dbc.Tooltip(
                            "Choose search mode: Keyword (whole word matching), Boolean (AND, OR, NOT operators), or Semantic (meaning-based)",
                            target="search-mode",
                        )
                    ], width=12)
                ]),
                
                # Search input
                dbc.Row([
                    dbc.Col([
                        html.Label("Search Terms:"),
                        dcc.Input(
                            id='search-input',
                            type='text',
                            placeholder='Enter search term...',
                            style={'width': '100%'}
                        ),
                        html.Div(id='search-syntax-help', style={'font-size': '0.8em', 'margin-top': '5px'})
                    ], width=12)
                ], className="mt-3"),
                
                # Filters
                dbc.Row([
                    dbc.Col([
                        html.Label("Language:"),
                        dcc.Dropdown(
                            id='search-language-dropdown',
                            options=LANGUAGE_OPTIONS,
                            placeholder='Select Language',
                            value='ALL'
                        ),
                    ], width=4),
                    dbc.Col([
                        html.Label("Database:"),
                        dcc.Dropdown(
                            id='search-database-dropdown',
                            options=db_options,
                            placeholder='Select Database',
                            value='ALL'
                        ),
                    ], width=4),
                    dbc.Col([
                        html.Label("Source Type:"),
                        dcc.Dropdown(
                            id='search-source-type-dropdown',
                            options=SOURCE_TYPE_OPTIONS,
                            placeholder='Select Source Type',
                            value='ALL'
                        ),
                    ], width=4)
                ], className="mt-3"),
                
                # Date range and search button
                dbc.Row([
                    dbc.Col([
                        html.Label("Date Range:"),
                        dcc.DatePickerRange(
                            id='search-date-range-picker',
                            start_date=min_date,
                            end_date=max_date,
                            min_date_allowed=min_date,
                            max_date_allowed=max_date,
                            display_format='YY-MM-DD'
                        ),
                    ], width=8),
                    dbc.Col([
                        html.Br(),
                        dbc.Button('Search', id='search-button', color="primary", className="mt-2"),
                    ], width=4, style={"text-align": "right"})
                ], className="mt-3"),
            ])
        ], className="mb-4"),
        
        # Results section
        html.Div(id='search-stats-container', className="mt-4"),
        
        dbc.Tabs([
            dbc.Tab([
                dcc.Loading(
                    id="loading-search-sunburst",
                    type="default",
                    children=[dcc.Graph(id='search-sunburst-chart', style={'height': '700px'})]
                )
            ], label="Category Distribution"),
            
            dbc.Tab([
                dcc.Loading(
                    id="loading-search-timeline",
                    type="default",
                    children=[dcc.Graph(id='search-timeline-chart', style={'height': '400px'})]
                )
            ], label="Timeline")
        ], id="search-results-tabs", style={'display': 'none'}),
        
        # Results container and pagination
        html.Div(id='search-results-header', style={'display': 'none'}),
        html.Div(id='search-chunks-container'),
        
        html.Div([
            dbc.Button('Previous Page', id='search-prev-page-button', color="secondary", className="me-2"),
            html.Span("Page 1", id='search-page-indicator', style={'margin': '0 10px'}),
            dbc.Button('Next Page', id='search-next-page-button', color="secondary"),
        ], id='search-pagination-controls', style={'margin-top': '20px', 'display': 'none', 'justify-content': 'center', 'align-items': 'center'}),
        
        # Download buttons
        html.Div([
            dbc.Button("Download CSV", id="search-btn-csv", color="success", className="me-2"),
            dbc.Button("Download JSON", id="search-btn-json", color="success"),
        ], id='search-download-buttons', style={'margin-top': '20px', 'text-align': 'center', 'display': 'none'}),
        
        # Storage components
        dcc.Store(id='search-results-store'),
        dcc.Store(id='search-current-page-store', data=0),
        
        # Search about modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Using the Search Functions"), 
                           style={"background-color": blue_color, "color": "white"}),
            dbc.ModalBody([
                html.P([
                    "This tab allows you to search through the document database using different search modes:"
                ]),
                
                html.Ul([
                    html.Li([
                        html.Strong("Keyword Search: "), 
                        "Performs exact matching of words. Searching for 'NATO' will only find documents containing that exact word."
                    ]),
                    html.Li([
                        html.Strong("Boolean Search: "), 
                        "Supports logical operators AND, OR, and NOT. For example: 'war AND (Ukraine OR Russia) NOT peace'"
                    ]),
                    html.Li([
                        html.Strong("Semantic Search (Beta): "), 
                        "Searches based on meaning rather than exact keywords. Can find conceptually related content."
                    ])
                ]),
                
                html.P([
                    "After searching, you can explore the results using:"
                ]),
                
                html.Ul([
                    html.Li([
                        html.Strong("Category Distribution: "), 
                        "Shows how search results are distributed across taxonomic categories"
                    ]),
                    html.Li([
                        html.Strong("Timeline: "), 
                        "Displays when the matching documents were published"
                    ]),
                    html.Li([
                        html.Strong("Text Results: "), 
                        "Shows the actual text chunks that match your search criteria"
                    ])
                ]),
                
                html.P([
                    html.Strong("Tip: "),
                    "Use filters to narrow down your search to specific languages, databases, source types, or date ranges."
                ], className="alert alert-info")
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-about-search", className="ms-auto", 
                          style={"background-color": blue_color, "border": "none"})
            ),
        ], id="search-about-modal", size="lg", is_open=False)
    ])


def create_freshness_layout() -> html.Div:
    """
    Create a freshness analysis layout.
    
    Returns:
        html.Div: Freshness layout component
    """
    # Use default layout color
    blue_color = "#13376f"
    
    return html.Div([
        # Header with title and About button
        html.Div([
            html.H3([
                html.Img(src=TAB_LOGOS.get('freshness', ""), className="tab-logo"),
                "Taxonomic Freshness"
            ], style={"display": "inline-block", "margin-right": "20px"}),
            dbc.Button(
                "About", 
                id="open-about-freshness", 
                color="secondary", 
                size="sm",
                className="ml-auto",
                style={"display": "inline-block"}
            ),
        ], style={"display": "flex", "align-items": "center", "margin-bottom": "20px"}),
        
        # Filter Controls
        dbc.Card([
            dbc.CardHeader("Freshness Analysis Controls"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Time Period:"),
                        dcc.RadioItems(
                            id='freshness-period',
                            options=[
                                {'label': 'Last Week', 'value': 'week'},
                                {'label': 'Last Month', 'value': 'month'},
                                {'label': 'Last Quarter', 'value': 'quarter'}
                            ],
                            value='week',
                            inline=True
                        ),
                    ], width=6),
                    dbc.Col([
                        html.Label("Select Filter:"),
                        dcc.Dropdown(
                            id='freshness-filter',
                            options=[
                                {'label': 'All Sources', 'value': 'all'},
                                {'label': 'Russian Sources', 'value': 'russian'},
                                {'label': 'Ukrainian Sources', 'value': 'ukrainian'},
                                {'label': 'Western Sources', 'value': 'western'},
                                {'label': 'Military Publications', 'value': 'military'},
                                {'label': 'Social Media', 'value': 'social_media'}
                            ],
                            value='all'
                        ),
                    ], width=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Br(),
                        dbc.Button('Analyze Freshness', id='freshness-button', color="primary"),
                    ], width=12, style={"text-align": "center"})
                ], className="mt-3")
            ])
        ], className="mb-4"),
        
        # Results section
        html.Div([
            dbc.Tabs([
                dbc.Tab([
                    dcc.Loading(
                        id="loading-freshness-chart",
                        type="default",
                        children=[dcc.Graph(id='freshness-chart', style={'height': '700px'})]
                    )
                ], label="Freshness Overview"),
                
                dbc.Tab([
                    dcc.Loading(
                        id="loading-freshness-timeline",
                        type="default",
                        children=[dcc.Graph(id='freshness-timeline-chart', style={'height': '500px'})]
                    )
                ], label="Freshness Timeline"),
                
                dbc.Tab([
                    html.Div([
                        html.H5("Select Category to Drill Down:", className="mt-3"),
                        dcc.Dropdown(
                            id='freshness-category-dropdown',
                            placeholder='Select a top-level category...'
                        ),
                        dcc.Loading(
                            id="loading-freshness-drilldown",
                            type="default",
                            children=[dcc.Graph(id='freshness-drilldown-chart', style={'height': '600px'})]
                        )
                    ])
                ], label="Category Drill-Down")
            ], id="freshness-tabs")
        ], id="freshness-results", style={"display": "none"}),
        
        # Explanation card
        dbc.Card([
            dbc.CardHeader("Understanding Freshness Analysis"),
            dbc.CardBody([
                html.P([
                    "The freshness analysis helps you identify which taxonomic elements are currently 'fresh' (recently and frequently discussed) ",
                    "and which ones are 'stale' (less discussed or mentioned less recently)."
                ]),
                
                html.H5("Freshness Score Components:"),
                html.Ul([
                    html.Li([
                        html.Strong("Recency (70%): "), 
                        "How recently the taxonomic element was mentioned"
                    ]),
                    html.Li([
                        html.Strong("Frequency (30%): "), 
                        "How frequently the taxonomic element appears in the corpus"
                    ])
                ]),
                
                html.H5("Color Scale:"),
                html.Ul([
                    html.Li([
                        html.Span("Green", style={"color": "#8BC34A", "font-weight": "bold"}), 
                        ": High freshness (recent and frequent mentions)"
                    ]),
                    html.Li([
                        html.Span("Yellow", style={"color": "#FFEB3B", "font-weight": "bold"}), 
                        ": Low freshness (older or less frequent mentions)"
                    ])
                ]),
                
                html.P([
                    "Use this analysis to identify trends and focus areas in the discourse around the Russian-Ukrainian War."
                ], className="mt-2")
            ])
        ], className="mt-4"),
        
        # Freshness about modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("About Freshness Analysis"), 
                           style={"background-color": blue_color, "color": "white"}),
            dbc.ModalBody([
                html.P([
                    "The Freshness Analysis tab helps you identify which topics are currently 'hot' in the discourse ",
                    "around the Russian-Ukrainian War, and which topics are receiving less attention."
                ]),
                
                html.H5("Key Features:"),
                html.Ul([
                    html.Li([
                        html.Strong("Freshness Overview: "), 
                        "Bar chart showing the freshness score of each taxonomic element"
                    ]),
                    html.Li([
                        html.Strong("Freshness Timeline: "), 
                        "Scatter plot showing when each taxonomic element was last mentioned, with size indicating frequency and color indicating freshness"
                    ]),
                    html.Li([
                        html.Strong("Category Drill-Down: "), 
                        "Detailed analysis of freshness for subcategories within a selected category"
                    ])
                ]),
                
                html.H5("How Freshness is Calculated:"),
                html.P([
                    "The freshness score is calculated using a weighted combination of:"
                ]),
                html.Ul([
                    html.Li([
                        html.Strong("Recency (70%): "), 
                        "Based on how recently the taxonomic element appeared in the corpus. Elements mentioned today receive the highest score."
                    ]),
                    html.Li([
                        html.Strong("Frequency (30%): "), 
                        "Based on how frequently the taxonomic element appears. More frequently mentioned elements receive a higher score."
                    ])
                ]),
                
                html.P([
                    "This analysis can help identify emerging topics, track which aspects of the conflict are receiving attention, ",
                    "and spot differences in discourse focus over time."
                ])
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-about-freshness", className="ms-auto", 
                          style={"background-color": blue_color, "border": "none"})
            ),
        ], id="freshness-about-modal", size="lg", is_open=False)
    ], style={'max-width': '1200px', 'margin': 'auto'})


def create_sources_layout(
    db_options: List = None,
    min_date: datetime = None,
    max_date: datetime = None
) -> html.Div:
    """
    Create a sources analysis layout.
    
    Args:
        db_options: Database options for dropdown
        min_date: Minimum date for date picker
        max_date: Maximum date for date picker
        
    Returns:
        html.Div: Sources layout component
    """
    # Use default layout color
    blue_color = "#13376f"
    
    return html.Div([
        # Header with title and explanation
        html.Div([
            html.H3([
                html.Img(src=TAB_LOGOS.get('sources', ""), className="tab-logo"),
                "Corpus Overview"
            ], className="mb-3"),
            dbc.Button(
                "About", 
                id="sources-about-button", 
                color="secondary", 
                size="sm",
                className="ml-2",
                style={"display": "inline-block", "margin-left": "10px"}
            ),
        ], className="d-flex align-items-center"),
        
        # Filter card
        create_filter_card(
            id_prefix="sources",
            db_options=db_options,
            min_date=min_date,
            max_date=max_date
        ),
        
        # Last updated info
        html.Div([
            "Data shown here reflects the latest state of the corpus and is updated regularly. Last updated: ",
            html.Span(datetime.now().strftime("%Y-%m-%d %H:%M"), id="sources-last-updated")
        ], className="text-muted mb-4", style={"border": "1px solid #ddd", "padding": "10px", "background": "#f9f9f9"}),
        
        # Subtabs placeholder - will be populated by callbacks
        dcc.Tabs([], id="sources-subtabs", className="custom-tabs"),
        
        # Sources about modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("About the Sources Tab"), 
                           style={"background-color": blue_color, "color": "white"}),
            dbc.ModalBody([
                html.P([
                    "The Sources tab provides an overview of the corpus used for this dashboard, ",
                    "including statistics about documents, text chunks, and taxonomic elements."
                ]),
                
                html.H5("Key Features:"),
                html.Ul([
                    html.Li([
                        html.Strong("Documents: "), 
                        "Statistics about the source documents, including counts, relevance rates, and distributions by language and database"
                    ]),
                    html.Li([
                        html.Strong("Chunks: "), 
                        "Information about the text chunks extracted from documents, which are the basic units of analysis"
                    ]),
                    html.Li([
                        html.Strong("Taxonomy Combinations: "), 
                        "Analysis of how taxonomic elements are distributed across chunks, including coverage rates and combination patterns"
                    ])
                ]),
                
                html.H5("Understanding the Metrics:"),
                html.Ul([
                    html.Li([
                        html.Strong("Relevance Rate: "), 
                        "Percentage of documents or chunks that have at least one taxonomic element assigned"
                    ]),
                    html.Li([
                        html.Strong("Taxonomy Coverage: "), 
                        "Percentage of chunks that have at least one taxonomic element assigned"
                    ]),
                    html.Li([
                        html.Strong("Avg. Taxonomies per Chunk: "), 
                        "Average number of taxonomic elements assigned to each chunk"
                    ])
                ]),
                
                html.P([
                    "Use the filters at the top of the page to focus on specific subsets of the data, ",
                    "such as documents in a particular language or from specific databases."
                ])
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-about-sources", className="ms-auto", 
                          style={"background-color": blue_color, "border": "none"})
            ),
        ], id="sources-about-modal", size="lg", is_open=False)
    ])


# Additional layout components

def create_visualization_guide_card() -> dbc.Card:
    """
    Create a card with visualization guide information.
    
    Returns:
        dbc.Card: Visualization guide card
    """
    return dbc.Card([
        dbc.CardHeader("Visualization Guide"),
        dbc.CardBody([
            html.P([
                "Each visualization type shows different aspects of the data comparison:"
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H6("Difference in Means", className="mt-1"),
                    html.P([
                        "Shows the percentage point difference between the two slices for each category. ",
                        "Bars extending left indicate higher concentration in Slice A, while bars extending right indicate higher concentration in Slice B."
                    ], style={"font-size": "0.9rem"})
                ], width=6),
                
                dbc.Col([
                    html.H6("Parallel Stacked Bars", className="mt-1"),
                    html.P([
                        "Shows category proportions in each slice side by side, with connecting lines to highlight differences. ",
                        "The size of each segment represents the percentage of that category within its slice."
                    ], style={"font-size": "0.9rem"})
                ], width=6)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H6("Radar Chart", className="mt-3"),
                    html.P([
                        "Displays category distributions in a circular format, making it easy to see where each slice has greater concentrations. ",
                        "Larger areas indicate higher percentages."
                    ], style={"font-size": "0.9rem"})
                ], width=6),
                
                dbc.Col([
                    html.H6("Sankey Diagram", className="mt-3"),
                    html.P([
                        "Shows the flow of data from slices to categories. ",
                        "Wider connections indicate a higher proportion of that category in the slice."
                    ], style={"font-size": "0.9rem"})
                ], width=6)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H6("Heatmap Comparison", className="mt-3"),
                    html.P([
                        "Uses color intensity to show differences between the two slices. ",
                        "Red indicates higher in Slice A, blue indicates higher in Slice B."
                    ], style={"font-size": "0.9rem"})
                ], width=6),
                
                dbc.Col([
                    html.H6("Sunburst Charts", className="mt-3"),
                    html.P([
                        "Displays two separate sunburst charts for side-by-side comparison. ",
                        "Shows all three levels of the taxonomy hierarchy."
                    ], style={"font-size": "0.9rem"})
                ], width=6)
            ]),
            
            html.Div([
                html.Strong("Tip: "),
                "For the Difference in Means and Parallel Bars visualizations, you can click on a category to drill down into its subcategories."
            ], className="alert alert-info mt-3 mb-0", style={"font-size": "0.9rem"})
        ])
    ], className="mb-4 d-none d-lg-block")  # Hidden on small screens


def create_stats_card(title: str, stats_data: Dict) -> dbc.Card:
    """
    Create a card with statistics information.
    
    Args:
        title: Card title
        stats_data: Statistics data dictionary
        
    Returns:
        dbc.Card: Stats card
    """
    return dbc.Card([
        dbc.CardHeader(title),
        dbc.CardBody([
            html.Div([
                html.H1(stats_data.get('total', 0), className="text-center display-4"),
                html.Div(stats_data.get('label', 'Total'), className="text-center text-muted")
            ], className="mb-4"),
            
            html.Hr(),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H5(stats_data.get('primary_stats', {}).get('value', 0), className="mb-0"),
                        html.Div(stats_data.get('primary_stats', {}).get('label', ''))
                    ], className="text-center")
                ], width=6),
                
                dbc.Col([
                    html.Div([
                        html.H5(stats_data.get('secondary_stats', {}).get('value', 0), className="mb-0"),
                        html.Div(stats_data.get('secondary_stats', {}).get('label', ''))
                    ], className="text-center")
                ], width=6)
            ]),
            
            html.Hr(),
            
            html.Div([
                html.H5(stats_data.get('rate', {}).get('value', '0%'), className="text-center text-success"),
                html.Div(stats_data.get('rate', {}).get('label', 'Rate'), className="text-center")
            ])
        ])
    ], className="h-100")