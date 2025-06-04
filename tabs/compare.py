#!/usr/bin/env python
# coding: utf-8

"""
Compare tab layout and callbacks for the dashboard.
This tab allows comparison of two different data slices using various visualization methods.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

import sys
import os

# Add the project root to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.data_fetchers import fetch_category_data
from database.data_fetchers_sources import fetch_keywords_data, fetch_named_entities_data
from components.layout import create_filter_card
from components.cards import create_comparison_card, create_visualization_guide_card
from utils.helpers import create_comparison_text
from visualizations.comparison import create_comparison_plot
from config import COMPARISON_VISUALIZATION_OPTIONS, DEFAULT_START_DATE, DEFAULT_END_DATE


def convert_keywords_to_comparison_format(keywords_data: Dict) -> pd.DataFrame:
    """
    Convert keywords data to comparison format.
    
    Args:
        keywords_data: Keywords data dictionary
        
    Returns:
        pd.DataFrame with category, subcategory, sub_subcategory, and count columns
    """
    logging.info(f"Converting keywords data: {keywords_data.keys() if keywords_data else 'None'}")
    
    if not keywords_data or 'top_keywords' not in keywords_data:
        logging.warning("Keywords data missing or no top_keywords key")
        return pd.DataFrame()
    
    # Check if we have data
    labels = keywords_data['top_keywords'].get('labels', [])
    values = keywords_data['top_keywords'].get('values', [])
    
    logging.info(f"Found {len(labels)} keywords to convert")
    
    if not labels or not values:
        logging.warning("No keyword labels or values found")
        return pd.DataFrame()
    
    # Create DataFrame from top keywords - use keywords as categories for comparison
    # This allows comparing the frequency of specific keywords between datasets
    rows = []
    for i, (keyword, count) in enumerate(zip(labels[:10], values[:10])):
        rows.append({
            'category': keyword,  # Use keyword as category for comparison
            'subcategory': 'Frequency',  # Simple subcategory
            'sub_subcategory': '',
            'count': count
        })
    
    df = pd.DataFrame(rows)
    logging.info(f"Converted keywords to DataFrame with {len(df)} rows")
    return df


def convert_entities_to_comparison_format(entities_data: Dict, entity_type: str = 'ALL') -> pd.DataFrame:
    """
    Convert named entities data to comparison format.
    
    Args:
        entities_data: Named entities data dictionary
        entity_type: Entity type filter ('ALL' or specific type like 'GPE', 'ORG', etc.)
        
    Returns:
        pd.DataFrame with category, subcategory, sub_subcategory, and count columns
    """
    logging.info(f"Converting entities data: {entities_data.keys() if entities_data else 'None'}")
    logging.info(f"Entity type filter: {entity_type}")
    
    if not entities_data:
        logging.warning("No entities data provided")
        return pd.DataFrame()
    
    rows = []
    
    # For comparison, we want to use entities as categories
    if 'top_entities' in entities_data:
        entity_labels = entities_data['top_entities'].get('labels', [])
        entity_types = entities_data['top_entities'].get('types', [])
        entity_values = entities_data['top_entities'].get('values', [])
        
        logging.info(f"Found {len(entity_labels)} top entities")
        
        if entity_type == 'ALL':
            # Show top 10 entities across all types
            for i, (entity, ent_type, count) in enumerate(zip(entity_labels[:10], entity_types[:10], entity_values[:10])):
                rows.append({
                    'category': entity,  # Use entity name as category for comparison
                    'subcategory': ent_type,  # Entity type as subcategory
                    'sub_subcategory': '',
                    'count': count
                })
        else:
            # Filter by specific entity type
            filtered_entities = [(label, ent_type, value) for label, ent_type, value in 
                               zip(entity_labels, entity_types, entity_values) 
                               if ent_type == entity_type][:10]
            
            for entity, ent_type, count in filtered_entities:
                rows.append({
                    'category': entity,  # Use entity name as category
                    'subcategory': ent_type,  # Entity type as subcategory
                    'sub_subcategory': '',
                    'count': count
                })
    
    # If we want to compare entity types instead of individual entities
    elif 'entity_types' in entities_data and entity_type == 'ALL':
        # Use entity types as categories
        entity_type_labels = entities_data['entity_types'].get('labels', [])
        entity_type_counts = entities_data['entity_types'].get('counts', [])
        
        for ent_type, count in zip(entity_type_labels, entity_type_counts):
            rows.append({
                'category': f'{ent_type} Entities',
                'subcategory': 'Count',
                'sub_subcategory': '',
                'count': count
            })
    
    df = pd.DataFrame(rows)
    logging.info(f"Converted entities to DataFrame with {len(df)} rows")
    return df


def convert_keywords_to_comparison_format_unified(keywords_data: Dict, unified_keywords: List[str]) -> pd.DataFrame:
    """
    Convert keywords data to comparison format using a unified set of keywords.
    
    Args:
        keywords_data: Keywords data dictionary
        unified_keywords: List of keywords to include (union from both datasets)
        
    Returns:
        pd.DataFrame with category, subcategory, sub_subcategory, and count columns
    """
    if not keywords_data or 'top_keywords' not in keywords_data:
        # Return empty rows for all unified keywords
        rows = []
        for keyword in unified_keywords:
            rows.append({
                'category': keyword,
                'subcategory': 'Frequency',
                'sub_subcategory': '',
                'count': 0
            })
        return pd.DataFrame(rows)
    
    # Create a mapping of keyword to count
    keyword_counts = dict(zip(
        keywords_data['top_keywords'].get('labels', []),
        keywords_data['top_keywords'].get('values', [])
    ))
    
    # Create rows for all unified keywords
    rows = []
    for keyword in unified_keywords:
        rows.append({
            'category': keyword,
            'subcategory': 'Frequency',
            'sub_subcategory': '',
            'count': keyword_counts.get(keyword, 0)  # Use 0 if keyword not in this dataset
        })
    
    return pd.DataFrame(rows)


def convert_entities_to_comparison_format_unified(entities_data: Dict, unified_entities: List[Tuple[str, str]], entity_type_filter: str = 'ALL') -> pd.DataFrame:
    """
    Convert named entities data to comparison format using a unified set of entities.
    
    Args:
        entities_data: Named entities data dictionary
        unified_entities: List of (entity, type) tuples to include
        entity_type_filter: Entity type filter
        
    Returns:
        pd.DataFrame with category, subcategory, sub_subcategory, and count columns
    """
    if not entities_data or 'top_entities' not in entities_data:
        # Return empty rows for all unified entities
        rows = []
        for entity, ent_type in unified_entities:
            if entity_type_filter == 'ALL' or ent_type == entity_type_filter:
                rows.append({
                    'category': entity,
                    'subcategory': ent_type,
                    'sub_subcategory': '',
                    'count': 0
                })
        return pd.DataFrame(rows)
    
    # Create a mapping of (entity, type) to count
    entity_counts = {}
    labels = entities_data['top_entities'].get('labels', [])
    types = entities_data['top_entities'].get('types', [])
    values = entities_data['top_entities'].get('values', [])
    
    for i in range(len(labels)):
        entity = labels[i]
        ent_type = types[i] if i < len(types) else 'Unknown'
        count = values[i] if i < len(values) else 0
        entity_counts[(entity, ent_type)] = count
    
    # Create rows for all unified entities
    rows = []
    for entity, ent_type in unified_entities:
        if entity_type_filter == 'ALL' or ent_type == entity_type_filter:
            rows.append({
                'category': entity,
                'subcategory': ent_type,
                'sub_subcategory': '',
                'count': entity_counts.get((entity, ent_type), 0)  # Use 0 if not in this dataset
            })
    
    return pd.DataFrame(rows)


def create_compare_tab_layout(db_options: List, min_date: datetime = None, max_date: datetime = None) -> html.Div:
    """
    Create the Compare tab layout.
    
    Args:
        db_options: Database options for filters
        min_date: Minimum date for filters
        max_date: Maximum date for filters
        
    Returns:
        html.Div: Compare tab layout
    """
    # Use default dates if not provided
    if min_date is None:
        min_date = DEFAULT_START_DATE
    if max_date is None:
        max_date = DEFAULT_END_DATE
        
    compare_tab_layout = html.Div([
        # Header with title and About button
        html.Div([
            html.H3("Compare Datasets", style={"display": "inline-block", "margin-right": "20px"}),
            dbc.Button(
                "About", 
                id="open-about-compare", 
                color="secondary", 
                size="sm",
                className="ml-auto",
                style={"display": "inline-block"}
            ),
        ], style={"display": "flex", "align-items": "center", "margin-bottom": "20px"}),
        
        # Two-column layout for filters
        dbc.Row([
            # Filters for Slice A (Russian)
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Russian Data Filters"),
                    dbc.CardBody([
                        html.Label("Language:"),
                        dcc.Dropdown(
                            id='compare-language-dropdown-a',
                            options=[
                                {'label': 'Both English and Russian', 'value': 'ALL'},
                                {'label': 'Russian', 'value': 'RU'},
                                {'label': 'English', 'value': 'EN'}
                            ],
                            placeholder='Select Language',
                            value='RU',  # Changed default to Russian
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
                            options=[
                                {'label': 'All Sources', 'value': 'ALL'},
                                {'label': 'Primary Sources', 'value': 'Primary'},
                                {'label': 'Military Publications', 'value': 'Military'},
                                {'label': 'Scholarly Sources', 'value': 'Scholarly'},
                                {'label': 'Social Media', 'value': 'Social Media'}
                            ],
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
                    dbc.CardHeader("Western Data Filters"),
                    dbc.CardBody([
                        html.Label("Language:"),
                        dcc.Dropdown(
                            id='compare-language-dropdown-b',
                            options=[
                                {'label': 'Both English and Russian', 'value': 'ALL'},
                                {'label': 'Russian', 'value': 'RU'},
                                {'label': 'English', 'value': 'EN'}
                            ],
                            placeholder='Select Language',
                            value='EN',  # Changed default to English
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
                            options=[
                                {'label': 'All Sources', 'value': 'ALL'},
                                {'label': 'Primary Sources', 'value': 'Primary'},
                                {'label': 'Military Publications', 'value': 'Military'},
                                {'label': 'Scholarly Sources', 'value': 'Scholarly'},
                                {'label': 'Social Media', 'value': 'Social Media'}
                            ],
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
        
        # Visualization Controls
        dbc.Card([
            dbc.CardHeader("Visualization Options - Enhanced with Keywords & Entities"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Data Type:"),
                        dcc.Dropdown(
                            id='compare-data-type',
                            options=[
                                {'label': 'Taxonomy Elements', 'value': 'taxonomy'},
                                {'label': 'Keywords', 'value': 'keywords'},
                                {'label': 'Named Entities', 'value': 'named_entities'}
                            ],
                            value='taxonomy',
                            className="mb-2"
                        ),
                    ], width=4),
                    dbc.Col([
                        html.Div([
                            html.Label("Entity Type Filter:", id='entity-type-label', style={'display': 'none'}),
                            dcc.Dropdown(
                                id='compare-entity-type-filter',
                                options=[
                                    {'label': 'All Entity Types', 'value': 'ALL'},
                                    {'label': 'Locations (GPE)', 'value': 'GPE'},
                                    {'label': 'Organizations (ORG)', 'value': 'ORG'},
                                    {'label': 'People (PERSON)', 'value': 'PERSON'},
                                    {'label': 'Nationalities (NORP)', 'value': 'NORP'},
                                    {'label': 'Dates (DATE)', 'value': 'DATE'},
                                    {'label': 'Events (EVENT)', 'value': 'EVENT'}
                                ],
                                value='ALL',
                                className="mb-2",
                                style={'display': 'none'}
                            ),
                        ], id='entity-filter-container'),
                    ], width=4),
                    dbc.Col([
                        html.Label("Visualization Type:"),
                        dcc.Dropdown(
                            id='compare-viz-type',
                            options=COMPARISON_VISUALIZATION_OPTIONS,
                            value='diff_means',  # Changed default to difference in means
                            className="mb-2"
                        ),
                    ], width=4),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Button(
                            'Compare', 
                            id='compare-button', 
                            color="primary", 
                            className="mt-3",
                            style={"width": "100%"}
                        ),
                    ], width=12)
                ])
            ])
        ], className="my-4"),
        
        # Results area - hidden initially
        html.Div([
            dbc.Row([
                dbc.Col([
                    create_visualization_guide_card()
                ], width=12)
            ], className="mb-4"),
            
            # Storage for data
            dcc.Store(id='compare-data-a'),
            dcc.Store(id='compare-data-b'),
            dcc.Store(id='current-taxonomy-level', data='top'),
            dcc.Store(id='current-parent-category'),
            
            # Side-by-side charts
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H5("Russian Data", className="text-center"),
                        dcc.Loading(
                            id="loading-chart-a",
                            type="default",
                            children=[dcc.Graph(id='compare-chart-a')]
                        )
                    ], width=6),
                    dbc.Col([
                        html.H5("Western Data", className="text-center"),
                        dcc.Loading(
                            id="loading-chart-b",
                            type="default",
                            children=[dcc.Graph(id='compare-chart-b')]
                        )
                    ], width=6)
                ])
            ], id='side-by-side-charts', style={'display': 'none'}),
            
            # Full-width chart
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H5("Comparison Visualization", className="text-center"),
                        dcc.Loading(
                            id="loading-chart-full",
                            type="default",
                            children=[dcc.Graph(id='compare-chart-full')]
                        )
                    ], width=12)
                ])
            ], id='full-width-chart', style={'display': 'none'}),
            
            # Text comparison
            dbc.Row([
                dbc.Col([
                    html.Div(id='compare-text-comparison', className="mt-4")
                ], width=12)
            ])
        ], id='comparison-results', style={'display': 'none'}),
        
        # Compare-specific About modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Using the Compare Tab"), 
                           style={"background-color": "#13376f", "color": "white"}),
            dbc.ModalBody([
                html.P([
                    "The Compare tab enables you to analyze and compare data from different sources side by side."
                ]),
                
                html.H5("How to Use This Tab:", style={"margin-top": "20px", "color": "#13376f"}),
                html.Ol([
                    html.Li([
                        html.Strong("Configure Data Slices:"), 
                        " Use the left panel to configure Russian data filters and the right panel for Western data filters."
                    ]),
                    html.Li([
                        html.Strong("Select Visualization:"), 
                        " Choose a visualization type from the dropdown."
                    ]),
                    html.Li([
                        html.Strong("Click Compare:"), 
                        " Generate the comparison visualization."
                    ]),
                    html.Li([
                        html.Strong("Analyze Results:"), 
                        " Examine the differences and patterns between the two data slices."
                    ]),
                ]),
                
                html.H5("Understanding the Visualizations:", style={"margin-top": "20px", "color": "#13376f"}),
                html.Ul([
                    html.Li([
                        html.Strong("Difference in Means:"), 
                        " Shows how the average values differ between the two datasets."
                    ]),
                    html.Li([
                        html.Strong("Parallel Stacked Bars:"), 
                        " Displays the distribution of categories side by side."
                    ]),
                    html.Li([
                        html.Strong("Radar Chart:"), 
                        " Presents a multi-dimensional comparison on a radial axis."
                    ]),
                    html.Li([
                        html.Strong("Sankey Diagram:"), 
                        " Illustrates flows between categories."
                    ]),
                    html.Li([
                        html.Strong("Heatmap Comparison:"), 
                        " Provides a color-coded matrix of relationships."
                    ]),
                    html.Li([
                        html.Strong("Sunburst Charts:"), 
                        " Shows hierarchical data in a concentric circle format."
                    ]),
                ]),
                
                html.H5("Interpretive Tips:", style={"margin-top": "20px", "color": "#13376f"}),
                html.Ul([
                    html.Li([
                        "Look for divergences in how Russian and Western sources cover the same topics."
                    ]),
                    html.Li([
                        "Note changes in emphasis across different time periods."
                    ]),
                    html.Li([
                        "Pay attention to differences in information density on specific topics."
                    ]),
                    html.Li([
                        "Consider both what is highlighted and what is omitted in each data slice."
                    ]),
                ]),
                
                html.P([
                    html.Strong("Tip:"), 
                    " For the richest analysis, try comparing different time periods, languages, and source types."
                ], style={"margin-top": "15px", "font-style": "italic"})
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-compare-about", className="ms-auto", 
                          style={"background-color": "#13376f", "border": "none"})
            ),
        ], id="compare-about-modal", size="lg", is_open=False)
    ], style={'max-width': '1200px', 'margin': 'auto'})
    
    return compare_tab_layout


def register_compare_callbacks(app):
    """
    Register callbacks for the Compare tab.
    
    Args:
        app: Dash application instance
    """
    # Callback to toggle the Compare About modal
    @app.callback(
        Output("compare-about-modal", "is_open"),
        [Input("open-about-compare", "n_clicks"), Input("close-compare-about", "n_clicks")],
        [State("compare-about-modal", "is_open")],
        prevent_initial_call=True
    )
    def toggle_compare_about_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open
    
    # Callback to show/hide entity type filter
    @app.callback(
        [Output('entity-type-label', 'style'),
         Output('compare-entity-type-filter', 'style')],
        Input('compare-data-type', 'value')
    )
    def toggle_entity_filter(data_type):
        if data_type == 'named_entities':
            return {'display': 'block'}, {'display': 'block'}
        return {'display': 'none'}, {'display': 'none'}
        
    # Callback to store data for both slices
    @app.callback(
        [
            Output('compare-data-a', 'data'),
            Output('compare-data-b', 'data'),
            Output('slice-a-stats', 'children'),
            Output('slice-b-stats', 'children')
        ],
        Input('compare-button', 'n_clicks'),
        [
            State('compare-data-type', 'value'),
            State('compare-entity-type-filter', 'value'),
            State('compare-language-dropdown-a', 'value'),
            State('compare-database-dropdown-a', 'value'),
            State('compare-source-type-a', 'value'),
            State('compare-date-picker-a', 'start_date'),
            State('compare-date-picker-a', 'end_date'),
            State('compare-language-dropdown-b', 'value'),
            State('compare-database-dropdown-b', 'value'),
            State('compare-source-type-b', 'value'),
            State('compare-date-picker-b', 'start_date'),
            State('compare-date-picker-b', 'end_date')
        ]
    )
    def store_comparison_data(n_clicks, data_type, entity_type, lang_a, db_a, source_a, start_date_a, end_date_a, 
                              lang_b, db_b, source_b, start_date_b, end_date_b):
        """
        Store comparison data for both slices.
        
        Args:
            n_clicks: Number of button clicks
            data_type: Type of data to compare (taxonomy, keywords, named_entities)
            entity_type: Entity type filter (for named entities)
            lang_a: Language for slice A
            db_a: Database for slice A
            source_a: Source type for slice A
            start_date_a: Start date for slice A
            end_date_a: End date for slice A
            lang_b: Language for slice B
            db_b: Database for slice B
            source_b: Source type for slice B
            start_date_b: Start date for slice B
            end_date_b: End date for slice B
            
        Returns:
            tuple: (df_a, df_b, stats_a, stats_b)
        """
        logging.info(f"Storing comparison data for {data_type}...")
        logging.info(f"Slice A filters: lang={lang_a}, db={db_a}, source={source_a}")
        logging.info(f"Slice B filters: lang={lang_b}, db={db_b}, source={source_b}")
        if not n_clicks:
            return [], [], "", ""
        
        # Prepare date ranges
        date_range_a = None
        if start_date_a and end_date_a:
            date_range_a = (start_date_a, end_date_a)
            
        date_range_b = None
        if start_date_b and end_date_b:
            date_range_b = (start_date_b, end_date_b)
        
        # Fetch data based on data type
        if data_type == 'taxonomy':
            df_a = fetch_category_data(lang_a, db_a, source_a, date_range_a)
            df_b = fetch_category_data(lang_b, db_b, source_b, date_range_b)
        elif data_type == 'keywords':
            # Fetch keywords data
            keywords_a = fetch_keywords_data(lang_a, db_a, source_a, date_range_a)
            keywords_b = fetch_keywords_data(lang_b, db_b, source_b, date_range_b)
            
            logging.info(f"Fetched keywords data - Slice A: {keywords_a.keys() if keywords_a else 'None'}")
            logging.info(f"Fetched keywords data - Slice B: {keywords_b.keys() if keywords_b else 'None'}")
            
            # Get the union of top keywords from both datasets for comparison
            all_keywords = set()
            if keywords_a and 'top_keywords' in keywords_a:
                all_keywords.update(keywords_a['top_keywords']['labels'][:20])
            if keywords_b and 'top_keywords' in keywords_b:
                all_keywords.update(keywords_b['top_keywords']['labels'][:20])
            
            # Convert to comparison format with unified keywords
            df_a = convert_keywords_to_comparison_format_unified(keywords_a, list(all_keywords)[:15])
            df_b = convert_keywords_to_comparison_format_unified(keywords_b, list(all_keywords)[:15])
            
            logging.info(f"Converted keywords - Slice A: {len(df_a)} rows, Slice B: {len(df_b)} rows")
        elif data_type == 'named_entities':
            # Fetch named entities data
            entities_a = fetch_named_entities_data(lang_a, db_a, source_a, date_range_a)
            entities_b = fetch_named_entities_data(lang_b, db_b, source_b, date_range_b)
            
            logging.info(f"Fetched entities data - Slice A: {entities_a.keys() if entities_a else 'None'}")
            logging.info(f"Fetched entities data - Slice B: {entities_b.keys() if entities_b else 'None'}")
            
            # Get the union of top entities from both datasets for comparison
            all_entities = set()
            if entities_a and 'top_entities' in entities_a:
                # Add tuples of (entity, type) to handle entity types
                for i in range(min(20, len(entities_a['top_entities']['labels']))):
                    all_entities.add((entities_a['top_entities']['labels'][i], 
                                    entities_a['top_entities']['types'][i] if i < len(entities_a['top_entities']['types']) else 'Unknown'))
            if entities_b and 'top_entities' in entities_b:
                for i in range(min(20, len(entities_b['top_entities']['labels']))):
                    all_entities.add((entities_b['top_entities']['labels'][i], 
                                    entities_b['top_entities']['types'][i] if i < len(entities_b['top_entities']['types']) else 'Unknown'))
            
            # Convert to comparison format with unified entities
            df_a = convert_entities_to_comparison_format_unified(entities_a, list(all_entities)[:15], entity_type)
            df_b = convert_entities_to_comparison_format_unified(entities_b, list(all_entities)[:15], entity_type)
            
            logging.info(f"Converted entities - Slice A: {len(df_a)} rows, Slice B: {len(df_b)} rows")
        else:
            df_a = pd.DataFrame()
            df_b = pd.DataFrame()
        
        # Create descriptive stats for slices based on data type
        if data_type == 'taxonomy':
            cat_count_a = df_a['category'].nunique() if not df_a.empty else 0
            total_count_a = df_a['count'].sum() if not df_a.empty else 0
            subcat_count_a = df_a['subcategory'].nunique() if not df_a.empty else 0
            
            cat_count_b = df_b['category'].nunique() if not df_b.empty else 0
            total_count_b = df_b['count'].sum() if not df_b.empty else 0
            subcat_count_b = df_b['subcategory'].nunique() if not df_b.empty else 0
            
            type_label = "Taxonomy"
            stats_label_a = f"Categories: {cat_count_a} | Subcategories: {subcat_count_a}"
            stats_label_b = f"Categories: {cat_count_b} | Subcategories: {subcat_count_b}"
        elif data_type == 'keywords':
            # For keywords, count unique keywords
            keyword_count_a = df_a['subcategory'].nunique() if not df_a.empty else 0
            total_count_a = df_a['count'].sum() if not df_a.empty else 0
            
            keyword_count_b = df_b['subcategory'].nunique() if not df_b.empty else 0
            total_count_b = df_b['count'].sum() if not df_b.empty else 0
            
            type_label = "Keywords"
            stats_label_a = f"Unique keywords: {keyword_count_a}"
            stats_label_b = f"Unique keywords: {keyword_count_b}"
        elif data_type == 'named_entities':
            # For entities, show entity types and count
            entity_types_a = df_a['category'].nunique() if not df_a.empty else 0
            entity_count_a = df_a['subcategory'].nunique() if not df_a.empty else 0
            total_count_a = df_a['count'].sum() if not df_a.empty else 0
            
            entity_types_b = df_b['category'].nunique() if not df_b.empty else 0
            entity_count_b = df_b['subcategory'].nunique() if not df_b.empty else 0
            total_count_b = df_b['count'].sum() if not df_b.empty else 0
            
            type_label = "Named Entities" if entity_type == 'ALL' else f"{entity_type} Entities"
            stats_label_a = f"Entity types: {entity_types_a} | Unique entities: {entity_count_a}"
            stats_label_b = f"Entity types: {entity_types_b} | Unique entities: {entity_count_b}"
        else:
            type_label = "Data"
            stats_label_a = "No data"
            stats_label_b = "No data"
            total_count_a = 0
            total_count_b = 0
        
        # Format filter information for display
        def format_filter_info(lang, db, source, date_range):
            lang_display = "All" if lang == "ALL" else lang
            db_display = "All" if db == "ALL" else db
            source_display = "All" if source == "ALL" or not source else source
            date_display = f"{date_range[0]} to {date_range[1]}" if date_range else "All dates"
            return f"Language: {lang_display}, DB: {db_display}, Source: {source_display}, Dates: {date_display}"
        
        filter_info_a = format_filter_info(lang_a, db_a, source_a, date_range_a)
        filter_info_b = format_filter_info(lang_b, db_b, source_b, date_range_b)
        
        stats_a = html.Div([
            html.P(f"{type_label}: {stats_label_a}", className="mb-0"),
            html.P(f"Total occurrences: {total_count_a:,}", className="mb-0"),
            html.P(filter_info_a, className="text-muted small")
        ])
        
        stats_b = html.Div([
            html.P(f"{type_label}: {stats_label_b}", className="mb-0"),
            html.P(f"Total occurrences: {total_count_b:,}", className="mb-0"),
            html.P(filter_info_b, className="text-muted small")
        ])
        
        logging.info(f"Comparison data stored for {data_type}. Slice A: {total_count_a} items. Slice B: {total_count_b} items.")
        
        # Debug: Log the actual data being returned
        if not df_a.empty:
            logging.info(f"DataFrame A sample:\n{df_a.head()}")
        else:
            logging.info("DataFrame A is empty!")
            
        if not df_b.empty:
            logging.info(f"DataFrame B sample:\n{df_b.head()}")
        else:
            logging.info("DataFrame B is empty!")
            
        return (
            df_a.to_dict('records') if not df_a.empty else [],
            df_b.to_dict('records') if not df_b.empty else [],
            stats_a,
            stats_b
        )

    # Create comparison visualizations
    @app.callback(
        [
            Output('comparison-results', 'style', allow_duplicate=True),
            Output('compare-chart-a', 'figure'),
            Output('compare-chart-b', 'figure'),
            Output('compare-chart-full', 'figure'),
            Output('compare-text-comparison', 'children'),
            Output('side-by-side-charts', 'style'),
            Output('full-width-chart', 'style')
        ],
        [
            Input('compare-data-a', 'data'),
            Input('compare-data-b', 'data'),
            Input('compare-viz-type', 'value'),
            Input('compare-chart-full', 'clickData')
        ],
        [
            State('current-taxonomy-level', 'data'),
            State('current-parent-category', 'data')
        ],
        prevent_initial_call=True
    )
    def update_comparison_visualizations(data_a, data_b, viz_type, click_data, current_level, current_parent):
        """
        Update comparison visualizations based on selected visualization type.
        
        Args:
            data_a: Data for slice A
            data_b: Data for slice B
            viz_type: Visualization type
            click_data: Click data from full-width chart
            current_level: Current taxonomy level
            current_parent: Current parent category
            
        Returns:
            tuple: Multiple outputs for updating visualizations
        """
        logging.info(f"Updating comparison visualizations with type: {viz_type}")
        
        # Set default slice names
        slice_a_name = "Russian"
        slice_b_name = "Western"
        
        # Default empty response
        empty_fig = go.Figure().update_layout(title="No data available")
        hidden_style = {'display': 'none'}
        
        # Check context to determine what triggered the callback
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
        
        # Initialize or use current taxonomy tracking
        if current_level is None:
            current_level = 'top'
        if current_parent is None:
            current_parent = None
        
        # Handle drill-down click if that's what triggered the callback
        if trigger_id == 'compare-chart-full' and click_data:
            if viz_type in ['parallel', 'diff_means'] and current_level == 'top':
                # Extract clicked category
                try:
                    clicked_category = click_data['points'][0]['y']
                    logging.info(f"User clicked on category: {clicked_category}")
                    
                    # Update to second level
                    current_level = 'second'
                    current_parent = clicked_category
                    
                    # Filter data for this category
                    df_a = pd.DataFrame(data_a)
                    df_b = pd.DataFrame(data_b)
                    
                    df_a_filtered = df_a[df_a['category'] == clicked_category]
                    df_b_filtered = df_b[df_b['category'] == clicked_category]
                    
                    # Show subcategory comparisons
                    fig_a, fig_b = create_comparison_plot(
                        df_a_filtered, 
                        df_b_filtered, 
                        viz_type, 
                        slice_a_name=slice_a_name, 
                        slice_b_name=slice_b_name
                    )
                    
                    comparison_text = create_comparison_text(
                        df_a_filtered, 
                        df_b_filtered, 
                        viz_type, 
                        slice_a_name=slice_a_name, 
                        slice_b_name=slice_b_name
                    )
                    
                    # Determine which layout to show
                    if viz_type in ['parallel', 'radar', 'sankey', 'diff_means', 'heatmap']:
                        side_by_side_style = {'display': 'none'}
                        full_width_style = {'display': 'block'}
                        
                        return {'display': 'block'}, fig_a, fig_b, fig_a, comparison_text, side_by_side_style, full_width_style
                    else:
                        side_by_side_style = {'display': 'block'}
                        full_width_style = {'display': 'none'}
                        
                        return {'display': 'block'}, fig_a, fig_b, empty_fig, comparison_text, side_by_side_style, full_width_style
                except Exception as e:
                    logging.error(f"Error processing drill-down: {e}")
        
        # Check if both datasets have data for regular visualization
        if not data_a or not data_b:
            return hidden_style, empty_fig, empty_fig, empty_fig, "No data available. Please select filters and click Compare.", hidden_style, hidden_style
        
        # Convert to DataFrames
        df_a = pd.DataFrame(data_a)
        df_b = pd.DataFrame(data_b)
        
        logging.info(f"DataFrame A shape: {df_a.shape}, columns: {df_a.columns.tolist() if not df_a.empty else 'empty'}")
        logging.info(f"DataFrame B shape: {df_b.shape}, columns: {df_b.columns.tolist() if not df_b.empty else 'empty'}")
        if not df_a.empty:
            logging.info(f"DataFrame A first 5 rows:\n{df_a.head()}")
        if not df_b.empty:
            logging.info(f"DataFrame B first 5 rows:\n{df_b.head()}")
        
        # If either is empty, show message
        if df_a.empty or df_b.empty:
            return hidden_style, empty_fig, empty_fig, empty_fig, "One or both datasets are empty. Please refine your filters.", hidden_style, hidden_style
        
        # Reset to top level if this is a new visualization request
        if trigger_id in ['compare-data-a', 'compare-data-b', 'compare-viz-type']:
            current_level = 'top'
            current_parent = None
        
        # Create comparison visualization based on selected type
        logging.info(f"Creating {viz_type} visualization")
        try:
            fig_a, fig_b = create_comparison_plot(df_a, df_b, viz_type, slice_a_name=slice_a_name, slice_b_name=slice_b_name)
            
            comparison_text = create_comparison_text(df_a, df_b, viz_type, slice_a_name=slice_a_name, slice_b_name=slice_b_name)
            
            # Determine which layout to show
            if viz_type in ['parallel', 'radar', 'sankey', 'diff_means', 'heatmap']:
                side_by_side_style = {'display': 'none'}
                full_width_style = {'display': 'block'}
                
                return {'display': 'block'}, fig_a, fig_b, fig_a, comparison_text, side_by_side_style, full_width_style
            else:
                # For sunburst and other viz that need side-by-side
                side_by_side_style = {'display': 'block'}
                full_width_style = {'display': 'none'}
                
                return {'display': 'block'}, fig_a, fig_b, empty_fig, comparison_text, side_by_side_style, full_width_style
        except Exception as e:
            logging.error(f"Error creating visualization: {e}")
            return hidden_style, empty_fig, empty_fig, empty_fig, f"Error creating visualization: {str(e)}", hidden_style, hidden_style
    
    # Callback to show/hide results initially
    @app.callback(
        Output('comparison-results', 'style'),
        Input('compare-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def show_results(n_clicks):
        """Show results area when Compare button is clicked."""
        if n_clicks:
            return {'display': 'block'}
        return {'display': 'none'}
    
    # Callback to update current taxonomy level
    @app.callback(
        Output('current-taxonomy-level', 'data'),
        Output('current-parent-category', 'data'),
        [
            Input('compare-chart-full', 'clickData'),
            Input('compare-button', 'n_clicks')
        ],
        [
            State('current-taxonomy-level', 'data'),
            State('current-parent-category', 'data'),
            State('compare-viz-type', 'value')
        ],
        prevent_initial_call=True
    )
    def update_taxonomy_level(click_data, n_clicks, current_level, current_parent, viz_type):
        """Update the current taxonomy level based on clicks."""
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
        
        if trigger_id == 'compare-button':
            # Reset to top level when Compare button is clicked
            return 'top', None
        
        if trigger_id == 'compare-chart-full' and click_data:
            if viz_type in ['parallel', 'diff_means'] and current_level == 'top':
                try:
                    clicked_category = click_data['points'][0]['y']
                    logging.info(f"Drilling down to {clicked_category}")
                    return 'second', clicked_category
                except Exception as e:
                    logging.error(f"Error processing drill-down click: {e}")
        
        # Default - no change
        return current_level, current_parent