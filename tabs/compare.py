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
    if not keywords_data or 'top_keywords' not in keywords_data:
        return pd.DataFrame()
    
    # Create DataFrame from top keywords
    rows = []
    for i, (keyword, count) in enumerate(zip(keywords_data['top_keywords']['labels'][:50], 
                                           keywords_data['top_keywords']['values'][:50])):
        rows.append({
            'category': 'Keywords',
            'subcategory': keyword,
            'sub_subcategory': '',
            'count': count
        })
    
    return pd.DataFrame(rows)


def convert_entities_to_comparison_format(entities_data: Dict, entity_type: str = 'ALL') -> pd.DataFrame:
    """
    Convert named entities data to comparison format.
    
    Args:
        entities_data: Named entities data dictionary
        entity_type: Entity type filter ('ALL' or specific type like 'GPE', 'ORG', etc.)
        
    Returns:
        pd.DataFrame with category, subcategory, sub_subcategory, and count columns
    """
    if not entities_data or 'top_entities' not in entities_data:
        return pd.DataFrame()
    
    # Create DataFrame from top entities
    rows = []
    
    # If filtering by entity type, use filtered data
    if entity_type != 'ALL' and 'entities_by_type' in entities_data:
        entity_types = entities_data['entities_by_type']
        if entity_type in entity_types:
            type_data = entity_types[entity_type]
            for entity, count in zip(type_data['entities'][:50], type_data['counts'][:50]):
                rows.append({
                    'category': f'{entity_type} Entities',
                    'subcategory': entity,
                    'sub_subcategory': '',
                    'count': count
                })
    else:
        # Use all entities grouped by type
        if 'entities_by_type' in entities_data:
            for ent_type, type_data in entities_data['entities_by_type'].items():
                for entity, count in zip(type_data['entities'][:10], type_data['counts'][:10]):
                    rows.append({
                        'category': f'{ent_type} Entities',
                        'subcategory': entity,
                        'sub_subcategory': '',
                        'count': count
                    })
        else:
            # Fallback to top entities without type grouping
            for i, (entity, count) in enumerate(zip(entities_data['top_entities']['labels'][:50], 
                                                   entities_data['top_entities']['values'][:50])):
                rows.append({
                    'category': 'Named Entities',
                    'subcategory': entity,
                    'sub_subcategory': '',
                    'count': count
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
            dbc.CardHeader("Visualization Options"),
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
            # Convert to comparison format
            df_a = convert_keywords_to_comparison_format(keywords_a)
            df_b = convert_keywords_to_comparison_format(keywords_b)
        elif data_type == 'named_entities':
            # Fetch named entities data
            entities_a = fetch_named_entities_data(lang_a, db_a, source_a, date_range_a)
            entities_b = fetch_named_entities_data(lang_b, db_b, source_b, date_range_b)
            # Convert to comparison format with optional entity type filter
            df_a = convert_entities_to_comparison_format(entities_a, entity_type)
            df_b = convert_entities_to_comparison_format(entities_b, entity_type)
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