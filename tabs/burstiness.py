#!/usr/bin/env python
# coding: utf-8

"""
Burstiness tab layout and callbacks for the dashboard.
This tab provides analysis of bursts in taxonomic elements, keywords, and named entities,
using Kleinberg's burst detection algorithm to identify significant spikes in frequency,
visualized in a CiteSpace-inspired style.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import numpy as np
import dash
from dash import html, dcc, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

import sys
import os

# Add the project root to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.data_fetchers_freshness import get_burst_data_for_periods, calculate_burst_summaries
from database.data_fetchers import fetch_all_databases, fetch_date_range
from components.layout import create_filter_card
from config import (
    LANGUAGE_OPTIONS,
    SOURCE_TYPE_OPTIONS,
    FRESHNESS_FILTER_OPTIONS as BURSTINESS_FILTER_OPTIONS
)
from visualizations.bursts import (
    create_burst_heatmap,
    create_burst_summary_chart,
    create_burst_timeline,
    create_burst_comparison_chart,
    create_citespace_timeline
)

# Theme colors for consistency
THEME_BLUE = "#13376f"  # Main dashboard theme color

# Named entity types for filtering
NAMED_ENTITY_TYPES = [
    {'label': 'Locations (GPE)', 'value': 'GPE'},
    {'label': 'Organizations (ORG)', 'value': 'ORG'},
    {'label': 'People (PERSON)', 'value': 'PERSON'},
    {'label': 'Nationalities (NORP)', 'value': 'NORP'},
    {'label': 'Dates (DATE)', 'value': 'DATE'},
    {'label': 'Events (EVENT)', 'value': 'EVENT'}
]

# Time period options
TIME_PERIOD_OPTIONS = [
    {'label': 'Last 10 Weeks', 'value': 'week'},
    {'label': 'Last 10 Months', 'value': 'month'},
    {'label': 'Last 10 Quarters', 'value': 'quarter'}
]

# Taxonomy level options
TAXONOMY_LEVEL_OPTIONS = [
    {'label': 'Main Categories', 'value': 'category'},
    {'label': 'Subcategories', 'value': 'subcategory'},
    {'label': 'Sub-subcategories', 'value': 'sub_subcategory'}
]


def create_burstiness_tab_layout():
    """
    Create the Burstiness tab layout.
    
    Returns:
        html.Div: Burstiness tab layout
    """
    # Get database options for filters
    db_options = [{'label': 'All Databases', 'value': 'ALL'}]
    try:
        databases = fetch_all_databases()
        db_options.extend([{'label': db, 'value': db} for db in databases])
    except Exception as e:
        logging.error(f"Error fetching database options: {e}")
    
    # Get default date range for filters
    min_date, max_date = fetch_date_range()
    
    burstiness_tab_layout = html.Div([
        # Header with title and About button
        html.Div([
            html.H3("Burstiness Analysis", style={"display": "inline-block", "margin-right": "20px"}),
            dbc.Button(
                "About", 
                id="open-about-burstiness", 
                color="secondary", 
                size="sm",
                className="ml-auto",
                style={"display": "inline-block"}
            ),
        ], style={"display": "flex", "align-items": "center", "margin-bottom": "20px"}),
        
        # Filter Controls
        dbc.Card([
            dbc.CardHeader("Burstiness Analysis Controls", style={"background-color": THEME_BLUE, "color": "white"}),
            dbc.CardBody([
                dbc.Row([
                    # Time period selection
                    dbc.Col([
                        html.Label("Time Period:"),
                        dcc.RadioItems(
                            id='burstiness-period',
                            options=TIME_PERIOD_OPTIONS,
                            value='month',
                            inline=True,
                            className="mb-3"
                        ),
                    ], width=12),
                ]),
                
                dbc.Row([
                    # Standard filters (collapsible)
                    dbc.Col([
                        dbc.Button(
                            "Standard Filters", 
                            id="burstiness-standard-filters-toggle",
                            className="mb-3",
                            color="secondary"
                        ),
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Language:"),
                                            dcc.Dropdown(
                                                id='burstiness-language-dropdown',
                                                options=LANGUAGE_OPTIONS,
                                                value='ALL',
                                                clearable=False,
                                            ),
                                        ], width=4),
                                        dbc.Col([
                                            html.Label("Database:"),
                                            dcc.Dropdown(
                                                id='burstiness-database-dropdown',
                                                options=db_options,
                                                value='ALL',
                                                clearable=False,
                                            )
                                        ], width=4),
                                        dbc.Col([
                                            html.Label("Source Type:"),
                                            dcc.Dropdown(
                                                id='burstiness-source-type-dropdown',
                                                options=SOURCE_TYPE_OPTIONS,
                                                value='ALL',
                                                clearable=False,
                                            )
                                        ], width=4),
                                    ]),
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Date Range:"),
                                            dcc.DatePickerRange(
                                                id='burstiness-date-range-picker',
                                                min_date_allowed=min_date if min_date else datetime(2022, 1, 1),
                                                max_date_allowed=max_date if max_date else datetime.now(),
                                                initial_visible_month=datetime.now() - timedelta(days=30),
                                                clearable=True,
                                                with_portal=True,
                                            ),
                                        ], width=12),
                                    ], className="mt-3"),
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Source Filter:"),
                                            dcc.Dropdown(
                                                id='burstiness-filter',
                                                options=BURSTINESS_FILTER_OPTIONS,
                                                value='all',
                                                clearable=False,
                                            ),
                                        ], width=12),
                                    ], className="mt-3"),
                                ])
                            ),
                            id="burstiness-standard-filters-collapse",
                            is_open=False,
                        ),
                    ], width=12),
                ]),
                
                dbc.Row([
                    # Data type filters (collapsible)
                    dbc.Col([
                        dbc.Button(
                            "Data Type Filters", 
                            id="burstiness-datatype-filters-toggle",
                            className="mb-3",
                            color="primary"
                        ),
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody([
                                    dbc.Row([
                                        # Taxonomy options
                                        dbc.Col([
                                            html.Label("Taxonomy Elements:"),
                                            dbc.Checklist(
                                                id="burstiness-include-taxonomy",
                                                options=[{'label': 'Include Taxonomy Elements', 'value': 'taxonomy'}],
                                                value=['taxonomy'],
                                                switch=True,
                                            ),
                                            html.Div([
                                                html.Label("Taxonomy Level:"),
                                                dcc.RadioItems(
                                                    id='burstiness-taxonomy-level',
                                                    options=TAXONOMY_LEVEL_OPTIONS,
                                                    value='category',
                                                    className="ml-3"
                                                )
                                            ], id="burstiness-taxonomy-level-container"),
                                        ], width=4),
                                        
                                        # Keywords options
                                        dbc.Col([
                                            html.Label("Keywords:"),
                                            dbc.Checklist(
                                                id="burstiness-include-keywords",
                                                options=[{'label': 'Include Keywords', 'value': 'keywords'}],
                                                value=['keywords'],
                                                switch=True,
                                            ),
                                            html.Div([
                                                html.Label("Number of Top Keywords:"),
                                                dcc.Slider(
                                                    id='burstiness-keywords-top-n',
                                                    min=5,
                                                    max=50,
                                                    step=5,
                                                    value=20,
                                                    marks={i: str(i) for i in range(5, 51, 5)},
                                                    className="ml-3"
                                                )
                                            ], id="burstiness-keywords-options-container"),
                                        ], width=4),
                                        
                                        # Named entities options
                                        dbc.Col([
                                            html.Label("Named Entities:"),
                                            dbc.Checklist(
                                                id="burstiness-include-entities",
                                                options=[{'label': 'Include Named Entities', 'value': 'named_entities'}],
                                                value=['named_entities'],
                                                switch=True,
                                            ),
                                            html.Div([
                                                html.Label("Entity Types:"),
                                                dbc.Checklist(
                                                    id="burstiness-entity-types",
                                                    options=NAMED_ENTITY_TYPES,
                                                    value=['GPE', 'ORG', 'PERSON', 'NORP'],
                                                    className="ml-3"
                                                ),
                                                html.Label("Number of Top Entities:"),
                                                dcc.Slider(
                                                    id='burstiness-entities-top-n',
                                                    min=5,
                                                    max=50,
                                                    step=5,
                                                    value=20,
                                                    marks={i: str(i) for i in range(5, 51, 5)},
                                                    className="ml-3 mt-2"
                                                )
                                            ], id="burstiness-entities-options-container"),
                                        ], width=4),
                                    ]),
                                ])
                            ),
                            id="burstiness-datatype-filters-collapse",
                            is_open=True,
                        ),
                    ], width=12),
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.Br(),
                        dbc.Button('Run Burstiness Analysis', id='burstiness-button', 
                                  color="danger", size="lg",
                                  style={"background-color": THEME_BLUE, "border": "none"}),
                    ], width=12, style={"text-align": "center"})
                ], className="mt-3")
            ])
        ], className="mb-4"),
        
        # Loading spinner for the entire results section
        dcc.Loading(
            id="loading-burstiness-results",
            type="circle",
            color=THEME_BLUE,
            children=[
                # Results section (initially hidden)
                html.Div([
                    dbc.Tabs([
                        # Overview tab with comparison across data types
                        dbc.Tab([
                            html.Div([
                                dbc.Row([
                                    dbc.Col([
                                        html.H4("Top Bursting Elements Comparison", className="mt-3"),
                                        html.P("This chart shows the elements with the highest burst intensity for each data type."),
                                        dcc.Loading(
                                            id="loading-comparison-chart",
                                            type="circle",
                                            color=THEME_BLUE,
                                            children=[dcc.Graph(id='burstiness-comparison-chart', style={'height': '500px'})]
                                        ),
                                    ], width=12),
                                ]),
                                
                                dbc.Row([
                                    dbc.Col([
                                        html.H4("CiteSpace-Style Burst Timeline", className="mt-3"),
                                        html.P("This visualization shows bursts as horizontal segments with varying thickness based on intensity, similar to CiteSpace."),
                                        dcc.Loading(
                                            id="loading-citespace-timeline",
                                            type="circle",
                                            color=THEME_BLUE,
                                            children=[dcc.Graph(id='burstiness-citespace-timeline', style={'height': '600px'})]
                                        )
                                    ], width=12),
                                ]),
                                
                                dbc.Row([
                                    dbc.Col([
                                        html.H4("Top Elements Burst Timeline", className="mt-3"),
                                        html.P("This timeline shows how the top elements from each data type change in burst intensity over time."),
                                        dcc.Loading(
                                            id="loading-overview-timeline",
                                            type="circle",
                                            color=THEME_BLUE,
                                            children=[dcc.Graph(id='burstiness-overview-timeline', style={'height': '500px'})]
                                        )
                                    ], width=12),
                                ]),
                            ], className="p-4")
                        ], label="Overview", tab_id="overview"),
                        
                        # Taxonomy tab
                        dbc.Tab([
                            html.Div([
                                dbc.Row([
                                    dbc.Col([
                                        html.H4("Taxonomy Elements Burst Analysis", className="mt-3"),
                                        html.P("This chart shows the taxonomy elements with the highest burst intensity."),
                                        dcc.Loading(
                                            id="loading-taxonomy-chart",
                                            type="circle",
                                            color=THEME_BLUE,
                                            children=[dcc.Graph(id='taxonomy-burst-chart', style={'height': '600px'})]
                                        )
                                    ], width=12),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        html.H4("Taxonomy Elements Burst Timeline", className="mt-3"),
                                        html.P("This timeline shows how the burst intensity of top taxonomy elements changes over time."),
                                        dcc.Loading(
                                            id="loading-taxonomy-timeline",
                                            type="circle",
                                            color=THEME_BLUE,
                                            children=[dcc.Graph(id='taxonomy-burst-timeline', style={'height': '500px'})]
                                        )
                                    ], width=12),
                                ]),
                            ], className="p-4")
                        ], label="Taxonomy Elements", tab_id="taxonomy"),
                        
                        # Keywords tab
                        dbc.Tab([
                            html.Div([
                                dbc.Row([
                                    dbc.Col([
                                        html.H4("Keywords Burst Analysis", className="mt-3"), 
                                        html.P("This chart shows the keywords with the highest burst intensity."),
                                        dcc.Loading(
                                            id="loading-keyword-chart",
                                            type="circle",
                                            color=THEME_BLUE,
                                            children=[dcc.Graph(id='keyword-burst-chart', style={'height': '600px'})]
                                        )
                                    ], width=12),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        html.H4("Keywords Burst Timeline", className="mt-3"),
                                        html.P("This timeline shows how the burst intensity of top keywords changes over time."),
                                        dcc.Loading(
                                            id="loading-keyword-timeline",
                                            type="circle",
                                            color=THEME_BLUE,
                                            children=[dcc.Graph(id='keyword-burst-timeline', style={'height': '500px'})]
                                        )
                                    ], width=12),
                                ]),
                            ], className="p-4")
                        ], label="Keywords", tab_id="keywords"),
                        
                        # Named Entities tab
                        dbc.Tab([
                            html.Div([
                                dbc.Row([
                                    dbc.Col([
                                        html.H4("Named Entities Burst Analysis", className="mt-3"),
                                        html.P("This chart shows the named entities with the highest burst intensity."),
                                        dcc.Loading(
                                            id="loading-entity-chart",
                                            type="circle",
                                            color=THEME_BLUE,
                                            children=[dcc.Graph(id='entity-burst-chart', style={'height': '600px'})]
                                        )
                                    ], width=12),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        html.H4("Named Entities Burst Timeline", className="mt-3"),
                                        html.P("This timeline shows how the burst intensity of top named entities changes over time."),
                                        dcc.Loading(
                                            id="loading-entity-timeline",
                                            type="circle",
                                            color=THEME_BLUE,
                                            children=[dcc.Graph(id='entity-burst-timeline', style={'height': '500px'})]
                                        )
                                    ], width=12),
                                ]),
                            ], className="p-4")
                        ], label="Named Entities", tab_id="entities")
                    ], id="burstiness-tabs")
                ], id="burstiness-results", style={"display": "none"})
            ]
        ),
        
        # About modal for Burstiness
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Using the Burstiness Tab"), 
                           style={"background-color": THEME_BLUE, "color": "white"}),
            dbc.ModalBody([
                html.P([
                    "The Burstiness tab helps you identify which topics, keywords, or entities are experiencing significant spikes in frequency over time. ",
                    "This is useful for identifying emerging trends, sudden events, or shifts in focus within the Russian-Ukrainian War discourse."
                ]),
                
                html.H5("About Burst Detection:", style={"margin-top": "20px", "color": THEME_BLUE}),
                html.P([
                    "Burst detection identifies significant spikes in the frequency of taxonomic elements, keywords, or named entities over time. ",
                    "Using Kleinberg's burst detection algorithm, we can identify when certain topics or entities suddenly gain prominence in the discourse."
                ]),
                html.P([
                    "A high burst intensity (score) indicates that an element has seen a significant increase in mentions during a specific time period, ",
                    "compared to its typical baseline frequency. This helps identify emerging or trending topics."
                ]),
                html.P([
                    "The analysis is performed across the selected time periods (weeks, months, or quarters), allowing you to see both ",
                    "short-term spikes and longer-term trends in the data."
                ]),
                
                html.H5("How to Use This Tab:", style={"margin-top": "20px", "color": THEME_BLUE}),
                html.Ol([
                    html.Li([
                        html.Strong("Select Time Period:"), 
                        " Choose whether to analyze burstiness over the last 10 weeks, months, or quarters."
                    ]),
                    html.Li([
                        html.Strong("Configure Filters:"), 
                        " Use the Standard Filters and Data Type Filters to refine your analysis."
                    ]),
                    html.Li([
                        html.Strong("Run Analysis:"), 
                        " Click the 'Run Burstiness Analysis' button to generate visualizations."
                    ]),
                    html.Li([
                        html.Strong("Explore Visualizations:"), 
                        " Switch between the different tabs to examine bursts from various perspectives."
                    ]),
                ]),
                
                html.H5("Understanding the Visualizations:", style={"margin-top": "20px", "color": THEME_BLUE}),
                html.Ul([
                    html.Li([
                        html.Strong("CiteSpace-Style Timeline:"), 
                        " Shows bursts as horizontal bars where the thickness indicates intensity, similar to CiteSpace's citation burst diagrams."
                    ]),
                    html.Li([
                        html.Strong("Comparison Chart:"), 
                        " Bar chart comparing the highest bursting elements across different data types."
                    ]),
                    html.Li([
                        html.Strong("Burst Timeline:"), 
                        " Line chart showing how burst intensity changes over time for top elements."
                    ]),
                    html.Li([
                        html.Strong("Data Type Tabs:"), 
                        " Dedicated visualizations for taxonomy elements, keywords, and named entities."
                    ]),
                ]),
                
                html.H5("Interpreting the Results:", style={"margin-top": "20px", "color": THEME_BLUE}),
                html.Ul([
                    html.Li([
                        html.Strong("High Burst Intensity:"), 
                        " Indicates a significant spike in mentions compared to baseline."
                    ]),
                    html.Li([
                        html.Strong("Sustained Bursts:"), 
                        " When an element maintains high burst intensity across multiple periods, it suggests a sustained focus or ongoing event."
                    ]),
                    html.Li([
                        html.Strong("Patterns Across Data Types:"), 
                        " Look for related bursts across taxonomy elements, keywords, and named entities to identify comprehensive trends."
                    ]),
                ]),
                
                html.P([
                    html.Strong("Tip:"), 
                    " Compare bursts across different time periods (weeks vs. months vs. quarters) to distinguish between short-term events and longer-term trends."
                ], style={"margin-top": "15px", "font-style": "italic"})
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-burstiness-about", className="ms-auto", 
                          style={"background-color": THEME_BLUE, "border": "none"})
            ),
        ], id="burstiness-about-modal", size="lg", is_open=False)
    ], style={'max-width': '1200px', 'margin': 'auto'})
    
    return burstiness_tab_layout


def register_burstiness_callbacks(app):
    """
    Register callbacks for the Burstiness tab.
    
    Args:
        app: Dash application instance
    """
    # Callback for About modal
    @app.callback(
        Output("burstiness-about-modal", "is_open"),
        [Input("open-about-burstiness", "n_clicks"), Input("close-burstiness-about", "n_clicks")],
        [State("burstiness-about-modal", "is_open")],
        prevent_initial_call=True
    )
    def toggle_burstiness_about_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open
    
    # Callback for standard filters collapse
    @app.callback(
        Output("burstiness-standard-filters-collapse", "is_open"),
        [Input("burstiness-standard-filters-toggle", "n_clicks")],
        [State("burstiness-standard-filters-collapse", "is_open")],
        prevent_initial_call=True
    )
    def toggle_standard_filters_collapse(n, is_open):
        if n:
            return not is_open
        return is_open
    
    # Callback for data type filters collapse
    @app.callback(
        Output("burstiness-datatype-filters-collapse", "is_open"),
        [Input("burstiness-datatype-filters-toggle", "n_clicks")],
        [State("burstiness-datatype-filters-collapse", "is_open")],
        prevent_initial_call=True
    )
    def toggle_datatype_filters_collapse(n, is_open):
        if n:
            return not is_open
        return is_open
    
    # Callbacks to toggle visibility of data type-specific options
    @app.callback(
        Output("burstiness-taxonomy-level-container", "style"),
        [Input("burstiness-include-taxonomy", "value")]
    )
    def toggle_taxonomy_options(value):
        if value and 'taxonomy' in value:
            return {"display": "block"}
        return {"display": "none"}
    
    @app.callback(
        Output("burstiness-keywords-options-container", "style"),
        [Input("burstiness-include-keywords", "value")]
    )
    def toggle_keywords_options(value):
        if value and 'keywords' in value:
            return {"display": "block"}
        return {"display": "none"}
    
    @app.callback(
        Output("burstiness-entities-options-container", "style"),
        [Input("burstiness-include-entities", "value")]
    )
    def toggle_entities_options(value):
        if value and 'named_entities' in value:
            return {"display": "block"}
        return {"display": "none"}
    
    # Callback to show/hide tabs based on data type selection
    @app.callback(
        [
            Output("taxonomy", "disabled"),
            Output("keywords", "disabled"),
            Output("entities", "disabled")
        ],
        [
            Input("burstiness-include-taxonomy", "value"),
            Input("burstiness-include-keywords", "value"),
            Input("burstiness-include-entities", "value")
        ]
    )
    def toggle_tabs_enabled(taxonomy_val, keywords_val, entities_val):
        taxonomy_disabled = 'taxonomy' not in (taxonomy_val or [])
        keywords_disabled = 'keywords' not in (keywords_val or [])
        entities_disabled = 'named_entities' not in (entities_val or [])
        return taxonomy_disabled, keywords_disabled, entities_disabled
    
    # Main callback for burstiness analysis
    @app.callback(
        [
            # Show/hide results
            Output('burstiness-results', 'style'),
            
            # Update visualizations
            Output('burstiness-comparison-chart', 'figure'),
            Output('burstiness-citespace-timeline', 'figure'),  # New CiteSpace timeline
            Output('burstiness-overview-timeline', 'figure'),
            Output('taxonomy-burst-chart', 'figure'),
            Output('taxonomy-burst-timeline', 'figure'),
            Output('keyword-burst-chart', 'figure'),
            Output('keyword-burst-timeline', 'figure'),
            Output('entity-burst-chart', 'figure'),
            Output('entity-burst-timeline', 'figure')
        ],
        [Input('burstiness-button', 'n_clicks')],
        [
            # Time period
            State('burstiness-period', 'value'),
            
            # Standard filters
            State('burstiness-language-dropdown', 'value'),
            State('burstiness-database-dropdown', 'value'),
            State('burstiness-source-type-dropdown', 'value'),
            State('burstiness-date-range-picker', 'start_date'),
            State('burstiness-date-range-picker', 'end_date'),
            State('burstiness-filter', 'value'),
            
            # Data type selections
            State('burstiness-include-taxonomy', 'value'),
            State('burstiness-taxonomy-level', 'value'),
            State('burstiness-include-keywords', 'value'),
            State('burstiness-keywords-top-n', 'value'),
            State('burstiness-include-entities', 'value'),
            State('burstiness-entity-types', 'value'),
            State('burstiness-entities-top-n', 'value')
        ]
    )
    def update_burstiness_analysis(
        n_clicks, 
        period,
        language, database, source_type, start_date, end_date, filter_value,
        include_taxonomy, taxonomy_level, include_keywords, keywords_top_n,
        include_entities, entity_types, entities_top_n
    ):
        """
        Update burstiness analysis visualizations based on user selections.
        
        Args:
            n_clicks: Button click trigger
            period: Selected time period (week, month, quarter)
            language, database, source_type: Standard filters
            start_date, end_date: Date range picker values
            filter_value: Additional filter from BURSTINESS_FILTER_OPTIONS
            include_taxonomy: Whether to include taxonomy elements
            taxonomy_level: Level of taxonomy to analyze (category, subcategory, sub_subcategory)
            include_keywords: Whether to include keywords
            keywords_top_n: Number of top keywords to include
            include_entities: Whether to include named entities
            entity_types: Types of named entities to include
            entities_top_n: Number of top entities to include
            
        Returns:
            tuple: Updated visualization figures
        """
        if not n_clicks:
            # Initial state - return empty visualizations
            empty_fig = go.Figure().update_layout(title="No data to display yet")
            return {'display': 'none'}, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
        
        logging.info(f"Running burstiness analysis with period={period}, filter={filter_value}")
        
        # Determine which data types to include
        data_types = []
        if include_taxonomy and 'taxonomy' in include_taxonomy:
            data_types.append('taxonomy')
        if include_keywords and 'keywords' in include_keywords:
            data_types.append('keywords')
        if include_entities and 'named_entities' in include_entities:
            data_types.append('named_entities')
        
        # If no data types selected, show message
        if not data_types:
            empty_fig = go.Figure().update_layout(
                title="No data types selected. Please select at least one data type.",
                annotations=[dict(
                    text="Please select at least one data type from the Data Type Filters section.",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    font=dict(size=16)
                )]
            )
            return {'display': 'block'}, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
        
        # Build filter settings
        filter_settings = {
            'language': language if language != 'ALL' else None,
            'database': database if database != 'ALL' else None,
            'source_type': source_type if source_type != 'ALL' else None,
            'date_range': (start_date, end_date) if start_date and end_date else None,
            'taxonomy_level': taxonomy_level or 'category',
            'entity_types': entity_types or ['GPE', 'ORG', 'PERSON', 'NORP'],
            'keywords_top_n': keywords_top_n or 20,
            'entities_top_n': entities_top_n or 20
        }
        
        # Get burst data for the selected period, filter, and data types
        burst_data = get_burst_data_for_periods(
            period_type=period,
            n_periods=10,  # Always use 10 periods as requested
            filter_value=filter_value,
            data_types=data_types,
            **filter_settings
        )
        
        # Calculate summaries for the burst data
        burst_summaries = calculate_burst_summaries(burst_data)
        
        # Create comparison chart for overview tab
        taxonomy_summary = burst_summaries.get('taxonomy', pd.DataFrame())
        keyword_summary = burst_summaries.get('keywords', pd.DataFrame())
        entity_summary = burst_summaries.get('named_entities', pd.DataFrame())
        
        comparison_fig = create_burst_comparison_chart(
            taxonomy_summary,
            keyword_summary,
            entity_summary,
            title=f"Top Bursting Elements (Last 10 {period}s)"
        )
        
        # Create CiteSpace-style timeline visualization
        citespace_timeline_data = []
        # Combine data from all types for the CiteSpace visualization
        for data_type, elements in burst_data.items():
            prefix = ""
            if data_type == 'taxonomy':
                prefix = "T: "
            elif data_type == 'keywords':
                prefix = "K: "
            else:
                prefix = "E: "
                
            for element, df in elements.items():
                if not df.empty and 'period' in df.columns and 'burst_intensity' in df.columns:
                    for _, row in df.iterrows():
                        citespace_timeline_data.append({
                            'element': prefix + element,
                            'period': row['period'],
                            'burst_intensity': row['burst_intensity']
                        })
        
        # Create DataFrame for CiteSpace timeline
        if citespace_timeline_data:
            citespace_df = pd.DataFrame(citespace_timeline_data)
            citespace_timeline_fig = create_citespace_timeline(
                citespace_df,
                title=f"CiteSpace-Style Burst Timeline (Last 10 {period}s)",
                color_scale=px.colors.sequential.Reds
            )
        else:
            citespace_timeline_fig = go.Figure().update_layout(
                title="No data available for CiteSpace timeline"
            )
        
        # Create overview timeline
        # Combine top elements from all data types
        all_elements = []
        for data_type, elements in burst_data.items():
            if data_type == 'taxonomy':
                prefix = 'T: '
            elif data_type == 'keywords':
                prefix = 'K: '
            else:
                prefix = 'E: '
            
            # Sort elements by max burst intensity and add top 3
            top_elements = sorted(
                [(elem, df['burst_intensity'].max()) for elem, df in elements.items()],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            for elem_name, _ in top_elements:
                prefixed_name = prefix + elem_name
                if elements[elem_name].empty:
                    continue
                
                # Add period to element data for overview timeline
                elements[elem_name]['element'] = prefixed_name
                all_elements.append(elements[elem_name])
        
        # Create combined timeline figure
        if all_elements:
            combined_df = pd.concat(all_elements)
            overview_timeline_fig = go.Figure()
            
            # Group by element and period
            for element in combined_df['element'].unique():
                element_df = combined_df[combined_df['element'] == element]
                
                # Group by period to get average intensity
                period_data = element_df.groupby('period').agg({
                    'burst_intensity': 'max'
                }).reset_index()
                
                # Determine color based on prefix
                if element.startswith('T:'):
                    color = '#4caf50'  # Green for taxonomy
                elif element.startswith('K:'):
                    color = '#2196f3'  # Blue for keywords
                else:
                    color = '#ff9800'  # Orange for entities
                
                # Add line to figure
                overview_timeline_fig.add_trace(go.Scatter(
                    x=period_data['period'],
                    y=period_data['burst_intensity'],
                    mode='lines+markers',
                    name=element,
                    line=dict(width=2, color=color),
                    marker=dict(size=8, color=color),
                    hovertemplate='<b>%{x}</b><br>Element: %{fullData.name}<br>Burst Intensity: %{y:.1f}<extra></extra>'
                ))
            
            overview_timeline_fig.update_layout(
                title=f"Top Elements Burst Timeline (Last 10 {period}s)",
                xaxis_title='Time Period',
                yaxis_title='Burst Intensity',
                height=500,
                margin=dict(l=20, r=20, t=40, b=50),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                ),
                hovermode="closest",
                plot_bgcolor='rgb(248, 248, 248)',
                paper_bgcolor='white',
            )
        else:
            overview_timeline_fig = go.Figure().update_layout(
                title="No timeline data available"
            )
        
        # Create taxonomy burst charts
        if 'taxonomy' in data_types and 'taxonomy' in burst_summaries and not burst_summaries['taxonomy'].empty:
            # Create a full DataFrame with element, period, burst_intensity for heatmap
            taxonomy_heatmap_data = []
            for element, df in burst_data['taxonomy'].items():
                if not df.empty and 'period' in df.columns and 'burst_intensity' in df.columns:
                    # Get max burst intensity for each period
                    period_data = df.groupby('period')['burst_intensity'].max().reset_index()
                    for _, row in period_data.iterrows():
                        taxonomy_heatmap_data.append({
                            'element': element,
                            'period': row['period'],
                            'burst_intensity': row['burst_intensity']
                        })
            
            taxonomy_heatmap_df = pd.DataFrame(taxonomy_heatmap_data)
            taxonomy_burst_fig = create_burst_heatmap(
                taxonomy_heatmap_df,
                title=f"Taxonomy {taxonomy_level.title()} Burst Intensity by Period",
                color_scale=[[0, '#f7f7f7'], [0.4, '#c8e6c9'], [0.7, '#66bb6a'], [1, '#2e7d32']]  # Green scale
            )
            
            taxonomy_timeline_fig = create_burst_timeline(
                burst_data['taxonomy'],
                title=f"Top Taxonomy {taxonomy_level.title()} Burst Timeline",
                color_base='#4caf50'
            )
        else:
            no_data_message = "Taxonomy elements not selected" if 'taxonomy' not in data_types else "No taxonomy burst data available"
            empty_fig = go.Figure().update_layout(title=no_data_message)
            taxonomy_burst_fig = empty_fig
            taxonomy_timeline_fig = empty_fig
        
        # Create keyword burst charts
        if 'keywords' in data_types and 'keywords' in burst_summaries and not burst_summaries['keywords'].empty:
            # Create a full DataFrame with element, period, burst_intensity for heatmap
            keyword_heatmap_data = []
            for element, df in burst_data['keywords'].items():
                if not df.empty and 'period' in df.columns and 'burst_intensity' in df.columns:
                    # Get max burst intensity for each period
                    period_data = df.groupby('period')['burst_intensity'].max().reset_index()
                    for _, row in period_data.iterrows():
                        keyword_heatmap_data.append({
                            'element': element,
                            'period': row['period'],
                            'burst_intensity': row['burst_intensity']
                        })
            
            keyword_heatmap_df = pd.DataFrame(keyword_heatmap_data)
            keyword_burst_fig = create_burst_heatmap(
                keyword_heatmap_df,
                title="Keyword Burst Intensity by Period",
                color_scale=[[0, '#f7f7f7'], [0.4, '#bbdefb'], [0.7, '#64b5f6'], [1, '#1976d2']]  # Blue scale
            )
            
            keyword_timeline_fig = create_burst_timeline(
                burst_data['keywords'],
                title="Top Keywords Burst Timeline",
                color_base='#2196f3'
            )
        else:
            no_data_message = "Keywords not selected" if 'keywords' not in data_types else "No keyword burst data available"
            empty_fig = go.Figure().update_layout(title=no_data_message)
            keyword_burst_fig = empty_fig
            keyword_timeline_fig = empty_fig
        
        # Create named entity burst charts
        if 'named_entities' in data_types and 'named_entities' in burst_summaries and not burst_summaries['named_entities'].empty:
            # Create a full DataFrame with element, period, burst_intensity for heatmap
            entity_heatmap_data = []
            for element, df in burst_data['named_entities'].items():
                if not df.empty and 'period' in df.columns and 'burst_intensity' in df.columns:
                    # Get max burst intensity for each period
                    period_data = df.groupby('period')['burst_intensity'].max().reset_index()
                    for _, row in period_data.iterrows():
                        entity_heatmap_data.append({
                            'element': element,
                            'period': row['period'],
                            'burst_intensity': row['burst_intensity']
                        })
            
            entity_heatmap_df = pd.DataFrame(entity_heatmap_data)
            entity_burst_fig = create_burst_heatmap(
                entity_heatmap_df,
                title="Named Entity Burst Intensity by Period",
                color_scale=[[0, '#f7f7f7'], [0.4, '#ffe0b2'], [0.7, '#ffb74d'], [1, '#e65100']]  # Orange scale
            )
            
            entity_timeline_fig = create_burst_timeline(
                burst_data['named_entities'],
                title="Top Named Entities Burst Timeline",
                color_base='#ff9800'
            )
        else:
            no_data_message = "Named entities not selected" if 'named_entities' not in data_types else "No named entity burst data available"
            empty_fig = go.Figure().update_layout(title=no_data_message)
            entity_burst_fig = empty_fig
            entity_timeline_fig = empty_fig
        
        return {'display': 'block'}, comparison_fig, citespace_timeline_fig, overview_timeline_fig, taxonomy_burst_fig, taxonomy_timeline_fig, keyword_burst_fig, keyword_timeline_fig, entity_burst_fig, entity_timeline_fig