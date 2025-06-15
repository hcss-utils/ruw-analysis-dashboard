#!/usr/bin/env python
# coding: utf-8

"""
Enhanced Burstiness tab layout and callbacks for the dashboard.
This tab provides analysis of bursts in taxonomic elements, keywords, and named entities,
using Kleinberg's burst detection algorithm to identify significant spikes in frequency,
visualized in CiteSpace-inspired styles with additional features:

- Algorithm parameter controls for fine-tuning
- Concordance table integration for document exploration
- Enhanced visualizations (CiteSpace timeline, co-occurrence network, predictions)
- Historical event management and annotation
- Document linking and cross-referencing
- Interactive callbacks for all new functionality
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from urllib.parse import quote

import pandas as pd
import numpy as np
import dash
from dash import html, dcc, callback_context
from dash.dependencies import Input, Output, State, ALL, MATCH
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from dash.exceptions import PreventUpdate

import sys
import os

# Add the project root to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.data_fetchers_freshness import get_burst_data_for_periods, calculate_burst_summaries
from database.data_fetchers import fetch_all_databases, fetch_date_range
from database.connection import get_engine
from components.layout import create_filter_card
# removed import create_collapsible_card
from config import (
    LANGUAGE_OPTIONS,
    SOURCE_TYPE_OPTIONS,
    FRESHNESS_FILTER_OPTIONS as BURSTINESS_FILTER_OPTIONS
)
from utils.keyword_mapping import get_mapping_status
from utils.burst_detection import (
    kleinberg_burst_detection,
    kleinberg_multi_state_burst_detection,
    detect_burst_co_occurrences,
    validate_bursts_statistically,
    find_cascade_patterns
)
from visualizations.bursts import (
    create_burst_heatmap,
    create_burst_summary_chart,
    create_burst_timeline,
    create_burst_comparison_chart,
    create_citespace_timeline,
    create_enhanced_citespace_timeline,
    create_predictive_visualization,
    load_historical_events,
    prepare_document_links
)
from visualizations.co_occurrence import (
    create_co_occurrence_network
)

# Theme colors for consistency
THEME_BLUE = "#13376f"  # Main dashboard theme color
TAXONOMY_COLOR = '#4caf50'  # Green for taxonomy
KEYWORD_COLOR = '#2196f3'   # Blue for keywords
ENTITY_COLOR = '#ff9800'    # Orange for entities

# Global variables to store data
global_burst_data = {}
global_burst_summaries = {}
global_document_links = {}
global_historical_events = []

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

# Algorithm parameter options
ALGORITHM_OPTIONS = [
    {'label': 'Basic Kleinberg (2-state)', 'value': 'basic'},
    {'label': 'Multi-state Kleinberg', 'value': 'multi_state'},
    {'label': 'Statistical Validation', 'value': 'statistical'},
    {'label': 'Density-based', 'value': 'density'},
    {'label': 'Cascade Detection', 'value': 'cascade'},
    {'label': 'Events Model', 'value': 'events'}
]

# Default historical events
DEFAULT_HISTORICAL_EVENTS = [
    {
        "date": "2022-02-24",
        "period": "Feb 2022", 
        "event": "Russian Invasion of Ukraine Begins",
        "impact": 1.0,
        "description": "Russia launches a full-scale invasion of Ukraine."
    },
    {
        "date": "2022-04-03",
        "period": "Apr 2022",
        "event": "Bucha Massacre Revealed",
        "impact": 0.9,
        "description": "Discovery of civilian killings in Bucha after Russian withdrawal."
    },
    {
        "date": "2022-09-21",
        "period": "Sep 2022",
        "event": "Russian Mobilization",
        "impact": 0.8,
        "description": "Russia announces partial military mobilization."
    },
    {
        "date": "2023-06-06",
        "period": "Jun 2023",
        "event": "Kakhovka Dam Collapse",
        "impact": 0.7,
        "description": "Massive flooding after the collapse of the Kakhovka Dam."
    },
    {
        "date": "2023-08-23",
        "period": "Aug 2023",
        "event": "Wagner Group Leader Death",
        "impact": 0.6,
        "description": "Yevgeny Prigozhin reportedly killed in plane crash."
    }
]


def create_burstiness_tab_layout():
    """
    Create the enhanced Burstiness tab layout with all advanced features.
    
    Returns:
        html.Div: Burstiness tab layout with all enhancements
    """
    # Get database options for filters
    db_options = [{'label': 'All Databases', 'value': 'ALL'}]
    try:
        databases = fetch_all_databases()
        db_options.extend([{'label': db, 'value': db} for db in databases])
    except Exception as e:
        logging.error(f"Error fetching database options: {e}")
    
    # Get keyword mapping status
    mapping_status = get_mapping_status()
    logging.info(f"Keyword mapping status: {mapping_status}")
    
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
        ], style={"display": "flex", "align-items": "center", "margin-bottom": "20px"}, id="burstiness-top"),
        
        # Fixed-position "Back to Top" button (matches style from other tabs)
        html.A(
            html.Button(
                "↑ Back to Tabs", 
                id="burstiness-back-to-tabs-btn",
                style={
                    "position": "fixed", 
                    "bottom": "20px", 
                    "right": "20px", 
                    "z-index": "9999",
                    "background-color": THEME_BLUE,
                    "color": "white",
                    "font-weight": "bold",
                    "border": "none",
                    "border-radius": "4px",
                    "padding": "10px 20px",
                    "box-shadow": "0 4px 8px rgba(0,0,0,0.2)",
                    "cursor": "pointer",
                    "width": "200px"
                }
            ),
            href="#burstiness-tabs"
        ),
        
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
                    ], width=6),
                    # Algorithm selection - removed to be replaced with model cards below
                    dbc.Col([
                        html.Label("Burst Detection Models:"),
                        html.P("Select a model by clicking on it", className="text-muted small"),
                    ], width=6),
                ]),
                
                # Model cards for burst detection algorithms
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            # Model cards container with horizontal layout
                            html.Div([
                                # Basic Kleinberg model card
                                html.Div([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H5("Basic Kleinberg", className="card-title text-center"),
                                            html.P("2-state burst detection", className="card-text text-center small"),
                                            html.Div(className="text-center"),
                                        ], className="p-3"),
                                    ], id="model-card-basic-inner", className="burst-model-card active", 
                                    style={"cursor": "pointer", "border": "2px solid #13376f"}),
                                ], id="model-card-basic", n_clicks=0),
                                
                                # Multi-state Kleinberg model card
                                html.Div([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H5("Multi-state", className="card-title text-center"),
                                            html.P("Advanced burst detection", className="card-text text-center small"),
                                            html.Div(className="text-center"),
                                        ], className="p-3"),
                                    ], id="model-card-multi_state-inner", className="burst-model-card",
                                    style={"cursor": "pointer", "border": "2px solid transparent"}),
                                ], id="model-card-multi_state", n_clicks=0),
                                
                                # Statistical Validation model card
                                html.Div([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H5("Statistical", className="card-title text-center"),
                                            html.P("With validation", className="card-text text-center small"),
                                            html.Div(className="text-center"),
                                        ], className="p-3"),
                                    ], id="model-card-statistical-inner", className="burst-model-card",
                                    style={"cursor": "pointer", "border": "2px solid transparent"}),
                                ], id="model-card-statistical", n_clicks=0),
                                
                                # Density-based model card
                                html.Div([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H5("Density-based", className="card-title text-center"),
                                            html.P("Density detection", className="card-text text-center small"),
                                            html.Div(className="text-center"),
                                        ], className="p-3"),
                                    ], id="model-card-density-inner", className="burst-model-card",
                                    style={"cursor": "pointer", "border": "2px solid transparent"}),
                                ], id="model-card-density", n_clicks=0),
                                
                                # Cascade Detection model card
                                html.Div([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H5("Cascade", className="card-title text-center"),
                                            html.P("Pattern detection", className="card-text text-center small"),
                                            html.Div(className="text-center"),
                                        ], className="p-3"),
                                    ], id="model-card-cascade-inner", className="burst-model-card",
                                    style={"cursor": "pointer", "border": "2px solid transparent"}),
                                ], id="model-card-cascade", n_clicks=0),
                                
                                # Events model card (disabled)
                                html.Div([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H5("Events Model", className="card-title text-center text-muted"),
                                            html.P("Not yet activated", className="card-text text-center small text-danger"),
                                            html.Div(className="text-center"),
                                        ], className="p-3"),
                                    ], id="model-card-events-inner", className="burst-model-card",
                                    style={"cursor": "not-allowed", "border": "2px solid transparent", "opacity": "0.6"}),
                                ], id="model-card-events", n_clicks=0),
                                
                            ], className="burst-models-container", style={
                                "display": "flex",
                                "flex-direction": "row",
                                "flex-wrap": "nowrap",
                                "justify-content": "space-between",
                                "width": "100%",
                                "gap": "10px",
                                "overflow-x": "auto"
                            }),
                        ]),
                        # Store for selected algorithm
                        dcc.Store(id='burstiness-algorithm', data='basic'),
                    ], width=12),
                ], className="mt-3 mb-3"),
                
                # Algorithm parameters (collapsible)
                dbc.Row([
                    dbc.Col([
                        dbc.Button(
                            "Algorithm Parameters", 
                            id="burstiness-algorithm-params-toggle",
                            className="mb-3",
                            color="info"
                        ),
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody([
                                    dbc.Row([
                                        # Basic algorithm parameters
                                        dbc.Col([
                                            html.Div([
                                                html.Label("State Parameter (s):"),
                                                dcc.Slider(
                                                    id='burstiness-s-parameter',
                                                    min=1.0,
                                                    max=5.0,
                                                    step=0.1,
                                                    value=2.0,
                                                    marks={i: str(i) for i in range(1, 6)},
                                                    tooltip={"placement": "bottom", "always_visible": True}
                                                ),
                                                html.Small("Higher values = more stringent burst detection", 
                                                         className="text-muted"),
                                            ], id="basic-params"),
                                        ], width=6),
                                        
                                        # Multi-state algorithm parameters
                                        dbc.Col([
                                            html.Div([
                                                html.Label("Number of States:"),
                                                dcc.Slider(
                                                    id='burstiness-num-states',
                                                    min=2,
                                                    max=5,
                                                    step=1,
                                                    value=3,
                                                    marks={i: str(i) for i in range(2, 6)},
                                                    tooltip={"placement": "bottom", "always_visible": True}
                                                ),
                                                html.Label("Transition Cost (γ):", className="mt-2"),
                                                dcc.Slider(
                                                    id='burstiness-gamma-parameter',
                                                    min=0.1,
                                                    max=3.0,
                                                    step=0.1,
                                                    value=1.0,
                                                    marks={i: str(i) for i in range(0, 4)},
                                                    tooltip={"placement": "bottom", "always_visible": True}
                                                ),
                                                html.Small("Higher γ = harder to transition between states", 
                                                         className="text-muted"),
                                            ], id="multi-state-params"),
                                        ], width=6),
                                    ]),
                                    
                                    # Statistical validation parameters
                                    dbc.Row([
                                        dbc.Col([
                                            html.Div([
                                                html.Label("Confidence Level:"),
                                                dcc.Slider(
                                                    id='burstiness-confidence-level',
                                                    min=0.8,
                                                    max=0.99,
                                                    step=0.01,
                                                    value=0.95,
                                                    marks={0.8: "80%", 0.9: "90%", 0.95: "95%", 0.99: "99%"},
                                                    tooltip={"placement": "bottom", "always_visible": True}
                                                ),
                                                html.Label("Window Size:", className="mt-2"),
                                                dcc.Slider(
                                                    id='burstiness-window-size',
                                                    min=2,
                                                    max=10,
                                                    step=1,
                                                    value=5,
                                                    marks={i: str(i) for i in range(2, 11, 2)},
                                                    tooltip={"placement": "bottom", "always_visible": True}
                                                ),
                                                html.Small("Periods to consider for local baseline calculation", 
                                                         className="text-muted"),
                                            ], id="statistical-params"),
                                        ], width=12),
                                    ], className="mt-3"),
                                ])
                            ),
                            id="burstiness-algorithm-params-collapse",
                            is_open=False,
                        ),
                    ], width=12),
                ]),
                
                # All collapsible sections in one horizontal row
                dbc.Row([
                    # Event Management (collapsible)
                    dbc.Col([
                        dbc.Button(
                            "Historical Events", 
                            id="burstiness-events-toggle",
                            className="mb-3 w-100",
                            color="danger"
                        ),
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Include Events in Visualizations:"),
                                            dbc.Checklist(
                                                id="burstiness-include-events",
                                                options=[{'label': 'Show Historical Events', 'value': 'show_events'}],
                                                value=['show_events'],
                                                switch=True,
                                            ),
                                            html.Div([
                                                html.Label("Manage Historical Events:"),
                                                html.Div(
                                                    id="burstiness-events-table-container",
                                                    children=[
                                                        # This will be populated by callback
                                                        html.Div(id="burstiness-events-table")
                                                    ],
                                                    style={"maxHeight": "250px", "overflowY": "auto"}
                                                ),
                                                dbc.Button(
                                                    "Add New Event", 
                                                    id="burstiness-add-event-btn", 
                                                    color="primary", 
                                                    size="sm",
                                                    className="mt-2"
                                                ),
                                            ]),
                                        ], width=12),
                                    ]),
                                ])
                            ),
                            id="burstiness-events-collapse",
                            is_open=False,
                        ),
                    ], width=4),
                    
                    # Standard filters (collapsible)
                    dbc.Col([
                        dbc.Button(
                            "Standard Filters", 
                            id="burstiness-standard-filters-toggle",
                            className="mb-3 w-100",
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
                                        ], width=12, lg=4),
                                        dbc.Col([
                                            html.Label("Database:"),
                                            dcc.Dropdown(
                                                id='burstiness-database-dropdown',
                                                options=db_options,
                                                value='ALL',
                                                clearable=False,
                                            )
                                        ], width=12, lg=4),
                                        dbc.Col([
                                            html.Label("Source Type:"),
                                            dcc.Dropdown(
                                                id='burstiness-source-type-dropdown',
                                                options=SOURCE_TYPE_OPTIONS,
                                                value='ALL',
                                                clearable=False,
                                            )
                                        ], width=12, lg=4),
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
                    ], width=4),
                    
                    # Data type filters (collapsible)
                    dbc.Col([
                        dbc.Button(
                            "Data Type Filters", 
                            id="burstiness-datatype-filters-toggle",
                            className="mb-3 w-100",
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
                                        ], width=12, lg=4),
                                        
                                        # Keywords options
                                        dbc.Col([
                                            html.Label("Keywords (Consolidated):"),
                                            dbc.Checklist(
                                                id="burstiness-include-keywords",
                                                options=[{'label': 'Include Mapped Keywords', 'value': 'keywords'}],
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
                                        ], width=12, lg=4),
                                        
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
                                        ], width=12, lg=4),
                                    ]),
                                ])
                            ),
                            id="burstiness-datatype-filters-collapse",
                            is_open=True,
                        ),
                    ], width=4),
                ], className="mb-3"), # End of horizontal row for all collapsible sections
                
                # Separate row for Visualization Options since it's wider
                dbc.Row([
                    # Visualization Options
                    dbc.Col([
                        dbc.Button(
                            "Visualization Options", 
                            id="burstiness-viz-options-toggle",
                            className="mb-3",
                            color="info"
                        ),
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Advanced Visualizations:"),
                                            dbc.Checklist(
                                                id="burstiness-viz-options",
                                                options=[
                                                    {'label': 'Show Co-occurrence Network', 'value': 'show_network'},
                                                    {'label': 'Show Predictive Analysis', 'value': 'show_prediction'},
                                                    {'label': 'Show Document Links', 'value': 'show_doc_links'}
                                                ],
                                                value=['show_network', 'show_prediction'],
                                                inline=True
                                            ),
                                        ], width=6),
                                        dbc.Col([
                                            html.Label("Network Parameters:"),
                                            html.Div([
                                                html.Label("Min. Co-occurrence Strength:"),
                                                dcc.Slider(
                                                    id='burstiness-network-min-strength',
                                                    min=0.1,
                                                    max=0.9,
                                                    step=0.1,
                                                    value=0.3,
                                                    marks={i/10: str(i/10) for i in range(1, 10)},
                                                    tooltip={"placement": "bottom", "always_visible": True}
                                                ),
                                            ]),
                                            html.Div([
                                                html.Label("Prediction Periods:"),
                                                dcc.Slider(
                                                    id='burstiness-prediction-periods',
                                                    min=1,
                                                    max=5,
                                                    step=1,
                                                    value=2,
                                                    marks={i: str(i) for i in range(1, 6)},
                                                    tooltip={"placement": "bottom", "always_visible": True}
                                                ),
                                            ], className="mt-2"),
                                        ], width=6),
                                    ]),
                                ])
                            ),
                            id="burstiness-viz-options-collapse",
                            is_open=False,
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
        
        # Loading spinner for the entire results section - using "default" type to match other tabs
        dcc.Loading(
            id="loading-burstiness-results",
            type="default",
            color=THEME_BLUE,
            children=[
                # Results section (initially hidden)
                html.Div([
                    # Add the document concordance section (initially hidden)
                    html.Div([
                        html.H4("Document Concordance", className="mt-3"),
                        html.P(
                            "This table shows document occurrences related to the selected burst element. "
                            "Click on table rows to view document content."
                        ),
                        dbc.Card([
                            dbc.CardBody([
                                # Dynamic concordance table will be inserted here
                                html.Div(id="burstiness-concordance-table"),
                                # Pagination for concordance table
                                dbc.Pagination(
                                    id="burstiness-concordance-pagination",
                                    max_value=1,  # Will be updated by callback
                                    first_last=True,
                                    previous_next=True,
                                    style={"marginTop": "10px", "justifyContent": "center"}
                                ),
                            ])
                        ]),
                        # Document preview modal
                        dbc.Modal([
                            dbc.ModalHeader(dbc.ModalTitle("Document Preview"), close_button=True),
                            dbc.ModalBody([
                                html.Div(id="burstiness-document-preview")
                            ]),
                            dbc.ModalFooter(
                                dbc.Button("Close", id="burstiness-close-preview", className="ms-auto")
                            ),
                        ], id="burstiness-document-modal", size="lg"),
                    ], id="burstiness-concordance-section", style={"display": "none"}),
                    
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
                                            type="default",
                                            color=THEME_BLUE,
                                            children=[dcc.Graph(id='burstiness-comparison-chart', style={'height': '500px'})]
                                        ),
                                    ], width=12),
                                ]),
                                
                                dbc.Row([
                                    dbc.Col([
                                        html.H4("Enhanced CiteSpace-Style Burst Timeline", className="mt-3"),
                                        html.P("This visualization shows bursts as horizontal segments with varying thickness based on intensity, with historical events and document links."),
                                        dcc.Loading(
                                            id="loading-enhanced-citespace-timeline",
                                            type="default",
                                            color=THEME_BLUE,
                                            children=[dcc.Graph(id='burstiness-enhanced-citespace-timeline', style={'height': '600px'})]
                                        )
                                    ], width=12),
                                ]),
                                
                                # Co-occurrence network visualization (initially hidden)
                                html.Div([
                                    dbc.Row([
                                        dbc.Col([
                                            html.H4("Burst Co-occurrence Network", className="mt-3"),
                                            html.P("This network visualization shows relationships between elements that burst together."),
                                            dcc.Loading(
                                                id="loading-co-occurrence-network",
                                                type="default",
                                                color=THEME_BLUE,
                                                children=[dcc.Graph(id='burstiness-co-occurrence-network', style={'height': '600px'})]
                                            )
                                        ], width=12),
                                    ]),
                                ], id="burstiness-network-container", style={"display": "none"}),
                                
                                # Predictive visualization (initially hidden)
                                html.Div([
                                    dbc.Row([
                                        dbc.Col([
                                            html.H4("Burst Trend Prediction", className="mt-3"),
                                            html.P("This visualization predicts future burst trends based on historical patterns."),
                                            dcc.Loading(
                                                id="loading-predictive-visualization",
                                                type="default",
                                                color=THEME_BLUE,
                                                children=[dcc.Graph(id='burstiness-predictive-visualization', style={'height': '600px'})]
                                            )
                                        ], width=12),
                                    ]),
                                ], id="burstiness-prediction-container", style={"display": "none"}),
                                
                                dbc.Row([
                                    dbc.Col([
                                        html.H4("CiteSpace-Style Burst Timeline", className="mt-3"),
                                        html.P("This visualization shows bursts as horizontal segments with varying thickness based on intensity, similar to CiteSpace."),
                                        dcc.Loading(
                                            id="loading-citespace-timeline",
                                            type="default",
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
                                            type="default",
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
                                            type="default",
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
                                            type="default",
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
                                        html.P("This chart shows the keywords with the highest burst intensity. Standardized through keyword mapping."),
                                        dcc.Loading(
                                            id="loading-keyword-chart",
                                            type="default",
                                            color=THEME_BLUE,
                                            children=[dcc.Graph(id='keyword-burst-chart', style={'height': '600px'})]
                                        )
                                    ], width=12),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        html.H4("Keywords Burst Timeline", className="mt-3"),
                                        html.P("This timeline shows how the burst intensity of top keywords changes over time. Keywords are consolidated using keyword mapping."),
                                        dcc.Loading(
                                            id="loading-keyword-timeline",
                                            type="default",
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
                                            type="default",
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
                                            type="default",
                                            color=THEME_BLUE,
                                            children=[dcc.Graph(id='entity-burst-timeline', style={'height': '500px'})]
                                        )
                                    ], width=12),
                                ]),
                            ], className="p-4")
                        ], label="Named Entities", tab_id="entities")
                    ], id="burstiness-tabs"),
                    
                    # Download buttons - matches the style from Search tab
                    html.Div([
                        dbc.Button("Download CSV", id="burstiness-btn-csv", color="success", className="me-2"),
                        dbc.Button("Download JSON", id="burstiness-btn-json", color="success"),
                        dbc.Button("Export Events", id="burstiness-btn-export-events", color="success", className="ms-2"),
                        # Hidden download components
                        dcc.Download(id="burstiness-download-csv"),
                        dcc.Download(id="burstiness-download-json"),
                        dcc.Download(id="burstiness-download-events"),
                    ], id='burstiness-download-buttons', style={'margin-top': '20px', 'text-align': 'center', 'display': 'none'})
                ], id="burstiness-results", style={"display": "none"})
            ]
        ),
        
        # Event management modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Add Historical Event")),
            dbc.ModalBody([
                dbc.Form([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Event Name"),
                            dbc.Input(id="burstiness-event-name", type="text", placeholder="Enter event name"),
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Event Date"),
                            dbc.Input(id="burstiness-event-date", type="date"),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Period"),
                            dbc.Input(id="burstiness-event-period", type="text", placeholder="e.g., Feb 2023"),
                        ], width=6),
                    ], className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Event Description"),
                            dbc.Textarea(id="burstiness-event-description", placeholder="Describe the event"),
                        ], width=12),
                    ], className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Event Impact (0.1-1.0)"),
                            dcc.Slider(
                                id='burstiness-event-impact',
                                min=0.1,
                                max=1.0,
                                step=0.1,
                                value=0.5,
                                marks={i/10: str(i/10) for i in range(1, 11)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                        ], width=12),
                    ], className="mt-3"),
                ]),
                dbc.Alert(
                    "Please fill in all required fields",
                    id="burstiness-event-alert",
                    color="danger",
                    dismissable=True,
                    is_open=False,
                    className="mt-3"
                ),
            ]),
            dbc.ModalFooter([
                dbc.Button(
                    "Close", id="burstiness-close-event-modal", className="me-2"
                ),
                dbc.Button(
                    "Save Event", id="burstiness-save-event", color="success"
                )
            ]),
        ], id="burstiness-event-modal", is_open=False),
        
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
                        html.Strong("Enhanced CiteSpace Timeline:"), 
                        " Shows bursts with historical events and document links for context."
                    ]),
                    html.Li([
                        html.Strong("Co-occurrence Network:"), 
                        " Network visualization showing which elements burst together."
                    ]),
                    html.Li([
                        html.Strong("Trend Prediction:"), 
                        " Uses historical data to predict future burst patterns."
                    ]),
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
                
                html.H5("Advanced Features:", style={"margin-top": "20px", "color": THEME_BLUE}),
                html.Ul([
                    html.Li([
                        html.Strong("Document Concordance:"), 
                        " Explore documents related to burst elements."
                    ]),
                    html.Li([
                        html.Strong("Historical Events:"), 
                        " Manage and display significant events on the timeline."
                    ]),
                    html.Li([
                        html.Strong("Algorithm Parameters:"), 
                        " Fine-tune burst detection sensitivity and behavior."
                    ]),
                ]),
                
                html.P([
                    html.Strong("Tip:"), 
                    " Compare bursts across different time periods (weeks vs. months vs. quarters) to distinguish between short-term events and longer-term trends."
                ], style={"margin-top": "15px", "font-style": "italic"}),
                
                html.H5("Keyword Mapping:", style={"margin-top": "20px", "color": THEME_BLUE}),
                html.P([
                    "This dashboard uses keyword mapping to standardize keyword variations. For example, 'USA', 'US', and 'United States' ",
                    "are mapped to a single canonical form. This mapping improves analysis by consolidating keyword variants."
                ]),
                html.P([
                    "The keyword mapping also excludes noise terms like 'https', 'www', and other technical terms that don't ",
                    "provide meaningful insight into trends or events."
                ])
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
        
    # Callback for algorithm parameters collapse
    @app.callback(
        Output("burstiness-algorithm-params-collapse", "is_open"),
        [Input("burstiness-algorithm-params-toggle", "n_clicks")],
        [State("burstiness-algorithm-params-collapse", "is_open")],
        prevent_initial_call=True
    )
    def toggle_algorithm_params_collapse(n, is_open):
        if n:
            return not is_open
        return is_open
    
    # Callback for visualization options collapse
    @app.callback(
        Output("burstiness-viz-options-collapse", "is_open"),
        [Input("burstiness-viz-options-toggle", "n_clicks")],
        [State("burstiness-viz-options-collapse", "is_open")],
        prevent_initial_call=True
    )
    def toggle_viz_options_collapse(n, is_open):
        if n:
            return not is_open
        return is_open
        
    # Callback for events collapse
    @app.callback(
        Output("burstiness-events-collapse", "is_open"),
        [Input("burstiness-events-toggle", "n_clicks")],
        [State("burstiness-events-collapse", "is_open")],
        prevent_initial_call=True
    )
    def toggle_events_collapse(n, is_open):
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
    
    # Callback for model card selection
    @app.callback(
        [
            Output('burstiness-algorithm', 'data'),
            Output('model-card-basic-inner', 'className'),
            Output('model-card-multi_state-inner', 'className'),
            Output('model-card-statistical-inner', 'className'),
            Output('model-card-density-inner', 'className'),
            Output('model-card-cascade-inner', 'className'),
            Output('model-card-events-inner', 'className'),
        ],
        [
            Input('model-card-basic', 'n_clicks'),
            Input('model-card-multi_state', 'n_clicks'),
            Input('model-card-statistical', 'n_clicks'),
            Input('model-card-density', 'n_clicks'),
            Input('model-card-cascade', 'n_clicks'),
        ],
        [State('burstiness-algorithm', 'data')],
        prevent_initial_call=True
    )
    def update_model_selection(basic_clicks, multi_clicks, stat_clicks, density_clicks, cascade_clicks, current_algorithm):
        """Handle model card selection"""
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update
        
        # Get which card was clicked
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Extract the algorithm value from the card ID
        algorithm_map = {
            'model-card-basic': 'basic',
            'model-card-multi_state': 'multi_state',
            'model-card-statistical': 'statistical',
            'model-card-density': 'density',
            'model-card-cascade': 'cascade'
        }
        
        selected_algorithm = algorithm_map.get(triggered_id, current_algorithm)
        
        # Update card classes
        basic_class = "burst-model-card active" if selected_algorithm == 'basic' else "burst-model-card"
        multi_class = "burst-model-card active" if selected_algorithm == 'multi_state' else "burst-model-card"
        stat_class = "burst-model-card active" if selected_algorithm == 'statistical' else "burst-model-card"
        density_class = "burst-model-card active" if selected_algorithm == 'density' else "burst-model-card"
        cascade_class = "burst-model-card active" if selected_algorithm == 'cascade' else "burst-model-card"
        events_class = "burst-model-card" # Events is always inactive
        
        return selected_algorithm, basic_class, multi_class, stat_class, density_class, cascade_class, events_class
    
    # Callback to set initial model card state on page load
    @app.callback(
        [
            Output('model-card-basic', 'className', allow_duplicate=True),
            Output('model-card-multi_state', 'className', allow_duplicate=True),
            Output('model-card-statistical', 'className', allow_duplicate=True),
            Output('model-card-density', 'className', allow_duplicate=True),
            Output('model-card-cascade', 'className', allow_duplicate=True),
            Output('model-card-events', 'className', allow_duplicate=True),
        ],
        [Input('burstiness-algorithm', 'data')],
        prevent_initial_call=True
    )
    def set_initial_model_card_state(selected_algorithm):
        """Set initial active state for model cards"""
        # Default to 'basic' if no algorithm is selected
        if not selected_algorithm:
            selected_algorithm = 'basic'
        
        basic_class = "burst-model-card active" if selected_algorithm == 'basic' else "burst-model-card"
        multi_class = "burst-model-card active" if selected_algorithm == 'multi_state' else "burst-model-card"
        stat_class = "burst-model-card active" if selected_algorithm == 'statistical' else "burst-model-card"
        density_class = "burst-model-card active" if selected_algorithm == 'density' else "burst-model-card"
        cascade_class = "burst-model-card active" if selected_algorithm == 'cascade' else "burst-model-card"
        events_class = "burst-model-card" # Events is always inactive
        
        return basic_class, multi_class, stat_class, density_class, cascade_class, events_class
        
    # Callbacks to toggle visibility of algorithm-specific parameters
    @app.callback(
        [
            Output("basic-params", "style"),
            Output("multi-state-params", "style"),
            Output("statistical-params", "style")
        ],
        [Input("burstiness-algorithm", "data")]
    )
    def toggle_algorithm_options(algorithm_value):
        basic_style = {"display": "block" if algorithm_value == "basic" else "none"}
        multi_state_style = {"display": "block" if algorithm_value == "multi_state" else "none"}
        statistical_style = {"display": "block" if algorithm_value == "statistical" else "none"}
        return basic_style, multi_state_style, statistical_style
        
    # Callback to toggle visibility of advanced visualization containers
    @app.callback(
        [
            Output("burstiness-network-container", "style"),
            Output("burstiness-prediction-container", "style")
        ],
        [Input("burstiness-viz-options", "value")]
    )
    def toggle_advanced_visualizations(viz_options):
        network_style = {"display": "block" if viz_options and "show_network" in viz_options else "none"}
        prediction_style = {"display": "block" if viz_options and "show_prediction" in viz_options else "none"}
        return network_style, prediction_style
    
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
            Output('burstiness-download-buttons', 'style'),
            
            # Update visualizations
            Output('burstiness-comparison-chart', 'figure'),
            Output('burstiness-citespace-timeline', 'figure'),  # Regular CiteSpace timeline
            Output('burstiness-overview-timeline', 'figure'),
            Output('taxonomy-burst-chart', 'figure'),
            Output('taxonomy-burst-timeline', 'figure'),
            Output('keyword-burst-chart', 'figure'),
            Output('keyword-burst-timeline', 'figure'),
            Output('entity-burst-chart', 'figure'),
            Output('entity-burst-timeline', 'figure'),
            # Enhanced visualizations
            Output('burstiness-enhanced-citespace-timeline', 'figure'),
            Output('burstiness-co-occurrence-network', 'figure'),
            Output('burstiness-predictive-visualization', 'figure')
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
            State('burstiness-entities-top-n', 'value'),
            
            # Algorithm parameters
            State('burstiness-algorithm', 'data'),
            State('burstiness-s-parameter', 'value'),
            State('burstiness-num-states', 'value'),
            State('burstiness-gamma-parameter', 'value'),
            State('burstiness-confidence-level', 'value'),
            State('burstiness-window-size', 'value'),
            
            # Visualization options
            State('burstiness-viz-options', 'value'),
            State('burstiness-network-min-strength', 'value'),
            State('burstiness-prediction-periods', 'value'),
            
            # Event options
            State('burstiness-include-events', 'value')
        ]
    )
    def update_burstiness_analysis(
        n_clicks, 
        period,
        language, database, source_type, start_date, end_date, filter_value,
        include_taxonomy, taxonomy_level, include_keywords, keywords_top_n,
        include_entities, entity_types, entities_top_n,
        algorithm, s_parameter, num_states, gamma_parameter, confidence_level, window_size,
        viz_options, network_min_strength, prediction_periods,
        include_events
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
            algorithm: Burst detection algorithm to use (basic, multi_state, statistical)
            s_parameter: State parameter for basic algorithm
            num_states: Number of states for multi-state algorithm
            gamma_parameter: Transition cost parameter for multi-state algorithm
            confidence_level: Confidence level for statistical validation
            window_size: Window size for statistical validation
            viz_options: Advanced visualization options
            network_min_strength: Minimum strength for co-occurrence network
            prediction_periods: Number of periods to predict
            include_events: Whether to include historical events in visualizations
            
        Returns:
            tuple: Updated visualization figures including enhanced visualizations
        """
        if not n_clicks:
            # Initial state - return empty visualizations
            empty_fig = go.Figure().update_layout(title="No data to display yet")
            return {'display': 'none'}, {'display': 'none'}, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
        
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
            return {'display': 'block'}, {'display': 'block'}, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
        
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
        
        # Show the download buttons with the results
        download_style = {'margin-top': '20px', 'text-align': 'center', 'display': 'block'}
        
        # Apply the selected algorithm 
        processed_burst_data = {}
        for data_type, elements in burst_data.items():
            processed_burst_data[data_type] = {}
            for element, df in elements.items():
                if df.empty:
                    continue
                    
                # Apply the selected burst detection algorithm
                if algorithm == 'basic':
                    processed_burst_data[data_type][element] = kleinberg_burst_detection(
                        df, s=s_parameter
                    )
                elif algorithm == 'multi_state':
                    processed_burst_data[data_type][element] = kleinberg_multi_state_burst_detection(
                        df, num_states=num_states, gamma=gamma_parameter
                    )
                elif algorithm == 'statistical':
                    processed_burst_data[data_type][element] = validate_bursts_statistically(
                        df, confidence_level=confidence_level, window_size=window_size
                    )
                    # Map stat_burst_intensity to burst_intensity for compatibility
                    if 'stat_burst_intensity' in processed_burst_data[data_type][element].columns:
                        processed_burst_data[data_type][element]['burst_intensity'] = processed_burst_data[data_type][element]['stat_burst_intensity']
                else:
                    # Fallback to original data
                    processed_burst_data[data_type][element] = df
        
        # Generate enhanced visualizations
        # 1. Enhanced CiteSpace timeline with events
        enhanced_timeline_data = []
        for data_type, elements in processed_burst_data.items():
            prefix = "T: " if data_type == 'taxonomy' else "K: " if data_type == 'keywords' else "E: "
            for element, df in elements.items():
                if not df.empty and 'period' in df.columns and 'burst_intensity' in df.columns:
                    qualified_name = prefix + element
                    for _, row in df.iterrows():
                        enhanced_timeline_data.append({
                            'element': qualified_name,
                            'period': row['period'],
                            'burst_intensity': row['burst_intensity']
                        })
        
        # Convert to DataFrame
        if enhanced_timeline_data:
            enhanced_timeline_df = pd.DataFrame(enhanced_timeline_data)
            
            # Determine if events should be included
            historical_events = None
            if include_events and 'show_events' in (include_events or []):
                global global_historical_events
                historical_events = global_historical_events
            
            # Generate document links
            global global_document_links
            document_links = None
            if viz_options and 'show_doc_links' in viz_options:
                document_links = prepare_document_links(processed_burst_data)
                global_document_links = document_links
            
            # Create enhanced timeline
            enhanced_citespace_fig = create_enhanced_citespace_timeline(
                enhanced_timeline_df,
                historical_events=historical_events,
                document_links=document_links,
                title=f"Enhanced CiteSpace Timeline (Last 10 {period}s)"
            )
        else:
            enhanced_citespace_fig = go.Figure().update_layout(
                title="No data available for enhanced timeline"
            )
            
        # 2. Co-occurrence network
        if viz_options and 'show_network' in viz_options:
            # Detect co-occurrences
            co_occurrences = detect_burst_co_occurrences(
                processed_burst_data,
                min_burst_intensity=20.0,
                min_periods=2
            )
            
            # Create network visualization
            network_fig = create_co_occurrence_network(
                processed_burst_data,
                min_burst_intensity=20.0,
                min_periods=2,
                min_strength=network_min_strength,
                title=f"Burst Co-occurrence Network (Last 10 {period}s)"
            )
        else:
            network_fig = go.Figure().update_layout(
                title="Co-occurrence network visualization disabled"
            )
            
        # 3. Predictive visualization
        if viz_options and 'show_prediction' in viz_options:
            prediction_fig = create_predictive_visualization(
                processed_burst_data,
                prediction_periods=prediction_periods,
                confidence_level=0.9,
                min_periods_for_prediction=4,
                title=f"Burst Trend Prediction (Next {prediction_periods} {period}s)",
                top_n=5
            )
        else:
            prediction_fig = go.Figure().update_layout(
                title="Predictive visualization disabled"
            )
        
        # Store the burst data globally for download access
        global global_burst_data
        global global_burst_summaries
        global_burst_data = processed_burst_data
        global_burst_summaries = burst_summaries
        
        return {'display': 'block'}, download_style, comparison_fig, citespace_timeline_fig, overview_timeline_fig, taxonomy_burst_fig, taxonomy_timeline_fig, keyword_burst_fig, keyword_timeline_fig, entity_burst_fig, entity_timeline_fig, enhanced_citespace_fig, network_fig, prediction_fig
    
    # Make sure globals are initialized
    if 'global_burst_data' not in globals():
        global global_burst_data
        global_burst_data = {}
    if 'global_burst_summaries' not in globals():
        global global_burst_summaries
        global_burst_summaries = {}
    if 'global_document_links' not in globals():
        global global_document_links
        global_document_links = {}
    if 'global_historical_events' not in globals():
        global global_historical_events
        global_historical_events = DEFAULT_HISTORICAL_EVENTS.copy()
        
    # Event management callbacks
    @app.callback(
        Output("burstiness-event-modal", "is_open"),
        [
            Input("burstiness-add-event-btn", "n_clicks"), 
            Input("burstiness-close-event-modal", "n_clicks"),
            Input("burstiness-save-event", "n_clicks")
        ],
        [State("burstiness-event-modal", "is_open")],
        prevent_initial_call=True
    )
    def toggle_event_modal(add_clicks, close_clicks, save_clicks, is_open):
        ctx = callback_context
        if not ctx.triggered:
            return is_open
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == "burstiness-add-event-btn":
            return True
        elif button_id in ["burstiness-close-event-modal", "burstiness-save-event"]:
            return False
        return is_open
    
    @app.callback(
        [
            Output("burstiness-events-table", "children"),
            Output("burstiness-event-alert", "is_open"),
        ],
        [
            Input("burstiness-save-event", "n_clicks"),
            Input("burstiness-button", "n_clicks"),  # Refresh when analysis runs
        ],
        [
            State("burstiness-event-name", "value"),
            State("burstiness-event-date", "value"),
            State("burstiness-event-period", "value"),
            State("burstiness-event-description", "value"),
            State("burstiness-event-impact", "value"),
            State("burstiness-event-alert", "is_open"),
        ],
        prevent_initial_call=True
    )
    def manage_events(save_clicks, run_clicks, event_name, event_date, event_period, 
                     event_description, event_impact, alert_is_open):
        global global_historical_events
        
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update, alert_is_open
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == "burstiness-save-event" and save_clicks:
            if not event_name or not event_date or not event_period:
                return dash.no_update, True
            
            # Add new event to the global list
            new_event = {
                "date": event_date,
                "period": event_period,
                "event": event_name,
                "impact": event_impact or 0.5,
                "description": event_description or ""
            }
            
            global_historical_events.append(new_event)
        
        # Create events table
        table = html.Table(
            # Header
            [html.Tr([
                html.Th("Event", style={"width": "30%"}),
                html.Th("Period", style={"width": "20%"}),
                html.Th("Impact", style={"width": "20%"}),
                html.Th("Actions", style={"width": "30%"})
            ])] +
            # Rows
            [html.Tr([
                html.Td(event["event"]),
                html.Td(event["period"]),
                html.Td(f"{event['impact']:.1f}"),
                html.Td([
                    dbc.Button(
                        "Remove", 
                        id={"type": "burstiness-delete-event", "index": i},
                        color="danger", 
                        size="sm",
                        className="me-1"
                    )
                ])
            ]) for i, event in enumerate(global_historical_events)],
            className="table table-striped table-hover table-sm"
        )
        
        return table, False
    
    @app.callback(
        Output("burstiness-events-table-container", "children"),
        [Input({"type": "burstiness-delete-event", "index": ALL}, "n_clicks")],
        [State({"type": "burstiness-delete-event", "index": ALL}, "id")],
        prevent_initial_call=True
    )
    def delete_event(n_clicks, button_ids):
        global global_historical_events
        
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update
        
        # Find which button was clicked
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        button_data = json.loads(button_id)
        event_index = button_data['index']
        
        # Remove the event
        if 0 <= event_index < len(global_historical_events):
            global_historical_events.pop(event_index)
        
        # Recreate table
        table = html.Table(
            # Header
            [html.Tr([
                html.Th("Event", style={"width": "30%"}),
                html.Th("Period", style={"width": "20%"}),
                html.Th("Impact", style={"width": "20%"}),
                html.Th("Actions", style={"width": "30%"})
            ])] +
            # Rows
            [html.Tr([
                html.Td(event["event"]),
                html.Td(event["period"]),
                html.Td(f"{event['impact']:.1f}"),
                html.Td([
                    dbc.Button(
                        "Remove", 
                        id={"type": "burstiness-delete-event", "index": i},
                        color="danger", 
                        size="sm",
                        className="me-1"
                    )
                ])
            ]) for i, event in enumerate(global_historical_events)],
            className="table table-striped table-hover table-sm"
        )
        
        return table
    
    # Export events callback
    @app.callback(
        Output("burstiness-download-events", "data"),
        Input("burstiness-btn-export-events", "n_clicks"),
        prevent_initial_call=True
    )
    def export_events(n_clicks):
        if not n_clicks:
            return dash.no_update
            
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"historical_events_{timestamp}.json"
        
        # Convert to JSON string
        json_str = json.dumps(global_historical_events, indent=2)
        
        return dict(content=json_str, filename=filename)
    
    # Burst data download callbacks
    @app.callback(
        Output("burstiness-download-csv", "data"),
        Input("burstiness-btn-csv", "n_clicks"),
        prevent_initial_call=True
    )
    def download_burstiness_csv(n_clicks):
        """
        Handle CSV download for burstiness results.
        
        Args:
            n_clicks: Button clicks
            
        Returns:
            dict: Download data
        """
        if not n_clicks or not global_burst_summaries:
            return dash.no_update
        
        # Combine all summaries
        all_data = []
        
        for data_type, df in global_burst_summaries.items():
            if not df.empty:
                # Add data_type as a column
                df_copy = df.copy()
                df_copy['data_type'] = data_type
                all_data.append(df_copy)
        
        if not all_data:
            return dash.no_update
            
        # Combine into one dataframe
        combined_df = pd.concat(all_data)
        
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"burstiness_data_{timestamp}.csv"
        
        # Create a string buffer, write the CSV, and encode as bytes
        from io import StringIO
        str_buffer = StringIO()
        combined_df.to_csv(str_buffer, index=False)
        
        # Use send_string
        return dcc.send_string(str_buffer.getvalue(), filename)
    
    @app.callback(
        Output("burstiness-download-json", "data"),
        Input("burstiness-btn-json", "n_clicks"),
        prevent_initial_call=True
    )
    def download_burstiness_json(n_clicks):
        """
        Handle JSON download for burstiness results.
        
        Args:
            n_clicks: Button clicks
            
        Returns:
            dict: Download data
        """
        if not n_clicks or not global_burst_data:
            return dash.no_update
        
        # Prepare the data for JSON serialization
        json_data = {}
        
        for data_type, elements in global_burst_data.items():
            json_data[data_type] = {}
            
            for element, df in elements.items():
                if not df.empty:
                    # Convert DataFrame to records format
                    json_data[data_type][element] = df.to_dict('records')
        
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"burstiness_data_{timestamp}.json"
        
        # Convert to JSON string
        import json
        json_str = json.dumps(json_data)
        
        # Use send_string for JSON
        return dcc.send_string(json_str, filename)
        
    # Concordance table and document linking callbacks
    @app.callback(
        Output("burstiness-concordance-section", "style"),
        [
            Input("burstiness-enhanced-citespace-timeline", "clickData"),
            Input("burstiness-co-occurrence-network", "clickData")
        ],
        prevent_initial_call=True
    )
    def show_concordance_section(citespace_click, network_click):
        """Show the concordance section when a visualization element is clicked."""
        ctx = callback_context
        if not ctx.triggered:
            return {"display": "none"}
            
        # Show the section
        return {"display": "block"}
    
    @app.callback(
        [
            Output("burstiness-concordance-table", "children"),
            Output("burstiness-concordance-pagination", "max_value")
        ],
        [
            Input("burstiness-enhanced-citespace-timeline", "clickData"),
            Input("burstiness-co-occurrence-network", "clickData"),
            Input("burstiness-concordance-pagination", "active_page")
        ],
        prevent_initial_call=True
    )
    def update_concordance_table(citespace_click, network_click, page):
        """
        Update the concordance table based on selected element from visualizations.
        """
        global global_document_links
        
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update, 1
            
        # Determine which visualization was clicked
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        clickData = citespace_click if trigger_id == "burstiness-enhanced-citespace-timeline" else network_click
        
        if not clickData:
            return html.Div("Click on a visualization element to view related documents."), 1
            
        try:
            # Extract element name from click data
            if trigger_id == "burstiness-enhanced-citespace-timeline":
                element_name = clickData['points'][0]['text'].split('<br>')[0].strip()
                if 'Element:' in element_name:
                    element_name = element_name.split('Element:')[1].strip()
                elif '<b>' in element_name:
                    element_name = element_name.replace('<b>', '').replace('</b>', '')
            else:  # Network click
                element_name = clickData['points'][0]['text']
                if ':' in element_name:
                    element_name = element_name.split(':', 1)[1].strip()
            
            # Default pagination values
            items_per_page = 10
            if not page:
                page = 1
                
            # Query for documents containing this element
            # This is a placeholder that would need to be connected to your actual database
            # For demo, we'll generate mock data if we don't have real links
            if element_name not in global_document_links:
                mock_docs = []
                for i in range(25):  # 25 mock documents
                    doc_id = 10000 + i
                    mock_docs.append({
                        'id': doc_id,
                        'title': f"Document related to {element_name} - #{i+1}",
                        'date': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'),
                        'preview': f"This is a sample document that mentions {element_name} multiple times..." 
                                  f"The document continues with information about {element_name}."
                    })
                global_document_links[element_name] = mock_docs
            
            documents = global_document_links[element_name]
            
            # Calculate pagination
            total_pages = (len(documents) + items_per_page - 1) // items_per_page
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(documents))
            current_page_docs = documents[start_idx:end_idx]
            
            # Create table
            table = html.Table(
                # Header
                [html.Thead(html.Tr([
                    html.Th("ID"),
                    html.Th("Title"),
                    html.Th("Date"),
                    html.Th("Actions")
                ]))] +
                # Body
                [html.Tbody([
                    html.Tr([
                        html.Td(doc['id']),
                        html.Td(doc['title']),
                        html.Td(doc['date']),
                        html.Td(
                            dbc.Button(
                                "View", 
                                id={"type": "burstiness-view-doc", "index": doc['id']},
                                color="primary", 
                                size="sm"
                            )
                        )
                    ]) for doc in current_page_docs
                ])],
                className="table table-striped table-hover"
            )
            
            return table, total_pages
            
        except Exception as e:
            return html.Div(f"Error loading concordance data: {str(e)}"), 1
    
    @app.callback(
        [
            Output("burstiness-document-modal", "is_open"),
            Output("burstiness-document-preview", "children")
        ],
        [
            Input({"type": "burstiness-view-doc", "index": ALL}, "n_clicks"),
            Input("burstiness-close-preview", "n_clicks")
        ],
        [State({"type": "burstiness-view-doc", "index": ALL}, "id")],
        prevent_initial_call=True
    )
    def show_document_preview(view_clicks, close_click, button_ids):
        """Show document preview in modal when View button is clicked."""
        ctx = callback_context
        if not ctx.triggered:
            return False, dash.no_update
            
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == "burstiness-close-preview":
            return False, dash.no_update
            
        # Find which document button was clicked
        button_data = json.loads(trigger_id)
        doc_id = button_data['index']
        
        try:
            # This would typically be a database query
            # For demonstration, we'll create mock content
            doc_content = f"""
            <h4>Document #{doc_id}</h4>
            <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>
            <p><strong>Source:</strong> Sample Database</p>
            <hr>
            <p>This is the full content of document #{doc_id}. In a real implementation, 
            this would contain the actual text of the document retrieved from your database.</p>
            <p>The document would include all the relevant information about the selected topic, 
            with the burst elements highlighted or emphasized in some way.</p>
            <p>You could also include metadata, links to related documents, or other relevant 
            information to help analysts understand the context.</p>
            """
            
            return True, html.Div([
                html.Div(dangerouslySetInnerHTML={'__html': doc_content})
            ])
        except Exception as e:
            return True, html.Div(f"Error loading document: {str(e)}")