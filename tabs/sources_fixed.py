#!/usr/bin/env python
# coding: utf-8

"""
Sources tab layout and callbacks for the dashboard - FIXED VERSION.
This tab provides corpus overview and statistics.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
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

from database.data_fetchers_sources import (
    fetch_corpus_stats,
    fetch_documents_data,
    fetch_chunks_data,
    fetch_taxonomy_combinations,
    fetch_keywords_data,
    fetch_named_entities_data,
    fetch_time_series_data,
    fetch_language_time_series,
    fetch_database_time_series,
    fetch_database_breakdown
)
from components.layout import create_filter_card
from utils.helpers import create_time_series_chart, create_language_chart

# Import after adding to path
from config import DATABASE_DISPLAY_MAP


def create_sources_tab_layout(db_options: List, min_date: datetime = None, max_date: datetime = None) -> html.Div:
    """
    Create the Sources tab layout.
    """
    # Create corpus overview section
    corpus_overview = html.Div([
        html.H3("Corpus Overview", className="mb-3"),
        
        # Filter controls
        create_filter_card(
            id_prefix="sources",
            db_options=db_options,
            min_date=min_date,
            max_date=max_date
        ),
        
        # Last updated info
        html.Div([
            "Data shown here reflects the latest state of the corpus. Last updated: ",
            html.Span(datetime.now().strftime("%Y-%m-%d %H:%M"), id="sources-last-updated")
        ], className="text-muted mb-3", style={"font-size": "0.9rem"})
    ])
    
    # Create the sources tab with loading wrapper
    sources_tab = html.Div([
        corpus_overview,
        
        # Wrap EVERYTHING in loading component
        dcc.Loading(
            id="sources-main-loading",
            type="custom",
            custom_spinner=html.Div([
                # Semi-transparent background overlay
                html.Div(style={
                    'position': 'fixed',
                    'top': '0',
                    'left': '0',
                    'width': '100%',
                    'height': '100%',
                    'background-color': 'rgba(0, 0, 0, 0.3)',
                    'z-index': '9998'
                }),
                # Loading message box
                html.Div([
                    # Add the radar pulse animation
                    html.Div([
                        html.Div(className="radar-pulse", children=[
                            html.Div(className="ring-1"),
                            html.Div(className="ring-2")
                        ]),
                        html.P("Preparing data visualizations... 🎉", 
                               className="text-center mt-4", 
                               style={'color': '#13376f', 'font-weight': 'bold', 'font-size': '18px'}),
                        html.P("(Our algorithms are doing their best impression of a speed reader!)", 
                               className="text-muted text-center", 
                               style={'font-size': '14px'}),
                        html.P("Did you know? The complete corpus contains more words than the entire Harry Potter series × 100! ⚡", 
                               className="text-info text-center mt-3",
                               style={'font-style': 'italic', 'font-size': '13px'})
                    ], style={
                        'background': 'rgba(255, 255, 255, 0.98)',
                        'border': '2px solid #13376f',
                        'border-radius': '12px',
                        'padding': '40px',
                        'box-shadow': '0 4px 20px rgba(0, 0, 0, 0.15)',
                        'max-width': '500px',
                        'margin': '0 auto',
                        'position': 'relative',
                        'z-index': '9999'
                    })
                ], style={
                    'position': 'fixed',
                    'top': '50%',
                    'left': '50%',
                    'transform': 'translate(-50%, -50%)',
                    'z-index': '9999'
                })
            ]),
            children=[
                # Main content area
                html.Div([
                    # Stats container
                    html.Div(id="sources-stats-display", style={'margin-bottom': '10px'}),
                    
                    # Content container
                    html.Div(id="sources-content-container", style={'min-height': '400px'})
                ])
            ]
        ),
        
        # Sources-specific About modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Using the Sources Tab"), 
                           style={"background-color": "#13376f", "color": "white"}),
            dbc.ModalBody([
                html.P([
                    "The Sources tab provides an overview of the corpus used for this dashboard, including statistics about documents, text chunks, taxonomic elements, keywords, and named entities."
                ]),
                
                html.H5("How to Use This Tab:", style={"margin-top": "20px", "color": "#13376f"}),
                html.Ol([
                    html.Li([
                        html.Strong("Apply Filters:"), 
                        " Use the filters at the top to focus on specific subsets of data by language, database, source type, or date range."
                    ]),
                    html.Li([
                        html.Strong("Browse Subtabs:"), 
                        " Navigate through the different subtabs to explore various aspects of the corpus."
                    ]),
                    html.Li([
                        html.Strong("Analyze Statistics:"), 
                        " Review count statistics, data distributions, and coverage metrics."
                    ]),
                ]),
                
                html.H5("Understanding the Metrics:", style={"margin-top": "20px", "color": "#13376f"}),
                html.Ul([
                    html.Li([
                        html.Strong("Document Counts:"), 
                        " Total number of documents in the corpus, broken down by language, database, and source type."
                    ]),
                    html.Li([
                        html.Strong("Chunk Counts:"), 
                        " Number of text chunks extracted from documents for analysis."
                    ]),
                    html.Li([
                        html.Strong("Taxonomy Statistics:"), 
                        " Distribution of taxonomic elements across the corpus."
                    ]),
                    html.Li([
                        html.Strong("Keywords:"), 
                        " Extracted keywords and their frequency distribution across documents."
                    ]),
                    html.Li([
                        html.Strong("Named Entities:"), 
                        " Identified entities (people, places, organizations, etc.) and their types."
                    ]),
                    html.Li([
                        html.Strong("Coverage Rate:"), 
                        " Percentage of chunks that have at least one element (taxonomy/keyword/entity) assigned."
                    ]),
                ]),
                
                html.H5("Key Insights:", style={"margin-top": "20px", "color": "#13376f"}),
                html.Ul([
                    html.Li([
                        "Understand the composition of the corpus to properly interpret results in other tabs."
                    ]),
                    html.Li([
                        "Identify potential biases in the dataset based on source distribution."
                    ]),
                    html.Li([
                        "Monitor data coverage to ensure comprehensive analysis."
                    ]),
                ]),
                
                html.P([
                    html.Strong("Tip:"), 
                    " Use the 'Last Updated' timestamp to verify the recency of the data you're analyzing."
                ], style={"margin-top": "15px", "font-style": "italic"})
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-sources-about", className="ms-auto", 
                          style={"background-color": "#13376f", "border": "none"})
            ),
        ], id="sources-about-modal", size="lg", is_open=False)
    ])
    
    return sources_tab


def register_sources_tab_callbacks(app):
    """
    Register callbacks for the Sources tab - SIMPLIFIED VERSION.
    """
    # Callback to toggle the Sources About modal
    @app.callback(
        Output("sources-about-modal", "is_open"),
        [Input("open-about-sources", "n_clicks"), Input("close-sources-about", "n_clicks")],
        [State("sources-about-modal", "is_open")],
        prevent_initial_call=True
    )
    def toggle_sources_about_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open
    
    # Main callback for updating sources tab content - NO LOADING MESSAGES OUTPUT
    @app.callback(
        [
            Output("sources-stats-display", "children"),
            Output("sources-content-container", "children"),
        ],
        [
            Input("sources-filter-button", "n_clicks"),
            Input("tabs", "value")  # Add tabs input to trigger on tab change
        ],
        [
            State("sources-language-dropdown", "value"),
            State("sources-database-dropdown", "value"),
            State("sources-source-type-dropdown", "value"),
            State("sources-date-range-picker", "start_date"),
            State("sources-date-range-picker", "end_date")
        ],
        prevent_initial_call=False  # Load data automatically on initial visit
    )
    def update_sources_tab(n_clicks, active_tab, lang_val, db_val, source_type, start_date, end_date):
        """
        Update the Sources tab based on filter selections.
        The dcc.Loading component will automatically show the loading overlay.
        """
        logging.info(f"Sources tab callback triggered - n_clicks: {n_clicks}, active_tab: {active_tab}")
        
        # Check if Sources tab is active
        if active_tab != "tab-sources":
            # Don't load data if not on Sources tab
            logging.info("Not on Sources tab, skipping update")
            return dash.no_update, dash.no_update
        
        logging.info("Sources tab is active, proceeding with data fetch")
        
        # Process filters - set defaults if None for initial load
        if lang_val is None:
            lang_val = 'ALL'
        if db_val is None:
            db_val = 'ALL'
        if source_type is None:
            source_type = 'ALL'
            
        date_range = None
        if start_date is not None and end_date is not None:
            date_range = (start_date, end_date)
    
        # Format filter description
        filter_desc = []
        if lang_val and lang_val != 'ALL':
            filter_desc.append(f"Language: {lang_val}")
        if db_val and db_val != 'ALL':
            filter_desc.append(f"Database: {db_val}")
        if source_type and source_type != 'ALL':
            filter_desc.append(f"Source Type: {source_type}")
        if date_range:
            filter_desc.append(f"Date Range: {date_range[0]} to {date_range[1]}")
    
        # Create stats HTML with filter description
        corpus_stats = fetch_corpus_stats()
        stats_html = html.Div([
            html.P(f"Docs: {corpus_stats['docs_count']:,} ({corpus_stats['docs_rel_count']:,} rel) | Chunks: {corpus_stats['chunks_count']:,} ({corpus_stats['chunks_rel_count']:,} rel) | Tax: {corpus_stats['tax_levels']:,} levels | Items: {corpus_stats['items_count']:,}"),
            html.P(f"Filters: {' | '.join(filter_desc) if filter_desc else 'None'}", className="text-muted")
        ])
        
        # Fetch filtered data with error handling
        try:
            logging.info(f"Starting to fetch data for Sources tab with filters")
            
            # Full data fetch when filters are applied
            taxonomy_data = fetch_taxonomy_combinations(lang_val, db_val, source_type, date_range)
            logging.info("Taxonomy data fetched")
            
            chunks_data = fetch_chunks_data(lang_val, db_val, source_type, date_range)
            logging.info("Chunks data fetched")
            
            documents_data = fetch_documents_data(lang_val, db_val, source_type, date_range)
            logging.info("Documents data fetched")
            
            keywords_data = fetch_keywords_data(lang_val, db_val, source_type, date_range)
            logging.info("Keywords data fetched")
            
            named_entities_data = fetch_named_entities_data(lang_val, db_val, source_type, date_range)
            logging.info("Named entities data fetched")
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            error_content = html.Div(f"Error loading data: {str(e)}", className="alert alert-danger")
            return error_content, error_content
        
        # Get time series data with error handling - ONLY for the active subtab later
        # For now, we'll fetch minimal time series data
        try:
            logging.info("Fetching minimal time series data")
            # Only fetch document time series initially as it's most likely to be viewed first
            doc_time_series = fetch_time_series_data('document', lang_val, db_val, source_type, date_range)
            doc_lang_time_series = fetch_language_time_series('document', lang_val, db_val, source_type, date_range)
            doc_db_time_series = fetch_database_time_series('document', lang_val, db_val, source_type, date_range)
            doc_db_breakdown = fetch_database_breakdown('document', lang_val, db_val, source_type, date_range)
            logging.info("Initial time series data fetched")
        except Exception as e:
            logging.error(f"Error fetching time series data: {e}")
            # Use empty dataframes as fallback
            doc_time_series = pd.DataFrame()
            doc_lang_time_series = pd.DataFrame()
            doc_db_time_series = pd.DataFrame()
            doc_db_breakdown = None
        
        # Create subtabs (simplified version)
        updated_tabs_content = dcc.Tabs([
            dcc.Tab(label="Documents", children=html.Div("Documents data loaded")),
            dcc.Tab(label="Chunks", children=html.Div("Chunks data loaded")),
            dcc.Tab(label="Taxonomy Combinations", children=html.Div("Taxonomy data loaded")),
            dcc.Tab(label="Keywords", children=html.Div("Keywords data loaded")),
            dcc.Tab(label="Named Entities", children=html.Div("Entities data loaded"))
        ], id="sources-subtabs", className="custom-tabs")
        
        # Return updated content
        logging.info("Sources tab data ready, returning content")
        return stats_html, updated_tabs_content