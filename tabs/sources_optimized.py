#!/usr/bin/env python
# coding: utf-8

"""
Optimized Sources tab with progressive loading and better performance.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

import pandas as pd
import dash
from dash import html, dcc, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.data_fetchers_sources import (
    fetch_corpus_stats,
    fetch_documents_data,
    fetch_chunks_data,
    fetch_taxonomy_combinations,
    fetch_keywords_data,
    fetch_named_entities_data,
)
from components.layout import create_filter_card
from config import DATABASE_DISPLAY_MAP, THEME_COLORS


def create_sources_tab_layout(db_options: List, min_date: datetime = None, max_date: datetime = None) -> html.Div:
    """
    Create the Sources tab layout with optimized loading.
    """
    blue_color = THEME_COLORS.get('western', '#13376f')
    
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
        
        # Apply button
        html.Div([
            dbc.Button('Apply Filters', id='sources-apply-button', color="primary", className="mt-2", size="lg"),
        ], style={"text-align": "center", "margin-top": "10px", "margin-bottom": "20px"}),
        
        # Last updated info
        html.Div([
            "Data shown here reflects the latest state of the corpus. Last updated: ",
            html.Span(datetime.now().strftime("%Y-%m-%d %H:%M"), id="sources-last-updated")
        ], className="text-muted mb-3", style={"font-size": "0.9rem"})
    ])
    
    # Create the sources tab
    sources_tab = html.Div([
        corpus_overview,
        
        # Results section
        html.Div(id='sources-stats-container', className="mt-4"),
        
        # Progress indicator
        html.Div([
            dbc.Progress(id="sources-progress", value=0, striped=True, animated=True, className="mb-3"),
            html.P(id="sources-progress-text", className="text-center text-muted")
        ], id="sources-progress-container", style={"display": "none"}),
        
        # Content tabs - show immediately with loading placeholders
        dcc.Tabs([
            dcc.Tab(label="Documents", id="tab-documents", children=[
                html.Div(id="documents-content", children=[
                    html.Div([
                        dbc.Spinner(size="lg", color="primary"),
                        html.P("Loading documents data...", className="text-muted mt-2")
                    ], className="text-center p-5")
                ])
            ]),
            dcc.Tab(label="Chunks", id="tab-chunks", children=[
                html.Div(id="chunks-content", children=[
                    html.Div([
                        dbc.Spinner(size="lg", color="primary"),
                        html.P("Loading chunks data...", className="text-muted mt-2")
                    ], className="text-center p-5")
                ])
            ]),
            dcc.Tab(label="Taxonomy", id="tab-taxonomy", children=[
                html.Div(id="taxonomy-content", children=[
                    html.Div([
                        dbc.Spinner(size="lg", color="primary"),
                        html.P("Loading taxonomy data...", className="text-muted mt-2")
                    ], className="text-center p-5")
                ])
            ]),
            dcc.Tab(label="Keywords", id="tab-keywords", children=[
                html.Div(id="keywords-content", children=[
                    html.Div([
                        dbc.Spinner(size="lg", color="primary"),
                        html.P("Loading keywords data...", className="text-muted mt-2")
                    ], className="text-center p-5")
                ])
            ]),
            dcc.Tab(label="Named Entities", id="tab-entities", children=[
                html.Div(id="entities-content", children=[
                    html.Div([
                        dbc.Spinner(size="lg", color="primary"),
                        html.P("Loading named entities data...", className="text-muted mt-2")
                    ], className="text-center p-5")
                ])
            ])
        ], id="sources-subtabs", className="custom-tabs"),
        
        # Hidden store for data
        dcc.Store(id="sources-data-store"),
        
        # Interval for progressive updates
        dcc.Interval(id="sources-interval", interval=500, disabled=True),
        
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
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-sources-about", className="ms-auto", 
                          style={"background-color": "#13376f", "border": "none"})
            ),
        ], id="sources-about-modal", size="lg", is_open=False)
    ])
    
    return sources_tab


# Global variable to track loading state
loading_state = {
    'is_loading': False,
    'progress': 0,
    'current_task': '',
    'data': {}
}
loading_lock = threading.Lock()


def update_loading_state(progress: int, task: str, data_key: str = None, data_value: Any = None):
    """Thread-safe update of loading state."""
    with loading_lock:
        loading_state['progress'] = progress
        loading_state['current_task'] = task
        if data_key and data_value is not None:
            loading_state['data'][data_key] = data_value


def fetch_data_async(lang_val, db_val, source_type, date_range):
    """Fetch data asynchronously with progress updates."""
    try:
        # Reset state
        update_loading_state(0, "Starting data fetch...", 'data', {})
        
        # Fetch corpus stats first (fast)
        update_loading_state(10, "Fetching corpus statistics...")
        corpus_stats = fetch_corpus_stats()
        update_loading_state(15, "Corpus statistics loaded", 'corpus_stats', corpus_stats)
        
        # Fetch documents data
        update_loading_state(20, "Fetching documents data...")
        documents_data = fetch_documents_data(lang_val, db_val, source_type, date_range)
        update_loading_state(35, "Documents data loaded", 'documents_data', documents_data)
        
        # Fetch chunks data
        update_loading_state(40, "Fetching chunks data...")
        chunks_data = fetch_chunks_data(lang_val, db_val, source_type, date_range)
        update_loading_state(55, "Chunks data loaded", 'chunks_data', chunks_data)
        
        # Fetch taxonomy data
        update_loading_state(60, "Fetching taxonomy data...")
        taxonomy_data = fetch_taxonomy_combinations(lang_val, db_val, source_type, date_range)
        update_loading_state(70, "Taxonomy data loaded", 'taxonomy_data', taxonomy_data)
        
        # Fetch keywords data
        update_loading_state(75, "Fetching keywords data...")
        keywords_data = fetch_keywords_data(lang_val, db_val, source_type, date_range)
        update_loading_state(85, "Keywords data loaded", 'keywords_data', keywords_data)
        
        # Fetch named entities data (slowest)
        update_loading_state(90, "Fetching named entities data (this may take a moment)...")
        named_entities_data = fetch_named_entities_data(lang_val, db_val, source_type, date_range)
        update_loading_state(100, "All data loaded!", 'named_entities_data', named_entities_data)
        
        with loading_lock:
            loading_state['is_loading'] = False
            
    except Exception as e:
        logging.error(f"Error in async data fetch: {e}")
        update_loading_state(0, f"Error: {str(e)}")
        with loading_lock:
            loading_state['is_loading'] = False


def register_sources_tab_callbacks(app):
    """
    Register callbacks for the optimized Sources tab.
    """
    
    # Toggle About modal
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
    
    # Main callback to start data loading
    @app.callback(
        [
            Output("sources-interval", "disabled"),
            Output("sources-progress-container", "style"),
            Output("sources-stats-container", "children")
        ],
        [
            Input("sources-apply-button", "n_clicks"),
            Input("tabs", "value")
        ],
        [
            State("sources-language-dropdown", "value"),
            State("sources-database-dropdown", "value"),
            State("sources-source-type-dropdown", "value"),
            State("sources-date-range-picker", "start_date"),
            State("sources-date-range-picker", "end_date")
        ],
        prevent_initial_call=False
    )
    def start_data_loading(n_clicks, active_tab, lang_val, db_val, source_type, start_date, end_date):
        """Start the data loading process."""
        if active_tab != "tab-sources":
            return True, {"display": "none"}, dash.no_update
        
        # Process filters
        if lang_val is None:
            lang_val = 'ALL'
        if db_val is None:
            db_val = 'ALL'
        if source_type is None:
            source_type = 'ALL'
            
        date_range = None
        if start_date and end_date:
            date_range = (start_date.split('T')[0], end_date.split('T')[0])
        
        # Format filter description
        filter_desc = []
        if lang_val != 'ALL':
            filter_desc.append(f"Language: {lang_val}")
        if db_val != 'ALL':
            filter_desc.append(f"Database: {db_val}")
        if source_type != 'ALL':
            filter_desc.append(f"Source Type: {source_type}")
        if date_range:
            filter_desc.append(f"Date Range: {date_range[0]} to {date_range[1]}")
        
        # Create initial stats HTML
        stats_html = html.Div([
            html.P("Loading corpus statistics...", className="text-muted"),
            html.P(f"Filters: {' | '.join(filter_desc) if filter_desc else 'None'}", className="text-muted")
        ])
        
        # Start async loading if not already running
        with loading_lock:
            if not loading_state['is_loading']:
                loading_state['is_loading'] = True
                # Start background thread
                thread = threading.Thread(
                    target=fetch_data_async,
                    args=(lang_val, db_val, source_type, date_range)
                )
                thread.daemon = True
                thread.start()
        
        # Enable interval for progress updates, show progress container
        return False, {"display": "block"}, stats_html
    
    # Progress update callback
    @app.callback(
        [
            Output("sources-progress", "value"),
            Output("sources-progress-text", "children"),
            Output("documents-content", "children"),
            Output("chunks-content", "children"),
            Output("taxonomy-content", "children"),
            Output("keywords-content", "children"),
            Output("entities-content", "children")
        ],
        [Input("sources-interval", "n_intervals")],
        prevent_initial_call=True
    )
    def update_progress(n):
        """Update progress and content as data loads."""
        with loading_lock:
            progress = loading_state['progress']
            task = loading_state['current_task']
            data = loading_state['data'].copy()
        
        # Progress text
        progress_text = f"{task} ({progress}%)"
        
        # Update content for each tab as data becomes available
        documents_content = create_documents_content(data.get('documents_data')) if 'documents_data' in data else dash.no_update
        chunks_content = create_chunks_content(data.get('chunks_data')) if 'chunks_data' in data else dash.no_update
        taxonomy_content = create_taxonomy_content(data.get('taxonomy_data')) if 'taxonomy_data' in data else dash.no_update
        keywords_content = create_keywords_content(data.get('keywords_data')) if 'keywords_data' in data else dash.no_update
        entities_content = create_entities_content(data.get('named_entities_data')) if 'named_entities_data' in data else dash.no_update
        
        return progress, progress_text, documents_content, chunks_content, taxonomy_content, keywords_content, entities_content
    
    # Update stats when corpus stats are loaded
    @app.callback(
        Output("sources-stats-container", "children", allow_duplicate=True),
        [Input("sources-interval", "n_intervals")],
        [State("sources-stats-container", "children")],
        prevent_initial_call=True
    )
    def update_stats(n, current_stats):
        """Update stats container when corpus stats are available."""
        with loading_lock:
            corpus_stats = loading_state['data'].get('corpus_stats')
        
        if corpus_stats and isinstance(current_stats, html.Div):
            # Check if we haven't already updated (look for "Loading" text)
            if current_stats.children and isinstance(current_stats.children[0], html.P):
                if "Loading corpus statistics" in str(current_stats.children[0].children):
                    # Create updated stats
                    return html.Div([
                        html.P(f"Docs: {corpus_stats['docs_count']:,} ({corpus_stats['docs_rel_count']:,} rel) | "
                               f"Chunks: {corpus_stats['chunks_count']:,} ({corpus_stats['chunks_rel_count']:,} rel) | "
                               f"Tax: {corpus_stats['tax_levels']:,} levels | Items: {corpus_stats['items_count']:,}"),
                        current_stats.children[1] if len(current_stats.children) > 1 else html.Div()
                    ])
        
        return dash.no_update


def create_documents_content(data):
    """Create documents tab content."""
    if not data:
        return html.Div([
            dbc.Spinner(size="lg", color="primary"),
            html.P("Loading documents data...", className="text-muted mt-2")
        ], className="text-center p-5")
    
    return html.Div([
        html.H5("Documents Overview"),
        html.P(f"Total documents: {data.get('total_documents', 0):,}"),
        html.P(f"Relevant documents (with taxonomy): {data.get('relevant_documents', 0):,}"),
        html.P(f"Relevance rate: {data.get('relevance_rate', 0):.1f}%"),
        html.P(f"Date range: {data.get('earliest_date', 'N/A')} to {data.get('latest_date', 'N/A')}", 
               className="text-muted"),
        dcc.Graph(
            figure=go.Figure(
                data=[go.Bar(
                    x=data.get('top_databases', {}).get('labels', [])[:10],
                    y=data.get('top_databases', {}).get('values', [])[:10],
                    text=data.get('top_databases', {}).get('percentages', [])[:10],
                    texttemplate='%{text:.1f}%',
                    textposition='auto',
                )],
                layout=go.Layout(
                    title="Top 10 Databases by Document Count",
                    xaxis={'title': 'Database'},
                    yaxis={'title': 'Document Count'},
                    height=300,
                    margin=dict(l=50, r=50, t=50, b=100)
                )
            ),
            style={'height': '300px'}
        ) if data.get('top_databases') else html.Div("No database distribution data available")
    ])


def create_chunks_content(data):
    """Create chunks tab content."""
    if not data:
        return html.Div([
            dbc.Spinner(size="lg", color="primary"),
            html.P("Loading chunks data...", className="text-muted mt-2")
        ], className="text-center p-5")
    
    return html.Div([
        html.H5("Text Chunks Overview"), 
        html.P(f"Total chunks: {data.get('total_chunks', 0):,}"),
        html.P(f"Relevant chunks: {data.get('relevant_chunks', 0):,}"),
        html.P(f"Coverage rate: {data.get('coverage_rate', 0):.1f}%"),
        html.P(f"Average chunks per document: {data.get('avg_chunks_per_doc', 0):.1f}")
    ])


def create_taxonomy_content(data):
    """Create taxonomy tab content."""
    if not data:
        return html.Div([
            dbc.Spinner(size="lg", color="primary"),
            html.P("Loading taxonomy data...", className="text-muted mt-2")
        ], className="text-center p-5")
    
    return html.Div([
        html.H5("Taxonomy Elements Overview"),
        html.P(f"Total taxonomy combinations: {data.get('total_combinations', 0):,}"),
        html.P(f"Unique categories: {data.get('categories', 0):,}"),
        html.P(f"Unique subcategories: {data.get('subcategories', 0):,}"),
        html.P(f"Coverage: {data.get('coverage_percentage', 0):.1f}% of chunks have taxonomy")
    ])


def create_keywords_content(data):
    """Create keywords tab content."""
    if not data:
        return html.Div([
            dbc.Spinner(size="lg", color="primary"),
            html.P("Loading keywords data...", className="text-muted mt-2")
        ], className="text-center p-5")
    
    return html.Div([
        html.H5("Keywords Overview"),
        html.P(f"Total unique keywords: {data.get('unique_keywords', 0):,}"),
        html.P(f"Total occurrences: {data.get('total_occurrences', 0):,}"),
        dcc.Graph(
            figure=go.Figure(
                data=[go.Bar(
                    x=data.get('top_keywords', {}).get('values', [])[:15],
                    y=data.get('top_keywords', {}).get('labels', [])[:15],
                    orientation='h',
                    text=data.get('top_keywords', {}).get('values', [])[:15],
                    texttemplate='%{text:,}',
                    textposition='auto',
                )],
                layout=go.Layout(
                    title="Top 15 Keywords by Frequency",
                    xaxis={'title': 'Frequency'},
                    yaxis={'title': 'Keyword', 'autorange': 'reversed'},
                    height=400,
                    margin=dict(l=150, r=50, t=50, b=50)
                )
            ),
            style={'height': '400px'}
        ) if data.get('top_keywords') else html.Div("No keyword data available")
    ])


def create_entities_content(data):
    """Create named entities tab content."""
    if not data:
        return html.Div([
            dbc.Spinner(size="lg", color="primary"),
            html.P("Loading named entities data...", className="text-muted mt-2")
        ], className="text-center p-5")
    
    return html.Div([
        html.H5("Named Entities Overview"),
        html.P(f"Total unique entities: {data.get('unique_entities', 0):,}"),
        html.P(f"Total occurrences: {data.get('total_occurrences', 0):,}"),
        dcc.Graph(
            figure=go.Figure(
                data=[go.Pie(
                    labels=data.get('by_type', {}).get('labels', []),
                    values=data.get('by_type', {}).get('values', []),
                    hole=0.3,
                    textinfo='label+percent',
                    textposition='auto',
                )],
                layout=go.Layout(
                    title="Named Entities by Type",
                    height=350,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
            ),
            style={'height': '350px'}
        ) if data.get('by_type') else html.Div("No entity type distribution available")
    ])