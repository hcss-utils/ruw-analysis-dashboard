#!/usr/bin/env python
# coding: utf-8

"""
Sources tab layout and callbacks for the dashboard - WORKING VERSION.
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
from utils.cache import clear_cache

# Import after adding to path
from config import DATABASE_DISPLAY_MAP, THEME_COLORS, DATABASE_COLORS
from utils.helpers import hex_to_rgba


def get_all_databases_from_data(data: Dict) -> List[str]:
    """Extract all unique databases from the data to ensure consistent coloring."""
    databases = set()
    
    # Add databases from top_databases
    if 'top_databases' in data and 'labels' in data['top_databases']:
        databases.update(data['top_databases']['labels'])
    
    # Add databases from top_databases_relevant
    if 'top_databases_relevant' in data and 'labels' in data['top_databases_relevant']:
        databases.update(data['top_databases_relevant']['labels'])
    
    # Add databases from per_database_relevance
    if 'per_database_relevance' in data:
        databases.update(data['per_database_relevance'].keys())
    
    return sorted(list(databases))


def get_database_color(db_name, all_databases=None):
    """Get consistent color for a database using tab20 colors."""
    # Use TRULY DISTINCT colors - no similar shades
    tab20_colors = [
        '#e41a1c',  # bright red
        '#377eb8',  # strong blue
        '#4daf4a',  # green
        '#984ea3',  # purple
        '#ff7f00',  # orange
        '#ffff33',  # yellow
        '#a65628',  # brown
        '#f781bf',  # pink
        '#999999',  # gray
        '#66c2a5',  # teal
        '#fc8d62',  # coral
        '#8da0cb',  # periwinkle
        '#e78ac3',  # magenta
        '#a6d854',  # lime
        '#ffd92f',  # gold
        '#e5c494',  # tan
        '#b3b3b3',  # light gray
        '#1b9e77',  # dark teal
        '#d95f02',  # dark orange
        '#7570b3'   # indigo
    ]
    
    # Create a global cache for database color assignments
    if not hasattr(get_database_color, '_color_cache'):
        get_database_color._color_cache = {}
    
    # If we've seen this database before, return its assigned color
    if db_name in get_database_color._color_cache:
        return get_database_color._color_cache[db_name]
    
    # Assign the next available color
    used_colors = set(get_database_color._color_cache.values())
    for color in tab20_colors:
        if color not in used_colors:
            get_database_color._color_cache[db_name] = color
            logging.info(f"Assigned new color {color} to database '{db_name}'")
            return color
    
    # If all colors are used, cycle through them
    index = len(get_database_color._color_cache) % len(tab20_colors)
    color = tab20_colors[index]
    get_database_color._color_cache[db_name] = color
    logging.info(f"Cycling colors - assigned {color} to database '{db_name}'")
    return color


def create_sources_tab_layout(db_options: List, min_date: datetime = None, max_date: datetime = None) -> html.Div:
    """
    Create the Sources tab layout - WORKING VERSION.
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
        
        # Results section - initial information area
        html.Div(id='sources-stats-container', className="mt-4", style={"scroll-margin-top": "100px"}),
        
        # Loading message overlay container with background and radar pulse
        html.Div([
            # Semi-transparent background overlay
            html.Div(style={
                'position': 'fixed',
                'top': '0',
                'left': '0',
                'width': '100%',
                'height': '100%',
                'background-color': 'rgba(0, 0, 0, 0.5)',
                'z-index': '999'
            }),
            # Loading message box
            html.Div([
                # Add the radar pulse animation
                html.Div([
                    html.Div(className="radar-pulse", children=[
                        html.Div(className="ring-1"),
                        html.Div(className="ring-2")
                    ]),
                    html.P("📊 Loading Sources Data...", 
                           className="text-center mt-4", 
                           style={'color': blue_color, 'fontWeight': 'bold', 'fontSize': '18px'}),
                    html.P("(Our algorithms are doing their best impression of a speed reader!)", 
                           className="text-muted text-center", 
                           style={'fontSize': '14px'}),
                    html.P("💡 Fun fact: The complete corpus contains more words than the entire Harry Potter series × 100! ⚡", 
                           className="text-center mt-3",
                           style={'fontStyle': 'italic', 'fontSize': '13px', 'color': blue_color})
                ], style={
                    'background': 'rgba(255, 255, 255, 0.98)',
                    'border': '2px solid ' + blue_color,
                    'border-radius': '12px',
                    'padding': '40px',
                    'box-shadow': '0 4px 20px rgba(0, 0, 0, 0.15)',
                    'max-width': '500px',
                    'margin': '0 auto',
                    'position': 'relative',
                    # Ensure loading message appears above the global radar sweep
                    # which uses z-index 9999 in dash-loading-monitor.js
                    'z-index': '10010'
                })
            ], style={
                'position': 'fixed',
                'top': '50%',
                'left': '50%',
                'transform': 'translate(-50%, -50%)',
                'z-index': '10010'
            })
        ], id="sources-loading-messages", 
           style={
               'display': 'none'
           }),
        
        
        # Content container - initially hidden
        html.Div([
            html.Div(id="sources-content-display")
        ], 
        style={'margin-bottom': '0px', 'width': '100%', 'justify-content': 'center', 'display': 'none'},
        className="sources-content-container", id="sources-results-tabs"),
        
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


def create_documents_visualizations(data: Dict) -> html.Div:
    """
    Create visualizations for the Documents subtab.
    
    Args:
        data: Documents data dictionary from fetch_documents_data
        
    Returns:
        html.Div: Container with visualizations
    """
    import logging
    logging.info(f"Documents visualization data keys: {list(data.keys()) if data else 'None'}")
    logging.info(f"Total documents: {data.get('total_documents', 0) if data else 0}")
    
    # Check for errors
    if data and 'error' in data:
        return html.Div([
            html.H5("Error Loading Documents Data", className="text-danger"),
            html.P(f"The query timed out or encountered an error. This may be due to the large dataset size."),
            html.P(f"Error details: {data['error']}", className="text-muted"),
            html.P("Please try applying filters to reduce the data size or contact support.")
        ], className="alert alert-warning mt-3")
    
    if not data or data.get('total_documents', 0) == 0:
        return html.Div([
            html.P("No documents found with the current filters.", className="text-muted text-center mt-5")
        ])
    
    # Extract data
    total_docs = data['total_documents']
    relevant_docs = data['relevant_documents']
    irrelevant_docs = data['irrelevant_documents']
    relevance_rate = data['relevance_rate']
    earliest_date = data['earliest_date']
    latest_date = data['latest_date']
    
    # Create relevance pie chart
    relevance_fig = go.Figure(data=[go.Pie(
        labels=['Relevant', 'Irrelevant'],
        values=[relevant_docs, irrelevant_docs],
        hole=0.3,
        marker_colors=[THEME_COLORS['western'], '#cccccc'],
        textinfo='label+percent',
        textposition='auto'
    )])
    relevance_fig.update_layout(
        title="Document Relevance Distribution",
        height=350,
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    relevance_fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>'
    )
    
    # Create language distribution bar chart (RELEVANT ONLY)
    lang_labels = data.get('by_language_relevant', data['by_language'])['labels']
    lang_values = data.get('by_language_relevant', data['by_language'])['values']
    
    if lang_labels:
        # Calculate total for percentages
        total_lang_docs = sum(lang_values)
        lang_percentages = [round((v / total_lang_docs * 100), 1) if total_lang_docs > 0 else 0 for v in lang_values]
        
        # Use distinctive colors for different values
        colors = px.colors.qualitative.Dark24 if len(lang_labels) <= 24 else px.colors.qualitative.Alphabet
        
        # Create text annotations with both count and percentage
        text_annotations = [f"{v:,} ({p}%)" for v, p in zip(lang_values, lang_percentages)]
        
        # Create hover text with both values
        hover_text = [f"Language: {lang}<br>Count: {v:,}<br>Percentage: {p}%" 
                     for lang, v, p in zip(lang_labels, lang_values, lang_percentages)]
        
        lang_fig = go.Figure(data=[go.Bar(
            x=lang_values,
            y=lang_labels,
            orientation='h',
            marker_color=colors[:len(lang_labels)],
            text=text_annotations,
            textposition='auto',
            hovertext=hover_text,
            hoverinfo='text'
        )])
        lang_fig.update_layout(
            title="Relevant Documents by Language",
            xaxis_title="Number of Documents",
            yaxis_title="Language",
            height=max(300, len(lang_labels) * 40),
            margin=dict(l=100, r=20, t=40, b=40)
        )
    else:
        lang_fig = None
    
    # Create database distribution bar chart (top 10) - RELEVANT ONLY
    db_labels = data.get('top_databases_relevant', data['top_databases'])['labels']
    db_values = data.get('top_databases_relevant', data['top_databases'])['values']
    
    if db_labels:
        # Map database names for display
        db_display_labels = [DATABASE_DISPLAY_MAP.get(db, db) for db in db_labels]
        
        # Calculate total for percentages
        total_relevant_docs = sum(db_values)
        db_percentages = [round((v / total_relevant_docs * 100), 1) if total_relevant_docs > 0 else 0 for v in db_values]
        
        # Use database-specific colors with consistent ordering
        db_colors = [get_database_color(db) for db in db_labels]
        
        # Create text annotations with both count and percentage
        text_annotations = [f"{v:,} ({p}%)" for v, p in zip(db_values, db_percentages)]
        
        # Create hover text with both values
        hover_text = [f"Database: {db}<br>Count: {v:,}<br>Percentage: {p}%" 
                     for db, v, p in zip(db_display_labels, db_values, db_percentages)]
        
        db_fig = go.Figure(data=[go.Bar(
            x=db_values,
            y=db_display_labels,
            orientation='h',
            marker_color=db_colors,
            text=text_annotations,
            textposition='auto',
            hovertext=hover_text,
            hoverinfo='text'
        )])
        db_fig.update_layout(
            title="Top 10 Databases by Relevant Document Count",
            xaxis_title="Number of Documents",
            yaxis_title="Database",
            height=max(350, len(db_labels) * 40),
            margin=dict(l=150, r=20, t=40, b=40)
        )
    else:
        db_fig = None
    
    # Create statistics panel
    stats_panel = dbc.Card([
        dbc.CardBody([
            html.H5("📊 Document Statistics", className="mb-3"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.P("📄 Total Documents", className="text-muted mb-1"),
                    html.H4(f"{total_docs:,}", style={"color": "black"})
                ], md=4),
                dbc.Col([
                    html.P("✅ Relevant Documents", className="text-muted mb-1"),
                    html.H4(f"{relevant_docs:,}", style={"color": THEME_COLORS['western']})  # Same blue as relevant wedge
                ], md=4),
                dbc.Col([
                    html.P("📈 Relevance Rate", className="text-muted mb-1"),
                    html.H4(f"{relevance_rate}%", className="text-info")
                ], md=4),
            ]),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.P("📅 Date Range", className="text-muted mb-1"),
                    html.P(f"{earliest_date} to {latest_date}", className="font-weight-bold")
                ], md=12),
            ])
        ])
    ], className="mb-4")
    
    # Build layout
    layout_components = [stats_panel]
    
    if lang_fig or db_fig:
        chart_row = dbc.Row([])
        if lang_fig:
            chart_row.children.append(
                dbc.Col([
                    dcc.Graph(figure=relevance_fig, config={'displayModeBar': False})
                ], md=6)
            )
            chart_row.children.append(
                dbc.Col([
                    dcc.Graph(figure=lang_fig, config={'displayModeBar': False})
                ], md=6)
            )
        layout_components.append(chart_row)
        
        if db_fig:
            layout_components.append(
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=db_fig, config={'displayModeBar': False})
                    ], md=12)
                ], className="mt-3")
            )
    
    # Add timeline section
    layout_components.append(
        html.Hr(className="mt-5 mb-4")
    )
    
    # Create timeline for relevant documents
    time_series_data = data.get('time_series_relevant', {})
    logging.info(f"Time series data: {time_series_data}")
    
    if not time_series_data or not time_series_data.get('dates'):
        # If no time series data, show a message
        layout_components.append(
            dbc.Row([
                dbc.Col([
                    html.P("Timeline data not available", className="text-muted text-center")
                ], md=12)
            ])
        )
    else:
        timeline_fig = go.Figure()
        timeline_fig.add_trace(go.Scatter(
            x=time_series_data['dates'],
            y=time_series_data['values'],
            mode='lines+markers',
            name='Relevant Documents',
            line=dict(color=THEME_COLORS['western'], width=2),
            marker=dict(size=6)
        ))
        timeline_fig.update_layout(
            title="Relevant Documents Over Time",
            xaxis_title="Date",
            yaxis_title="Number of Documents",
            height=300,
            margin=dict(l=60, r=20, t=60, b=60),
            hovermode='x unified'
        )
        
        layout_components.append(
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=timeline_fig, config={'displayModeBar': False})
                ], md=12)
            ])
        )
    
    # Add per-database relevance donuts
    db_relevance_data = data.get('per_database_relevance', {})
    logging.info(f"Database relevance data: {list(db_relevance_data.keys()) if db_relevance_data else 'None'}")
    logging.info(f"Number of databases: {len(db_relevance_data) if db_relevance_data else 0}")
    
    if not db_relevance_data:
        # If no database relevance data, show a message
        layout_components.append(
            html.Div([
                html.H5("Database Relevance Breakdown", className="text-center mt-4 mb-3"),
                html.P("No database relevance data available", className="text-muted text-center")
            ])
        )
    else:
        # Create small donuts for each database
        donut_rows = []
        databases = list(db_relevance_data.keys())[:12]  # Limit to 12 databases
        
        for i in range(0, len(databases), 4):  # 4 donuts per row
            row_databases = databases[i:i+4]
            donut_cols = []
            
            for db in row_databases:
                db_data = db_relevance_data[db]
                relevant = db_data.get('relevant', 0)
                irrelevant = db_data.get('irrelevant', 0)
                total = relevant + irrelevant
                
                if total > 0:
                    # Create small donut for this database
                    donut_fig = go.Figure(data=[go.Pie(
                        labels=['Relevant', 'Irrelevant'],
                        values=[relevant, irrelevant],
                        hole=0.5,
                        marker_colors=[THEME_COLORS['western'], '#e0e0e0'],
                        textinfo='percent',
                        textposition='inside',
                        insidetextorientation='radial'
                    )])
                    
                    # Format the total with thousands separator
                    total_formatted = f"{total:,}"
                    
                    # Get database color with consistent ordering
                    db_color = get_database_color(db)
                    
                    donut_fig.update_layout(
                        title=dict(
                            text=f"<b>{DATABASE_DISPLAY_MAP.get(db, db)}</b>",
                            font=dict(size=16, color=db_color),
                            x=0.5,
                            xanchor='center',
                            y=0.95,  # Position title closer to the donut
                            yanchor='top'
                        ),
                        height=300,  # Much larger height for visibility
                        showlegend=False,
                        margin=dict(l=5, r=5, t=50, b=5),  # Enough margin for full title visibility
                        annotations=[
                            dict(
                                text=f'{total_formatted}<br>docs',
                                x=0.5, y=0.5,
                                font_size=16,
                                showarrow=False
                            )
                        ]
                    )
                    
                    donut_fig.update_traces(
                        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>'
                    )
                    
                    donut_cols.append(
                        dbc.Col([
                            dcc.Graph(figure=donut_fig, config={'displayModeBar': False}, style={'marginBottom': '-20px', 'marginTop': '-10px'})
                        ], md=3, sm=6)
                    )
            
            if donut_cols:
                donut_rows.append(dbc.Row(donut_cols, className="mb-0"))
        
        if donut_rows:
            layout_components.append(
                html.Div([
                    html.H5("Relevance Breakdown by Database", className="text-center mt-4 mb-3"),
                    *donut_rows
                ])
            )
    
    return html.Div(layout_components)


def create_chunks_visualizations(data: Dict) -> html.Div:
    """
    Create visualizations for the Chunks subtab.
    
    Args:
        data: Chunks data dictionary from fetch_chunks_data
        
    Returns:
        html.Div: Container with visualizations
    """
    if not data or data.get('total_chunks', 0) == 0:
        return html.Div([
            html.P("No chunks found with the current filters.", className="text-muted text-center mt-5")
        ])
    
    # Extract data
    total_chunks = data['total_chunks']
    relevant_chunks = data['relevant_chunks']
    irrelevant_chunks = data['irrelevant_chunks']
    relevance_rate = data['relevance_rate']
    avg_chunk_length = data.get('avg_chunk_length', 0)
    
    # Create relevance donut chart with total in center
    relevance_fig = go.Figure(data=[go.Pie(
        labels=['Relevant', 'Irrelevant'],
        values=[relevant_chunks, irrelevant_chunks],
        hole=0.6,
        marker_colors=[THEME_COLORS['western'], '#e0e0e0'],
        textinfo='label+percent',
        textposition='auto'
    )])
    relevance_fig.update_layout(
        title="Chunk Relevance Distribution",
        height=350,
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=20),
        annotations=[dict(
            text=f'{total_chunks:,}<br>Total',
            x=0.5, y=0.5,
            font_size=20,
            showarrow=False
        )]
    )
    
    # Create language distribution pie chart - RELEVANT ONLY
    lang_labels = data.get('by_language_relevant', data['by_language'])['labels']
    lang_values = data.get('by_language_relevant', data['by_language'])['values']
    
    if lang_labels:
        lang_fig = go.Figure(data=[go.Pie(
            labels=lang_labels,
            values=lang_values,
            textinfo='label+percent',
            textposition='auto',
            marker_colors=px.colors.qualitative.Set3
        )])
        lang_fig.update_layout(
            title="Relevant Chunks by Language",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )
    else:
        lang_fig = None
    
    # Create database distribution bar chart - RELEVANT ONLY
    db_labels = data.get('top_databases_relevant', data['top_databases'])['labels']
    db_values = data.get('top_databases_relevant', data['top_databases'])['values']
    
    if db_labels:
        # Map database names for display
        db_display_labels = [DATABASE_DISPLAY_MAP.get(db, db) for db in db_labels]
        
        # Calculate total for percentages
        total_relevant = sum(db_values)
        db_percentages = [round((v / total_relevant * 100), 1) if total_relevant > 0 else 0 for v in db_values]
        
        # Use database-specific colors with consistent ordering
        db_colors = [get_database_color(db) for db in db_labels]
        
        # Create text annotations with both count and percentage
        text_annotations = [f"{v:,}<br>({p}%)" for v, p in zip(db_values, db_percentages)]
        
        # Create hover text with both values
        hover_text = [f"Database: {db}<br>Count: {v:,}<br>Percentage: {p}%" 
                     for db, v, p in zip(db_display_labels, db_values, db_percentages)]
        
        db_fig = go.Figure(data=[go.Bar(
            x=db_display_labels,
            y=db_values,
            marker_color=db_colors,
            text=text_annotations,
            textposition='auto',
            hovertext=hover_text,
            hoverinfo='text'
        )])
        db_fig.update_layout(
            title="Top Databases by Relevant Chunk Count",
            xaxis_title="Database",
            yaxis_title="Number of Chunks",
            height=350,
            margin=dict(l=40, r=20, t=40, b=100),
            xaxis_tickangle=-45
        )
    else:
        db_fig = None
    
    # Create statistics panel
    stats_panel = dbc.Card([
        dbc.CardBody([
            html.H5("📊 Chunk Statistics", className="mb-3"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.P("📄 Total Chunks", className="text-muted mb-1"),
                    html.H4(f"{total_chunks:,}", style={"color": "black"})
                ], md=3),
                dbc.Col([
                    html.P("✅ Relevant Chunks", className="text-muted mb-1"),
                    html.H4(f"{relevant_chunks:,}", style={"color": px.colors.qualitative.Set2[0]})  # Same green as relevant wedge
                ], md=3),
                dbc.Col([
                    html.P("📈 Relevance Rate", className="text-muted mb-1"),
                    html.H4(f"{relevance_rate}%", className="text-info")
                ], md=3),
                dbc.Col([
                    html.P("📏 Avg per Relevant Chunk", className="text-muted mb-1"),
                    html.H4([
                        f"{avg_chunk_length:,} chars",
                        html.Br(),
                        html.Span(f"(~{avg_chunk_length // 5:,} words)", style={"fontSize": "0.8em"})
                    ], className="text-warning")
                ], md=3),
            ])
        ])
    ], className="mb-4")
    
    # Build layout
    layout_components = [stats_panel]
    
    if lang_fig or db_fig:
        chart_row = dbc.Row([])
        if relevance_fig:
            chart_row.children.append(
                dbc.Col([
                    dcc.Graph(figure=relevance_fig, config={'displayModeBar': False})
                ], md=6)
            )
        if lang_fig:
            chart_row.children.append(
                dbc.Col([
                    dcc.Graph(figure=lang_fig, config={'displayModeBar': False})
                ], md=6)
            )
        layout_components.append(chart_row)
        
        if db_fig:
            layout_components.append(
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=db_fig, config={'displayModeBar': False})
                    ], md=12)
                ], className="mt-3")
            )
    
    # Add timeline section
    layout_components.append(
        html.Hr(className="mt-5 mb-4")
    )
    
    # Create timeline for relevant chunks
    time_series_data = data.get('time_series_relevant', {})
    logging.info(f"Chunks time series data: {time_series_data}")
    
    if not time_series_data or not time_series_data.get('dates'):
        # If no time series data, show a message
        layout_components.append(
            dbc.Row([
                dbc.Col([
                    html.P("Timeline data not available", className="text-muted text-center")
                ], md=12)
            ])
        )
    else:
        timeline_fig = go.Figure()
        timeline_fig.add_trace(go.Scatter(
            x=time_series_data['dates'],
            y=time_series_data['values'],
            mode='lines+markers',
            name='Relevant Chunks',
            line=dict(color=THEME_COLORS['western'], width=2),
            marker=dict(size=6)
        ))
        timeline_fig.update_layout(
            title="Relevant Chunks Over Time",
            xaxis_title="Date",
            yaxis_title="Number of Chunks",
            height=300,
            margin=dict(l=60, r=20, t=60, b=60),
            hovermode='x unified'
        )
        
        layout_components.append(
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=timeline_fig, config={'displayModeBar': False})
                ], md=12)
            ])
        )
    
    # Add per-database relevance donuts
    db_relevance_data = data.get('per_database_relevance', {})
    logging.info(f"Chunks database relevance data: {list(db_relevance_data.keys()) if db_relevance_data else 'None'}")
    
    if not db_relevance_data:
        # If no database relevance data, show a message
        layout_components.append(
            html.Div([
                html.H5("Database Relevance Breakdown", className="text-center mt-4 mb-3"),
                html.P("No database relevance data available", className="text-muted text-center")
            ])
        )
    else:
        # Create small donuts for each database
        donut_rows = []
        databases = list(db_relevance_data.keys())[:12]  # Limit to 12 databases
        
        for i in range(0, len(databases), 4):  # 4 donuts per row
            row_databases = databases[i:i+4]
            donut_cols = []
            
            for db in row_databases:
                db_data = db_relevance_data[db]
                relevant = db_data.get('relevant', 0)
                irrelevant = db_data.get('irrelevant', 0)
                total = relevant + irrelevant
                
                if total > 0:
                    # Create small donut for this database
                    donut_fig = go.Figure(data=[go.Pie(
                        labels=['Relevant', 'Irrelevant'],
                        values=[relevant, irrelevant],
                        hole=0.5,
                        marker_colors=[THEME_COLORS['western'], '#e0e0e0'],
                        textinfo='percent',
                        textposition='inside',
                        insidetextorientation='radial'
                    )])
                    
                    # Format the total with thousands separator
                    total_formatted = f"{total:,}"
                    
                    # Get database color with consistent ordering
                    db_color = get_database_color(db)
                    
                    donut_fig.update_layout(
                        title=dict(
                            text=f"<b>{DATABASE_DISPLAY_MAP.get(db, db)}</b>",
                            font=dict(size=16, color=db_color),
                            x=0.5,
                            xanchor='center',
                            y=0.95,  # Position title closer to the donut
                            yanchor='top'
                        ),
                        height=300,  # Much larger height for visibility
                        showlegend=False,
                        margin=dict(l=5, r=5, t=50, b=5),  # Enough margin for full title visibility
                        annotations=[
                            dict(
                                text=f'{total_formatted}<br>chunks',
                                x=0.5, y=0.5,
                                font_size=16,
                                showarrow=False
                            )
                        ]
                    )
                    
                    donut_fig.update_traces(
                        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>'
                    )
                    
                    donut_cols.append(
                        dbc.Col([
                            dcc.Graph(figure=donut_fig, config={'displayModeBar': False}, style={'marginBottom': '-20px', 'marginTop': '-10px'})
                        ], md=3, sm=6)
                    )
            
            if donut_cols:
                donut_rows.append(dbc.Row(donut_cols, className="mb-0"))
        
        if donut_rows:
            layout_components.append(
                html.Div([
                    html.H5("Relevance Breakdown by Database", className="text-center mt-4 mb-3"),
                    *donut_rows
                ])
            )
    
    return html.Div(layout_components)


def create_taxonomy_visualizations(data: Dict) -> html.Div:
    """
    Create visualizations for the Taxonomy subtab.
    
    Args:
        data: Taxonomy data dictionary from fetch_taxonomy_combinations
        
    Returns:
        html.Div: Container with visualizations
    """
    if not data or data.get('total_chunks', 0) == 0:
        return html.Div([
            html.P("No taxonomy data found with the current filters.", className="text-muted text-center mt-5")
        ])
    
    # Extract data
    chunks_with_taxonomy = data['chunks_with_taxonomy']
    taxonomy_coverage = data['taxonomy_coverage']
    avg_taxonomies = data['avg_taxonomies_per_chunk']
    total_chunks = data['total_chunks']
    
    # Create distribution bar chart - EXCLUDE chunks with 0 taxonomies
    labels = data['combinations_per_chunk']['labels']
    values = data['combinations_per_chunk']['values']
    percentages = data['combinations_per_chunk']['percentages']
    
    if labels and values:
        # Filter out the '0' category to show only chunks with taxonomies
        filtered_data = [(l, v, p) for l, v, p in zip(labels, values, percentages) if l != '0']
        
        if filtered_data:
            filtered_labels, filtered_values, filtered_percentages = zip(*filtered_data)
            
            # Create text for hover showing both count and percentage
            hover_text = [f"Count: {v:,}<br>Percentage: {p}%" for v, p in zip(filtered_values, filtered_percentages)]
            
            # Use distinctive colors for different taxonomy counts
            tax_colors = px.colors.qualitative.T10
            color_map = {str(i): tax_colors[(i-1) % len(tax_colors)] for i in range(1, 6)}
            color_map['5+'] = tax_colors[4 % len(tax_colors)]
            
            # Create text annotations with both count and percentage
            text_annotations = [f"{v:,} ({p}%)" for v, p in zip(filtered_values, filtered_percentages)]
            
            dist_fig = go.Figure(data=[go.Bar(
                x=filtered_labels,
                y=filtered_values,
                marker_color=[color_map.get(l, '#cccccc') for l in filtered_labels],
                text=text_annotations,
                textposition='auto',
                customdata=hover_text
            )])
            dist_fig.update_layout(
                title="Taxonomy Assignments per Chunk (Relevant Only)",
                xaxis_title="Number of Taxonomies",
                yaxis_title="Number of Chunks",
                height=400,
                margin=dict(l=60, r=20, t=60, b=60)
            )
            dist_fig.update_traces(
                hovertemplate='<b>%{x} taxonomies</b><br>%{customdata}<extra></extra>'
            )
        else:
            dist_fig = None
    else:
        dist_fig = None
    
    # Create coverage gauge chart
    coverage_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=taxonomy_coverage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Taxonomy Coverage Rate (%)"},
        delta={'reference': 100, 'relative': False},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': THEME_COLORS['western']},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': hex_to_rgba(THEME_COLORS['ZH'], 0.3)},
                {'range': [50, 80], 'color': hex_to_rgba(THEME_COLORS['western'], 0.3)},
                {'range': [80, 100], 'color': hex_to_rgba(THEME_COLORS['russian'], 0.3)}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    coverage_fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Create statistics panel
    stats_panel = dbc.Card([
        dbc.CardBody([
            html.H5("📊 Taxonomy Statistics", className="mb-3"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.P("📄 Total Chunks", className="text-muted mb-1"),
                    html.H4(f"{total_chunks:,}", className="text-primary")
                ], md=3),
                dbc.Col([
                    html.P("🏷️ Chunks w/ Taxonomy", className="text-muted mb-1"),
                    html.H4(f"{chunks_with_taxonomy:,}", className="text-success")
                ], md=3),
                dbc.Col([
                    html.P("📈 Coverage Rate", className="text-muted mb-1"),
                    html.H4(f"{taxonomy_coverage}%", className="text-info")
                ], md=3),
                dbc.Col([
                    html.P("📊 Avg per Chunk", className="text-muted mb-1"),
                    html.H4(f"{avg_taxonomies:.2f}", className="text-warning")
                ], md=3),
            ])
        ])
    ], className="mb-4")
    
    # Build layout
    layout_components = [stats_panel]
    
    if dist_fig:
        layout_components.append(
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=dist_fig, config={'displayModeBar': False})
                ], md=8),
                dbc.Col([
                    dcc.Graph(figure=coverage_fig, config={'displayModeBar': False})
                ], md=4)
            ])
        )
    
    # Add timeline section
    layout_components.append(
        html.Hr(className="mt-5 mb-4")
    )
    
    # Create timeline for taxonomy assignments
    time_series_data = data.get('time_series_relevant', {})
    logging.info(f"Taxonomy time series data: {time_series_data}")
    
    if not time_series_data or not time_series_data.get('dates'):
        # If no time series data, show a message
        layout_components.append(
            dbc.Row([
                dbc.Col([
                    html.P("Timeline data not available", className="text-muted text-center")
                ], md=12)
            ])
        )
    else:
        timeline_fig = go.Figure()
        timeline_fig.add_trace(go.Scatter(
            x=time_series_data['dates'],
            y=time_series_data['values'],
            mode='lines+markers',
            name='Taxonomy Assignments',
            line=dict(color=THEME_COLORS['western'], width=2),
            marker=dict(size=6)
        ))
        timeline_fig.update_layout(
            title="Taxonomy Assignments Over Time",
            xaxis_title="Date",
            yaxis_title="Number of Assignments",
            height=300,
            margin=dict(l=60, r=20, t=60, b=60),
            hovermode='x unified'
        )
        
        layout_components.append(
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=timeline_fig, config={'displayModeBar': False})
                ], md=12)
            ])
        )
    
    # Add per-database relevance donuts
    db_relevance_data = data.get('per_database_relevance', {})
    logging.info(f"Taxonomy database relevance data: {list(db_relevance_data.keys()) if db_relevance_data else 'None'}")
    
    if not db_relevance_data:
        # If no database relevance data, show a message
        layout_components.append(
            html.Div([
                html.H5("Database Coverage Breakdown", className="text-center mt-4 mb-3"),
                html.P("No database coverage data available", className="text-muted text-center")
            ])
        )
    else:
        # Create small donuts for each database
        donut_rows = []
        databases = list(db_relevance_data.keys())[:12]  # Limit to 12 databases
        
        for i in range(0, len(databases), 4):  # 4 donuts per row
            row_databases = databases[i:i+4]
            donut_cols = []
            
            for db in row_databases:
                db_data = db_relevance_data[db]
                with_taxonomy = db_data.get('relevant', 0)
                without_taxonomy = db_data.get('irrelevant', 0)
                total = with_taxonomy + without_taxonomy
                
                if total > 0:
                    # Create small donut for this database
                    donut_fig = go.Figure(data=[go.Pie(
                        labels=['With Taxonomy', 'Without'],
                        values=[with_taxonomy, without_taxonomy],
                        hole=0.5,
                        marker_colors=[THEME_COLORS['western'], '#e0e0e0'],
                        textinfo='percent',
                        textposition='inside',
                        insidetextorientation='radial'
                    )])
                    
                    # Format the total with thousands separator
                    total_formatted = f"{total:,}"
                    
                    # Get database color with consistent ordering
                    db_color = get_database_color(db)
                    
                    donut_fig.update_layout(
                        title=dict(
                            text=f"<b>{DATABASE_DISPLAY_MAP.get(db, db)}</b>",
                            font=dict(size=16, color=db_color),
                            x=0.5,
                            xanchor='center',
                            y=0.95,  # Position title closer to the donut
                            yanchor='top'
                        ),
                        height=300,  # Much larger height for visibility
                        showlegend=False,
                        margin=dict(l=5, r=5, t=50, b=5),  # Enough margin for full title visibility
                        annotations=[
                            dict(
                                text=f'{total_formatted}<br>chunks',
                                x=0.5, y=0.5,
                                font_size=16,
                                showarrow=False
                            )
                        ]
                    )
                    
                    donut_fig.update_traces(
                        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>'
                    )
                    
                    donut_cols.append(
                        dbc.Col([
                            dcc.Graph(figure=donut_fig, config={'displayModeBar': False}, style={'marginBottom': '-20px', 'marginTop': '-10px'})
                        ], md=3, sm=6)
                    )
            
            if donut_cols:
                donut_rows.append(dbc.Row(donut_cols, className="mb-0"))
        
        if donut_rows:
            layout_components.append(
                html.Div([
                    html.H5("Relevance Breakdown by Database", className="text-center mt-4 mb-3"),
                    *donut_rows
                ])
            )
    
    return html.Div(layout_components)


def create_keywords_visualizations(data: Dict) -> html.Div:
    """
    Create visualizations for the Keywords subtab.
    
    Args:
        data: Keywords data dictionary from fetch_keywords_data
        
    Returns:
        html.Div: Container with visualizations
    """
    if not data or data.get('total_unique_keywords', 0) == 0:
        return html.Div([
            html.P("No keywords found with the current filters.", className="text-muted text-center mt-5")
        ])
    
    # Extract data - using correct field names based on the documentation
    total_unique = data.get('total_unique_keywords', 0)
    total_occurrences = data.get('total_keyword_occurrences', 0)
    chunks_with_keywords = data.get('chunks_with_keywords', 0)
    keyword_coverage = data.get('keyword_coverage', 0)
    avg_per_chunk = data.get('avg_keywords_per_chunk', 0)
    
    # Create top keywords horizontal bar chart
    top_keywords = data.get('top_keywords', {})
    kw_labels = top_keywords.get('labels', [])
    kw_values = top_keywords.get('values', [])
    
    if kw_labels and kw_values:
        # Reverse for horizontal display (highest at top)
        kw_labels_rev = kw_labels[::-1]
        kw_values_rev = kw_values[::-1]
        
        # Calculate total for percentages
        total_kw_occurrences = sum(kw_values)
        kw_percentages_rev = [round((v / total_kw_occurrences * 100), 1) if total_kw_occurrences > 0 else 0 for v in kw_values_rev]
        
        # Use distinctive colors for keywords
        kw_colors = px.colors.qualitative.Plotly if len(kw_labels_rev) <= 10 else px.colors.qualitative.Dark24
        
        # Create text annotations with both count and percentage
        text_annotations = [f"{v:,} ({p}%)" for v, p in zip(kw_values_rev, kw_percentages_rev)]
        
        # Create hover text with both values
        hover_text = [f"Keyword: {kw}<br>Occurrences: {v:,}<br>Percentage: {p}%" 
                     for kw, v, p in zip(kw_labels_rev, kw_values_rev, kw_percentages_rev)]
        
        keywords_fig = go.Figure(data=[go.Bar(
            x=kw_values_rev,
            y=kw_labels_rev,
            orientation='h',
            marker_color=kw_colors[:len(kw_labels_rev)],
            text=text_annotations,
            textposition='auto',
            hovertext=hover_text,
            hoverinfo='text'
        )])
        keywords_fig.update_layout(
            title="Top 20 Keywords by Frequency",
            xaxis_title="Occurrences",
            yaxis_title="Keyword",
            height=max(500, len(kw_labels) * 25),
            margin=dict(l=200, r=20, t=60, b=40)
        )
    else:
        keywords_fig = None
    
    # Create statistics panel
    stats_panel = dbc.Card([
        dbc.CardBody([
            html.H5("📊 Keywords Statistics", className="mb-3"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.P("🔤 Unique Keywords", className="text-muted mb-1"),
                    html.H4(f"{total_unique:,}", className="text-primary")
                ], md=3),
                dbc.Col([
                    html.P("📝 Total Occurrences", className="text-muted mb-1"),
                    html.H4(f"{total_occurrences:,}", className="text-success")
                ], md=3),
                dbc.Col([
                    html.P("📈 Coverage Rate", className="text-muted mb-1"),
                    html.H4(f"{keyword_coverage}%", className="text-info")
                ], md=3),
                dbc.Col([
                    html.P("📊 Avg per Chunk", className="text-muted mb-1"),
                    html.H4(f"{avg_per_chunk:.1f}", className="text-warning")
                ], md=3),
            ]),
            html.Hr(),
            html.P(f"📄 Chunks with keywords: {chunks_with_keywords:,}", className="text-muted mb-0")
        ])
    ], className="mb-4")
    
    # Build layout
    layout_components = [stats_panel]
    
    if keywords_fig:
        layout_components.append(
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=keywords_fig, config={'displayModeBar': False})
                ], md=12)
            ])
        )
    
    return html.Div(layout_components)


def create_entities_visualizations(data: Dict) -> html.Div:
    """
    Create visualizations for the Named Entities subtab.
    
    Args:
        data: Named entities data dictionary from fetch_named_entities_data
        
    Returns:
        html.Div: Container with visualizations
    """
    if not data or data.get('total_unique_entities', 0) == 0:
        return html.Div([
            html.P("No named entities found with the current filters.", className="text-muted text-center mt-5")
        ])
    
    # Extract data - using correct field names based on the documentation
    total_unique = data.get('total_unique_entities', 0)
    total_occurrences = data.get('total_entity_occurrences', 0)
    
    # Create entity type distribution pie chart
    entity_types = data.get('entity_types', {})
    type_labels = entity_types.get('labels', [])
    type_counts = entity_types.get('counts', [])
    
    if type_labels and type_counts:
        type_fig = go.Figure(data=[go.Pie(
            labels=type_labels,
            values=type_counts,
            textinfo='label+percent',
            textposition='auto',
            marker_colors=px.colors.qualitative.Set2
        )])
        type_fig.update_layout(
            title="Entity Types Distribution",
            height=400,
            margin=dict(l=20, r=20, t=60, b=20)
        )
    else:
        type_fig = None
    
    # Get top entities by type for ORG, LOC, and PER
    top_entities_by_type = data.get('top_entities_by_type', {})
    
    # Create separate bar charts for ORG, LOC, and PER
    entity_charts = []
    
    # Define colors for each entity type
    entity_type_colors = {
        'ORG': px.colors.qualitative.Plotly[0],  # Blue
        'LOC': px.colors.qualitative.Plotly[1],  # Red
        'GPE': px.colors.qualitative.Plotly[1],  # Red (same as LOC)
        'PER': px.colors.qualitative.Plotly[2],  # Green
        'PERSON': px.colors.qualitative.Plotly[2]  # Green (same as PER)
    }
    
    # Process ORG entities
    if 'ORG' in top_entities_by_type and top_entities_by_type['ORG']:
        org_data = top_entities_by_type['ORG']
        org_labels = [item['entity'] for item in org_data[:15]]
        org_values = [item['count'] for item in org_data[:15]]
        
        # Calculate total for percentages
        total_org = sum(org_values)
        org_percentages = [round((v / total_org * 100), 1) if total_org > 0 else 0 for v in org_values]
        
        # Create text annotations with both count and percentage
        text_annotations = [f"{v:,}<br>({p}%)" for v, p in zip(org_values, org_percentages)]
        
        # Create hover text with both values
        hover_text = [f"Organization: {org}<br>Count: {v:,}<br>Percentage: {p}%" 
                     for org, v, p in zip(org_labels, org_values, org_percentages)]
        
        org_fig = go.Figure(data=[go.Bar(
            x=org_labels,
            y=org_values,
            marker_color=entity_type_colors['ORG'],
            text=text_annotations,
            textposition='auto',
            hovertext=hover_text,
            hoverinfo='text'
        )])
        org_fig.update_layout(
            title="Top 15 Organizations (ORG)",
            xaxis_title="Organization",
            yaxis_title="Occurrences",
            height=350,
            margin=dict(l=60, r=20, t=60, b=100),
            xaxis_tickangle=-45
        )
        entity_charts.append(('ORG', org_fig))
    
    # Process LOC/GPE entities (locations)
    loc_key = 'GPE' if 'GPE' in top_entities_by_type else 'LOC'
    if loc_key in top_entities_by_type and top_entities_by_type[loc_key]:
        loc_data = top_entities_by_type[loc_key]
        loc_labels = [item['entity'] for item in loc_data[:15]]
        loc_values = [item['count'] for item in loc_data[:15]]
        
        # Calculate total for percentages
        total_loc = sum(loc_values)
        loc_percentages = [round((v / total_loc * 100), 1) if total_loc > 0 else 0 for v in loc_values]
        
        # Create text annotations with both count and percentage
        text_annotations = [f"{v:,}<br>({p}%)" for v, p in zip(loc_values, loc_percentages)]
        
        # Create hover text with both values
        hover_text = [f"Location: {loc}<br>Count: {v:,}<br>Percentage: {p}%" 
                     for loc, v, p in zip(loc_labels, loc_values, loc_percentages)]
        
        loc_fig = go.Figure(data=[go.Bar(
            x=loc_labels,
            y=loc_values,
            marker_color=entity_type_colors['LOC'],
            text=text_annotations,
            textposition='auto',
            hovertext=hover_text,
            hoverinfo='text'
        )])
        loc_fig.update_layout(
            title="Top 15 Locations (GPE/LOC)",
            xaxis_title="Location",
            yaxis_title="Occurrences",
            height=350,
            margin=dict(l=60, r=20, t=60, b=100),
            xaxis_tickangle=-45
        )
        entity_charts.append(('LOC', loc_fig))
    
    # Process PER/PERSON entities
    per_key = 'PERSON' if 'PERSON' in top_entities_by_type else 'PER'
    if per_key in top_entities_by_type and top_entities_by_type[per_key]:
        per_data = top_entities_by_type[per_key]
        per_labels = [item['entity'] for item in per_data[:15]]
        per_values = [item['count'] for item in per_data[:15]]
        
        # Calculate total for percentages
        total_per = sum(per_values)
        per_percentages = [round((v / total_per * 100), 1) if total_per > 0 else 0 for v in per_values]
        
        # Create text annotations with both count and percentage
        text_annotations = [f"{v:,}<br>({p}%)" for v, p in zip(per_values, per_percentages)]
        
        # Create hover text with both values
        hover_text = [f"Person: {per}<br>Count: {v:,}<br>Percentage: {p}%" 
                     for per, v, p in zip(per_labels, per_values, per_percentages)]
        
        per_fig = go.Figure(data=[go.Bar(
            x=per_labels,
            y=per_values,
            marker_color=entity_type_colors['PER'],
            text=text_annotations,
            textposition='auto',
            hovertext=hover_text,
            hoverinfo='text'
        )])
        per_fig.update_layout(
            title="Top 15 People (PERSON)",
            xaxis_title="Person",
            yaxis_title="Occurrences",
            height=350,
            margin=dict(l=60, r=20, t=60, b=100),
            xaxis_tickangle=-45
        )
        entity_charts.append(('PER', per_fig))
    
    # Create statistics panel
    stats_panel = dbc.Card([
        dbc.CardBody([
            html.H5("📊 Named Entities Statistics", className="mb-3"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.P("👤 Unique Entities", className="text-muted mb-1"),
                    html.H4(f"{total_unique:,}", className="text-primary")
                ], md=6),
                dbc.Col([
                    html.P("📝 Total Occurrences", className="text-muted mb-1"),
                    html.H4(f"{total_occurrences:,}", className="text-success")
                ], md=6),
            ]),
            html.Hr(),
            html.P("💡 Entity co-occurrence analysis available in the Explore tab", 
                   className="text-info mb-0", style={'font-style': 'italic'})
        ])
    ], className="mb-4")
    
    # Build layout
    layout_components = [stats_panel]
    
    # Add entity type distribution chart
    if type_fig:
        layout_components.append(
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=type_fig, config={'displayModeBar': False})
                ], md=12)
            ], className="mb-4")
        )
    
    # Add entity charts (ORG, LOC, PER) in a grid
    if entity_charts:
        # Create rows of charts
        for i in range(0, len(entity_charts), 2):
            row = dbc.Row([])
            for j in range(2):
                if i + j < len(entity_charts):
                    _, fig = entity_charts[i + j]
                    row.children.append(
                        dbc.Col([
                            dcc.Graph(figure=fig, config={'displayModeBar': False})
                        ], md=6)
                    )
            layout_components.append(row)
    
    return html.Div(layout_components)


def register_sources_tab_callbacks(app):
    """
    Register callbacks for the Sources tab - WORKING VERSION.
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
    
    # Clientside callback to show loading and hide content when Apply button is clicked
    app.clientside_callback(
        """
        function(n_clicks) {
            if (n_clicks && n_clicks > 0) {
                return [
                    {'display': 'flex'},  // Show loading
                    {'display': 'none'}   // Hide content
                ];
            }
            return [
                {'display': 'none'},   // Hide loading  
                {'display': 'none'}    // Hide content
            ];
        }
        """,
        [Output("sources-loading-messages", "style", allow_duplicate=True),
         Output("sources-results-tabs", "style", allow_duplicate=True)],
        [Input("sources-apply-button", "n_clicks")],
        prevent_initial_call=True
    )
    
    # Auto-load on tab visit
    app.clientside_callback(
        """
        function(active_tab) {
            if (active_tab === "tab-sources") {
                // Force initialize fun facts multiple times to ensure it works
                setTimeout(() => {
                    if (window.initializeSourcesFunFacts) {
                        window.initializeSourcesFunFacts();
                    }
                }, 100);
                
                setTimeout(() => {
                    if (window.initializeSourcesFunFacts) {
                        window.initializeSourcesFunFacts();
                    }
                }, 500);
                
                setTimeout(() => {
                    if (window.initializeSourcesFunFacts) {
                        window.initializeSourcesFunFacts();
                    }
                }, 1000);
                
                return [
                    {'display': 'flex'},  // Show loading
                    {'display': 'none'}   // Hide content
                ];
            }
            return [
                {'display': 'none'},   // Hide loading
                {'display': 'none'}    // Hide content  
            ];
        }
        """,
        [Output("sources-loading-messages", "style", allow_duplicate=True),
         Output("sources-results-tabs", "style", allow_duplicate=True)],
        [Input("tabs", "value")],
        prevent_initial_call=True
    )
    
    # Main callback for updating sources tab content
    @app.callback(
        [
            Output("sources-stats-container", "children"),
            Output("sources-content-display", "children"),
            Output("sources-loading-messages", "style"),
            Output("sources-results-tabs", "style"),
        ],
        [
            Input("sources-apply-button", "n_clicks"),
            Input("tabs", "value")  # Auto-load when tab is visited
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
        """
        logging.info(f"Sources tab callback triggered - n_clicks: {n_clicks}, active_tab: {active_tab}")
        
        # Check if Sources tab is active
        if active_tab != "tab-sources":
            logging.info("Not on Sources tab, skipping update")
            return dash.no_update, dash.no_update, {'display': 'none'}, {'display': 'none'}
        
        logging.info("Sources tab is active, proceeding with data fetch")
        
        # Clear cache on manual apply button click
        if n_clicks and n_clicks > 0:
            logging.info("Manual apply button clicked, clearing cache")
            clear_cache()
        
        try:
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
            try:
                corpus_stats = fetch_corpus_stats()
            except Exception as e:
                logging.error(f"Error fetching corpus stats: {e}")
                corpus_stats = {
                    "docs_count": "Loading...",
                    "docs_rel_count": "Loading...",
                    "chunks_count": "Loading...",
                    "chunks_rel_count": "Loading...",
                    "tax_levels": "Loading...",
                    "items_count": "Loading..."
                }
            
            stats_html = html.Div([
                html.P(f"Docs: {corpus_stats['docs_count']:,} ({corpus_stats['docs_rel_count']:,} rel) | Chunks: {corpus_stats['chunks_count']:,} ({corpus_stats['chunks_rel_count']:,} rel) | Tax: {corpus_stats['tax_levels']:,} levels | Items: {corpus_stats['items_count']:,}") if isinstance(corpus_stats['docs_count'], int) else html.P("Loading corpus statistics..."),
                html.P(f"Filters: {' | '.join(filter_desc) if filter_desc else 'None'}", className="text-muted")
            ])
            
            # Fetch filtered data with timeout protection
            logging.info(f"Starting to fetch data for Sources tab with filters")
            
            try:
                documents_data = fetch_documents_data(lang_val, db_val, source_type, date_range)
                logging.info("Documents data fetched")
            except Exception as e:
                logging.error(f"Error fetching documents data: {e}")
                documents_data = {"error": str(e)}
            
            try:
                chunks_data = fetch_chunks_data(lang_val, db_val, source_type, date_range)
                logging.info("Chunks data fetched")
            except Exception as e:
                logging.error(f"Error fetching chunks data: {e}")
                chunks_data = {"error": str(e)}
            
            try:
                taxonomy_data = fetch_taxonomy_combinations(lang_val, db_val, source_type, date_range)
                logging.info("Taxonomy data fetched")
            except Exception as e:
                logging.error(f"Error fetching taxonomy data: {e}")
                taxonomy_data = {"error": str(e)}
            
            try:
                keywords_data = fetch_keywords_data(lang_val, db_val, source_type, date_range)
                logging.info("Keywords data fetched")
            except Exception as e:
                logging.error(f"Error fetching keywords data: {e}")
                keywords_data = {"error": str(e)}
            
            try:
                named_entities_data = fetch_named_entities_data(lang_val, db_val, source_type, date_range)
                logging.info("Named entities data fetched")
            except Exception as e:
                logging.error(f"Error fetching named entities data: {e}")
                named_entities_data = {"error": str(e)}
            
            # Create visualizations for each subtab
            documents_viz = create_documents_visualizations(documents_data)
            chunks_viz = create_chunks_visualizations(chunks_data)
            taxonomy_viz = create_taxonomy_visualizations(taxonomy_data)
            keywords_viz = create_keywords_visualizations(keywords_data)
            entities_viz = create_entities_visualizations(named_entities_data)
            
            updated_tabs_content = dcc.Tabs([
                dcc.Tab(label="Documents", children=documents_viz),
                dcc.Tab(label="Chunks", children=chunks_viz),
                dcc.Tab(label="Taxonomy", children=taxonomy_viz),
                dcc.Tab(label="Keywords", children=keywords_viz),
                dcc.Tab(label="Named Entities", children=entities_viz)
            ], id="sources-subtabs", className="custom-tabs")
            
            # Return updated content, hide loading, show content
            logging.info("Sources tab data ready, returning content")
            
            # Add a client-side callback to explicitly stop the radar sweep
            updated_tabs_with_callback = html.Div([
                updated_tabs_content,
                html.Script("""
                    // Force hide radar sweep when Sources tab content is loaded
                    setTimeout(function() {
                        if (window.dashLoadingMonitor && window.dashLoadingMonitor.forceHide) {
                            window.dashLoadingMonitor.forceHide();
                            console.log('[Sources Tab] Forced radar sweep hide');
                        }
                    }, 100);
                """)
            ])
            
            return stats_html, updated_tabs_with_callback, {'display': 'none'}, {'display': 'block'}
            
        except Exception as e:
            logging.error(f"CRITICAL ERROR in Sources tab callback: {e}")
            error_content = html.Div([
                html.H5("Error Loading Sources Data", className="text-danger"),
                html.P(f"Error: {str(e)}"),
                html.P("Please try refreshing the page or contact support.")
            ], className="alert alert-danger")
            # Always hide loading and show content (error message) even on error
            return error_content, error_content, {'display': 'none'}, {'display': 'block'}
    
