#!/usr/bin/env python
# coding: utf-8

"""
Fixed version of the sources.py file with proper indentation and variable references.
Replace your entire sources.py file with this version.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import html, dcc, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import sys
import os

# Add the project root to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DEFAULT_START_DATE, DEFAULT_END_DATE, THEME_COLORS
from database.data_fetchers import fetch_all_databases
from components.layout import create_filter_card
from components.cards import create_stats_card
# Import the database fetchers directly
from database.data_fetchers_sources import (
    fetch_corpus_stats,
    fetch_taxonomy_combinations,
    fetch_chunks_data,
    fetch_documents_data,
    fetch_time_series_data,
    fetch_language_time_series,
    fetch_database_time_series,
    fetch_keywords_data,
    fetch_named_entities_data,
    fetch_database_breakdown
)


def create_pie_chart(
    inner_value: int,
    inner_label: str,
    values: List[int],
    labels: List[str],
    colors: List[str] = None,
    title: str = ""
):
    """
    Create a donut chart for displaying data.
    
    Args:
        inner_value: Value to display in the center
        inner_label: Label for the center value
        values: Values for the pie chart
        labels: Labels for the pie chart
        colors: Colors for the pie chart, if None uses language colors from config
        title: Chart title
        
    Returns:
        go.Figure: Plotly Figure object
    """
    # If colors not provided, use language colors or default colors
    if colors is None:
        # If labels are language codes, use language colors
        colors = []
        for label in labels:
            if label in THEME_COLORS:
                colors.append(THEME_COLORS[label])
            else:
                # Use a suitable default color from px.colors
                colors.append(px.colors.qualitative.Pastel[len(colors) % len(px.colors.qualitative.Pastel)])
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.6,
        textinfo='percent',
        textposition='inside',
        marker_colors=colors,
        hoverinfo='label+percent+value',
        hovertemplate='<b>%{label}</b><br>Value: %{value:,}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=title,
        annotations=[dict(
            text=f'<b>{inner_label}</b><br>{inner_value:,}',
            x=0.5, y=0.5,
            font_size=14,
            showarrow=False
        )],
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        height=400,
        margin=dict(l=20, r=20, t=40, b=100)
    )
    
    return fig


def create_time_series_chart(
    x_values: List[datetime],
    y_values: List[int],
    title: str = "",
    x_title: str = "Date",
    y_title: str = "Count",
    color: str = "#1F77B4"
):
    """
    Create a time series chart.
    
    Args:
        x_values: X-axis values (dates)
        y_values: Y-axis values (counts)
        title: Chart title
        x_title: X-axis title
        y_title: Y-axis title
        color: Line color
        
    Returns:
        go.Figure: Plotly Figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='lines+markers',
        line=dict(color=color, width=2),
        marker=dict(size=6),
        hovertemplate='%{x|%Y-%m-%d}<br>Count: %{y:,}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=300,
        margin=dict(l=20, r=20, t=40, b=40),
        showlegend=False
    )
    
    return fig


def create_bar_chart(
    x_values: List[str],
    y_values: List[float],
    title: str = "",
    x_title: str = "",
    y_title: str = "Percentage (%)",
    color: Union[str, List[str]] = "#1F77B4",
    orientation: str = "v"
):
    """
    Create a bar chart.
    
    Args:
        x_values: X-axis values
        y_values: Y-axis values
        title: Chart title
        x_title: X-axis title
        y_title: Y-axis title
        color: Bar color or list of colors for each bar
        orientation: Bar orientation ('v' for vertical, 'h' for horizontal)
        
    Returns:
        go.Figure: Plotly Figure object
    """
    # Check if color is a string (single color) or a list (multiple colors)
    use_single_color = isinstance(color, str)
    
    if orientation == 'h':
        if use_single_color:
            fig = go.Figure(go.Bar(
                y=x_values,
                x=y_values,
                orientation='h',
                marker_color=color,
                text=y_values,
                texttemplate='%{text:.1f}%',
                textposition='inside',
                hovertemplate='%{y}<br>%{x:.1f}%<extra></extra>'
            ))
        else:
            # Use individual colors for each bar
            fig = go.Figure(go.Bar(
                y=x_values,
                x=y_values,
                orientation='h',
                marker_color=color[:len(x_values)],  # Limit colors to number of bars
                text=y_values,
                texttemplate='%{text:.1f}%',
                textposition='inside',
                hovertemplate='%{y}<br>%{x:.1f}%<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            yaxis_title=x_title,
            xaxis_title=y_title,
            height=300,
            margin=dict(l=20, r=20, t=40, b=40)
        )
    else:
        if use_single_color:
            fig = go.Figure(go.Bar(
                x=x_values,
                y=y_values,
                marker_color=color,
                text=y_values,
                texttemplate='%{text:.1f}%',
                textposition='auto',
                hovertemplate='%{x}<br>%{y:.1f}%<extra></extra>'
            ))
        else:
            # Use individual colors for each bar
            fig = go.Figure(go.Bar(
                x=x_values,
                y=y_values,
                marker_color=color[:len(x_values)],  # Limit colors to number of bars
                text=y_values,
                texttemplate='%{text:.1f}%',
                textposition='auto',
                hovertemplate='%{x}<br>%{y:.1f}%<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_title,
            yaxis_title=y_title,
            height=300,
            margin=dict(l=20, r=20, t=40, b=40)
        )
    
    return fig


def create_multi_line_chart(data, title: str = "", x_title: str = "Date", y_title: str = "Count"):
    """
    Create a multi-line chart for time series data.
    
    Args:
        data: Dictionary of data for each line
        title: Chart title
        x_title: X-axis title
        y_title: Y-axis title
        
    Returns:
        go.Figure: Plotly Figure object
    """
    fig = go.Figure()
    
    for name, series in data.items():
        # Use custom color if provided, otherwise use Plotly's default
        line_color = series.get('color') if 'color' in series else None
        
        fig.add_trace(go.Scatter(
            x=series['x'],
            y=series['y'],
            mode='lines+markers',
            name=name,
            line=dict(width=3, color=line_color),
            marker=dict(size=8),
            hovertemplate='<b>%{x|%Y-%m}</b><br>Count: %{y:,}<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=400,
        margin=dict(l=20, r=20, t=40, b=40),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def create_database_breakdown_charts(entity_type, data_dict, title_prefix=""):
    """
    Create detailed donut charts for database breakdown.
    
    Args:
        entity_type: Type of entity ('document', 'chunk', etc.)
        data_dict: Dictionary with database breakdown data
        title_prefix: Prefix for the chart titles
        
    Returns:
        html.Div: Container with donut charts
    """
    if not data_dict:
        return html.Div("No database breakdown data available", className="text-muted text-center")
    
    charts = []
    # Create donut charts in rows of 3
    db_names = list(data_dict.keys())
    
    for i in range(0, len(db_names), 3):
        row_dbs = db_names[i:i+3]
        row_charts = []
        
        for db_name in row_dbs:
            db_data = data_dict[db_name]
            
            # Create donut chart for this database
            fig = create_pie_chart(
                inner_value=db_data['total'],
                inner_label="Total",
                values=[db_data['relevant'], db_data['irrelevant']],
                labels=["Relevant", "Irrelevant"],
                colors=['#4caf50', '#e0e0e0'],
                title=f"{db_name}\nCoverage: {db_data['coverage']}%"
            )
            
            # Make chart smaller
            fig.update_layout(
                height=250,
                margin=dict(l=10, r=10, t=50, b=60),
                font=dict(size=11)
            )
            
            row_charts.append(
                dbc.Col([
                    dcc.Graph(figure=fig, config={'displayModeBar': False})
                ], width=4)
            )
        
        charts.append(dbc.Row(row_charts, className="mb-3"))
    
    return html.Div([
        html.Hr(className="mt-5 mb-4"),
        html.H4(f"{title_prefix} by Database", className="mt-4 mb-3 text-center", style={"color": "#13376f"}),
        html.P(f"Showing coverage statistics for top {len(db_names)} databases", className="text-center text-muted mb-4"),
        html.Div(charts, style={"background-color": "#f8f9fa", "padding": "20px", "border-radius": "10px"})
    ], style={"margin-top": "50px"})


def create_documents_tab(
    documents_data,
    time_series_data=None,
    language_time_series=None,
    database_time_series=None,
    database_breakdown=None
):
    """
    Create the Documents tab content.
    
    Args:
        documents_data: Documents data dictionary
        time_series_data: Time series data DataFrame
        language_time_series: Language time series data DataFrame
        database_time_series: Database time series data DataFrame
        database_breakdown: Database breakdown data
        
    Returns:
        html.Div: Tab content
    """
    # About box for Documents tab
    about_box = html.Div([
        html.H5("About Documents", style={"color": "#13376f"}),
        html.P([
            "This tab shows statistics about documents in the corpus. A document is considered 'relevant' "
            "if it contains at least one chunk with taxonomic classification. The coverage metrics help "
            "identify which sources contribute most to the analyzed content."
        ]),
        html.Ul([
            html.Li("Total Documents: All documents in the filtered dataset"),
            html.Li("Relevant Documents: Documents containing classified content"),
            html.Li("Relevance Rate: Percentage of documents with taxonomic elements"),
            html.Li("Date Range: Time span of documents in the current filter")
        ])
    ], className="about-box mb-4")
    # Document Statistics card
    documents_stats_card = html.Div([
        html.Div([
            html.H1(f"{documents_data['total_documents']:,}", 
                    style={'text-align': 'center', 'font-size': '48px'}),
            html.Div("Total Documents", 
                     style={'text-align': 'center', 'color': '#666'})
        ], className="mb-4"),
        
        html.Hr(),
        
        html.Div([
            html.Div([
                html.Div(f"{documents_data['relevant_documents']:,}", className="mb-1", 
                         style={'font-weight': 'bold', 'font-size': '20px'}),
                html.Div("Relevant Documents")
            ], style={'display': 'inline-block', 'width': '48%', 'text-align': 'center'}),
            
            html.Div([
                html.Div(f"{documents_data['irrelevant_documents']:,}", className="mb-1", 
                         style={'font-weight': 'bold', 'font-size': '20px'}),
                html.Div("Irrelevant Documents")
            ], style={'display': 'inline-block', 'width': '48%', 'text-align': 'center'})
        ]),
        
        html.Hr(),
        
        html.Div([
            html.Div(f"Relevance Rate: {documents_data['relevance_rate']}%", 
                     style={'font-weight': 'bold', 'text-align': 'center', 'color': '#4caf50'})
        ]),
        
        html.Hr(),
        
        html.Div([
            html.Div([
                html.Div("Date Range:", className="mb-1", style={'font-weight': 'bold'}),
                html.Div(f"Earliest: {documents_data.get('earliest_date', 'N/A')}", className="mb-1"),
                html.Div(f"Latest: {documents_data.get('latest_date', 'N/A')}")
            ], style={'text-align': 'center'})
        ])
    ], className="card p-3")
    
    # Create pie chart for document relevance
    relevance_values = [documents_data['relevant_documents'], documents_data['irrelevant_documents']]
    relevance_labels = ["Relevant", "Irrelevant"]
    
    relevance_fig = create_pie_chart(
        inner_value=documents_data['total_documents'],
        inner_label="Total",
        values=relevance_values,
        labels=relevance_labels,
        colors=['#4caf50', '#ffc107'],  # Green for relevant, yellow for irrelevant
        title="Document Relevance Distribution"
    )
    
    # Create time series chart for documents over time
    if time_series_data is not None and not time_series_data.empty:
        # CRITICAL FIX: Apply date range filtering for time series data
        # Check if date columns are datetime objects
        if not pd.api.types.is_datetime64_any_dtype(time_series_data['date']):
            time_series_data['date'] = pd.to_datetime(time_series_data['date'])
            
        # Filter the time series data to match the current date range filter
        # This ensures the chart only shows data within the selected date range
        documents_time_fig = create_time_series_chart(
            x_values=time_series_data['date'].tolist(),
            y_values=time_series_data['count'].tolist(),
            title="Documents Over Time",
            color="#4caf50"  # Green color
        )
    else:
        # Create placeholder
        documents_time_fig = go.Figure().update_layout(title="No time series data available")
    
    # Language distribution chart with custom colors - FIX HERE
    by_language = documents_data['by_language']
    
    # Define custom colors for languages, using THEME_COLORS from config
    language_colors = []
    for lang in by_language['labels']:
        if lang in THEME_COLORS:
            language_colors.append(THEME_COLORS[lang])
        else:
            # Use a suitable default color
            language_colors.append(px.colors.qualitative.Pastel[len(language_colors) % len(px.colors.qualitative.Pastel)])
    
    language_fig = create_bar_chart(
        x_values=by_language['labels'],
        y_values=by_language['percentages'],
        title="Documents by Language",
        color=language_colors  # Use the custom colors array instead of a single color
    )
    
    # Database distribution chart with custom colors - FIX HERE
    top_databases = documents_data['top_databases']
    
    # Define custom colors for databases, using a different color sequence
    database_colors = []
    for i, db in enumerate(top_databases['labels']):
        # Use a different color for each database
        database_colors.append(px.colors.qualitative.Bold[i % len(px.colors.qualitative.Bold)])
    
    database_fig = create_bar_chart(
        x_values=top_databases['labels'],
        y_values=top_databases['percentages'],
        title="Top 10 Databases",
        color=database_colors  # Use the custom colors array
    )
    
    # Create language time series chart
    if language_time_series is not None and not language_time_series.empty:
        # Process the language time series data
        language_data = {}
        # Create a color map for languages
        language_color_map = {}
        for i, lang in enumerate(language_time_series['language'].unique()):
            # Check if language has a defined color in THEME_COLORS
            if lang in THEME_COLORS:
                language_color_map[lang] = THEME_COLORS[lang]
            else:
                language_color_map[lang] = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                
            lang_df = language_time_series[language_time_series['language'] == lang]
            language_data[lang] = {
                'x': lang_df['date'].tolist(),
                'y': lang_df['count'].tolist(),
                'color': language_color_map[lang]  # Add color to the data
            }
        
        language_time_fig = create_multi_line_chart(
            data=language_data,
            title="Documents by Language Over Time",
            x_title="Month",
            y_title="Document Count"
        )
    else:
        # Create placeholder
        language_time_fig = go.Figure().update_layout(title="No language time series data available")
    
    # Create database time series chart
    if database_time_series is not None and not database_time_series.empty:
        # Process the database time series data
        database_data = {}
        # Create a color map for databases
        database_color_map = {}
        for i, db in enumerate(database_time_series['database'].unique()):
            database_color_map[db] = px.colors.qualitative.Bold[i % len(px.colors.qualitative.Bold)]
            
            db_df = database_time_series[database_time_series['database'] == db]
            database_data[db] = {
                'x': db_df['date'].tolist(),
                'y': db_df['count'].tolist(),
                'color': database_color_map[db]  # Add color to the data
            }
        
        database_time_fig = create_multi_line_chart(
            data=database_data,
            title="Documents by Database Over Time (Top 5)",
            x_title="Month",
            y_title="Document Count"
        )
    else:
        # Create placeholder
        database_time_fig = go.Figure().update_layout(title="No database time series data available")
    
    # Combine everything into the tab layout
    documents_tab = html.Div([
        about_box,
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=relevance_fig)
            ], width=6),
            
            dbc.Col([
                documents_stats_card
            ], width=6)
        ], className="mb-4"),
        
        html.H5("Documents Over Time", className="mt-4"),
        dcc.Graph(figure=documents_time_fig),
        
        dbc.Row([
            dbc.Col([
                html.H5("By Language", className="mt-4"),
                dcc.Graph(figure=language_fig)
            ], width=6),
            
            dbc.Col([
                html.H5("By Database", className="mt-4"),
                dcc.Graph(figure=database_fig)
            ], width=6)
        ], className="mb-4"),
        
        dcc.Graph(figure=language_time_fig),
        
        dcc.Graph(figure=database_time_fig),
        
        # Add database breakdown charts at the bottom
        create_database_breakdown_charts('document', database_breakdown, "Document Coverage")
    ])
    
    return documents_tab

def create_chunks_tab(
    chunks_data,
    time_series_data=None,
    language_time_series=None,
    database_time_series=None,
    database_breakdown=None
):
    """
    Create the Chunks tab content.
    
    Args:
        chunks_data: Chunks data dictionary
        time_series_data: Time series data DataFrame
        language_time_series: Language time series data DataFrame
        database_time_series: Database time series data DataFrame
        database_breakdown: Database breakdown data
        
    Returns:
        html.Div: Tab content
    """
    # About box for Chunks tab
    about_box = html.Div([
        html.H5("About Chunks", style={"color": "#13376f"}),
        html.P([
            "Chunks are segments of text extracted from documents for analysis. A chunk is 'relevant' "
            "if it has been assigned at least one taxonomic classification. This tab shows how content "
            "is distributed across the corpus at the chunk level."
        ]),
        html.Ul([
            html.Li("Total Chunks: All text segments in the filtered dataset"),
            html.Li("Relevant Chunks: Chunks with taxonomic classifications"),
            html.Li("Avg. Chunks per Document: Average segmentation of documents"),
            html.Li("Coverage metrics show the proportion of analyzed content")
        ])
    ], className="about-box mb-4")
    # Chunk Statistics card
    chunks_stats_card = html.Div([
        html.Div([
            html.H1(f"{chunks_data['total_chunks']:,}", 
                    style={'text-align': 'center', 'font-size': '48px'}),
            html.Div("Total Chunks", 
                     style={'text-align': 'center', 'color': '#666'})
        ], className="mb-4"),
        
        html.Hr(),
        
        html.Div([
            html.Div([
                html.Div(f"{chunks_data['relevant_chunks']:,}", className="mb-1", 
                         style={'font-weight': 'bold', 'font-size': '20px'}),
                html.Div("Relevant Chunks")
            ], style={'display': 'inline-block', 'width': '48%', 'text-align': 'center'}),
            
            html.Div([
                html.Div(f"{chunks_data['irrelevant_chunks']:,}", className="mb-1", 
                         style={'font-weight': 'bold', 'font-size': '20px'}),
                html.Div("Irrelevant Chunks")
            ], style={'display': 'inline-block', 'width': '48%', 'text-align': 'center'})
        ]),
        
        html.Hr(),
        
        html.Div([
            html.Div(f"Relevance Rate: {chunks_data['relevance_rate']}%", 
                     style={'font-weight': 'bold', 'text-align': 'center', 'color': '#ff851b'})
        ], className="mb-4"),
        
        html.Div([
            html.Div(f"Avg. Chunks per Document: {chunks_data['avg_chunks_per_document']}")
        ], style={'text-align': 'center'}),
        
        html.Hr(),
        
        html.Div([
            html.Div([
                html.Div("Date Range:", className="mb-1", style={'font-weight': 'bold'}),
                html.Div(f"Earliest: 2001-04-16", className="mb-1"),
                html.Div(f"Latest: 2025-05-12")
            ], style={'text-align': 'center'})
        ])
    ], className="card p-3")
    
    # Create pie chart for chunk relevance
    relevance_values = [chunks_data['relevant_chunks'], chunks_data['irrelevant_chunks']]
    relevance_labels = ["Relevant", "Irrelevant"]
    
    relevance_fig = create_pie_chart(
        inner_value=chunks_data['total_chunks'],
        inner_label="Total",
        values=relevance_values,
        labels=relevance_labels,
        colors=['#2196f3', '#ff9800'],  # Blue for relevant, orange for irrelevant
        title="Chunk Relevance Distribution"
    )
    
    # Create time series chart for chunks over time
    if time_series_data is not None and not time_series_data.empty:
        # CRITICAL FIX: Apply date range filtering for time series data
        # Check if date columns are datetime objects
        if not pd.api.types.is_datetime64_any_dtype(time_series_data['date']):
            time_series_data['date'] = pd.to_datetime(time_series_data['date'])
            
        chunks_time_fig = create_time_series_chart(
            x_values=time_series_data['date'].tolist(),
            y_values=time_series_data['count'].tolist(),
            title="Chunks Over Time",
            color="#2196f3"  # Blue color
        )
    else:
        # Create placeholder
        chunks_time_fig = go.Figure().update_layout(title="No time series data available")
    
    # Language distribution chart with custom colors - FIX HERE
    by_language = chunks_data['by_language']
    
    # Define custom colors for languages, using THEME_COLORS from config
    language_colors = []
    for lang in by_language['labels']:
        if lang in THEME_COLORS:
            language_colors.append(THEME_COLORS[lang])
        else:
            # Use a suitable default color
            language_colors.append(px.colors.qualitative.Pastel[len(language_colors) % len(px.colors.qualitative.Pastel)])
    
    language_fig = create_bar_chart(
        x_values=by_language['labels'],
        y_values=by_language['percentages'],
        title="Chunks by Language",
        color=language_colors  # Use the custom colors array
    )
    
    # Database distribution chart with custom colors - FIX HERE
    top_databases = chunks_data['top_databases']
    
    # Define custom colors for databases, using a different color sequence
    database_colors = []
    for i, db in enumerate(top_databases['labels']):
        # Use a different color for each database
        database_colors.append(px.colors.qualitative.Bold[i % len(px.colors.qualitative.Bold)])
    
    database_fig = create_bar_chart(
        x_values=top_databases['labels'],
        y_values=top_databases['percentages'],
        title="Top 10 Databases by Chunks",
        color=database_colors  # Use the custom colors array
    )
    
    # Create language time series chart with custom colors
    if language_time_series is not None and not language_time_series.empty:
        # Process the language time series data
        language_data = {}
        # Create a color map for languages
        language_color_map = {}
        for i, lang in enumerate(language_time_series['language'].unique()):
            # Check if language has a defined color in THEME_COLORS
            if lang in THEME_COLORS:
                language_color_map[lang] = THEME_COLORS[lang]
            else:
                language_color_map[lang] = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                
            lang_df = language_time_series[language_time_series['language'] == lang]
            language_data[lang] = {
                'x': lang_df['date'].tolist(),
                'y': lang_df['count'].tolist(),
                'color': language_color_map[lang]  # Add color to the data
            }
        
        language_time_fig = create_multi_line_chart(
            data=language_data,
            title="Chunks by Language Over Time",
            x_title="Month",
            y_title="Chunk Count"
        )
    else:
        # Create placeholder
        language_time_fig = go.Figure().update_layout(title="No language time series data available")
    
    # Create database time series chart with custom colors
    if database_time_series is not None and not database_time_series.empty:
        # Process the database time series data
        database_data = {}
        # Create a color map for databases
        database_color_map = {}
        for i, db in enumerate(database_time_series['database'].unique()):
            database_color_map[db] = px.colors.qualitative.Bold[i % len(px.colors.qualitative.Bold)]
            
            db_df = database_time_series[database_time_series['database'] == db]
            database_data[db] = {
                'x': db_df['date'].tolist(),
                'y': db_df['count'].tolist(),
                'color': database_color_map[db]  # Add color to the data
            }
        
        database_time_fig = create_multi_line_chart(
            data=database_data,
            title="Chunks by Database Over Time (Top 5)",
            x_title="Month",
            y_title="Chunk Count"
        )
    else:
        # Create placeholder
        database_time_fig = go.Figure().update_layout(title="No database time series data available")
    
    # Combine everything into the tab layout
    chunks_tab = html.Div([
        about_box,
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=relevance_fig)
            ], width=6),
            
            dbc.Col([
                chunks_stats_card
            ], width=6)
        ], className="mb-4"),
        
        html.H5("Chunks Over Time", className="mt-4"),
        dcc.Graph(figure=chunks_time_fig),
        
        dbc.Row([
            dbc.Col([
                html.H5("By Language", className="mt-4"),
                dcc.Graph(figure=language_fig)
            ], width=6),
            
            dbc.Col([
                html.H5("By Database", className="mt-4"),
                dcc.Graph(figure=database_fig)
            ], width=6)
        ], className="mb-4"),
        
        dcc.Graph(figure=language_time_fig),
        
        dcc.Graph(figure=database_time_fig),
        
        # Add database breakdown charts at the bottom
        create_database_breakdown_charts('chunk', database_breakdown, "Chunk Coverage")
    ])
    
    return chunks_tab


def create_taxonomy_combinations_tab(
    taxonomy_data,
    time_series_data=None,
    language_time_series=None,
    database_time_series=None
):
    """
    Create the Taxonomy Combinations tab content.
    
    Args:
        taxonomy_data: Taxonomy data dictionary
        time_series_data: Time series data DataFrame
        language_time_series: Language time series data DataFrame
        database_time_series: Database time series data DataFrame
        
    Returns:
        html.Div: Tab content
    """
    # About box for Taxonomy Combinations tab
    about_box = html.Div([
        html.H5("About Taxonomy Combinations", style={"color": "#13376f"}),
        html.P([
            "This tab shows how taxonomic classifications are distributed across chunks. Many chunks have "
            "multiple taxonomic elements assigned, indicating complex or multi-faceted content. The distribution "
            "helps understand content complexity."
        ]),
        html.Ul([
            html.Li("0 combinations: Chunks without any taxonomic classification"),
            html.Li("1-4 combinations: Chunks with single or few classifications"),
            html.Li("5+ combinations: Highly complex chunks with many themes"),
            html.Li("Coverage shows the percentage of chunks with classifications")
        ])
    ], className="about-box mb-4")
    # Taxonomy Statistics card WITH THOUSANDS SEPARATORS
    taxonomy_stats_card = html.Div([
        html.Div([
            html.H1(f"{taxonomy_data['chunks_with_taxonomy']:,}", 
                    style={'text-align': 'center', 'font-size': '48px'}),
            html.Div("Chunks with Taxonomy", 
                     style={'text-align': 'center', 'color': '#666'})
        ], className="mb-4"),
        
        html.Hr(),
        
        html.Div([
            html.Div([
                html.Div(f"Taxonomy Coverage: {taxonomy_data['taxonomy_coverage']}%", 
                         className="mb-1"),
                html.Div(f"Avg. Taxonomies per Chunk: {taxonomy_data['avg_taxonomies_per_chunk']}")
            ])
        ]),
        
        html.Hr(),
        
        html.Div([
            html.Div([
                html.Div("Date Range:", className="mb-1", style={'font-weight': 'bold'}),
                html.Div(f"Earliest: 2001-04-16", className="mb-1"),
                html.Div(f"Latest: 2025-05-12")
            ], style={'text-align': 'center'})
        ]),
        
        html.Hr(),
        
        html.Div([
            html.H6("Distribution by Combination Count"),
            html.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Combinations"),
                        html.Th("Count"),
                        html.Th("Percentage")
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(comb),
                        html.Td(f"{count:,}"),
                        html.Td(f"{pct:.1f}%")
                    ]) for comb, count, pct in zip(
                        taxonomy_data['combinations_per_chunk']['labels'],
                        taxonomy_data['combinations_per_chunk']['values'],
                        taxonomy_data['combinations_per_chunk']['percentages']
                    )
                ])
            ], className="table table-striped table-sm")
        ])
    ], className="card p-3")
    
    # Create pie chart for taxonomy combinations
    pie_values = taxonomy_data['combinations_per_chunk']['values']
    pie_labels = taxonomy_data['combinations_per_chunk']['labels']
    
    # Create pie chart for taxonomy combinations
    combinations_fig = create_pie_chart(
        inner_value=taxonomy_data['total_chunks'],
        inner_label="Total",
        values=pie_values,
        labels=pie_labels,
        colors=['#ffeb3a', '#98df8a', '#aec7e8', '#1f77b4', '#ff7f0e', '#d62728'],
        title="Taxonomy Combinations per Chunk"
    )
    
    # Create time series chart for taxonomy elements over time
    if time_series_data is not None and not time_series_data.empty:
        # CRITICAL FIX: Apply date range filtering for time series data
        # Check if date columns are datetime objects
        if not pd.api.types.is_datetime64_any_dtype(time_series_data['date']):
            time_series_data['date'] = pd.to_datetime(time_series_data['date'])
            
        taxonomy_time_fig = create_time_series_chart(
            x_values=time_series_data['date'].tolist(),
            y_values=time_series_data['count'].tolist(),
            title="Taxonomy Elements Over Time",
            color="#00a65a"  # Green color
        )
    else:
        # Create placeholder
        taxonomy_time_fig = go.Figure().update_layout(title="No taxonomy time series data available")
    
    # Create language time series chart with custom colors
    if language_time_series is not None and not language_time_series.empty:
        # Process the language time series data
        language_data = {}
        # Create a color map for languages
        language_color_map = {}
        for i, lang in enumerate(language_time_series['language'].unique()):
            # Check if language has a defined color in THEME_COLORS
            if lang in THEME_COLORS:
                language_color_map[lang] = THEME_COLORS[lang]
            else:
                language_color_map[lang] = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                
            lang_df = language_time_series[language_time_series['language'] == lang]
            language_data[lang] = {
                'x': lang_df['date'].tolist(),
                'y': lang_df['count'].tolist(),
                'color': language_color_map[lang]  # Add color to the data
            }
        
        language_distribution_data = {
            'x': list(language_data.keys()),
            'y': [sum(language_data[lang]['y']) for lang in language_data.keys()]
        }
        
        # Calculate percentages
        total = sum(language_distribution_data['y'])
        language_distribution_data['y'] = [round(val / total * 100, 1) if total > 0 else 0 for val in language_distribution_data['y']]
        
        # Create language bar chart with custom colors
        language_colors = []
        for lang in language_distribution_data['x']:
            if lang in THEME_COLORS:
                language_colors.append(THEME_COLORS[lang])
            else:
                language_colors.append(px.colors.qualitative.Pastel[len(language_colors) % len(px.colors.qualitative.Pastel)])
        
        language_fig = create_bar_chart(
            x_values=language_distribution_data['x'],
            y_values=language_distribution_data['y'],
            title="Taxonomy Elements by Language",
            color=language_colors  # Use custom colors
        )
        
        language_time_fig = create_multi_line_chart(
            data=language_data,
            title="Taxonomy Elements by Language Over Time",
            x_title="Month",
            y_title="Taxonomy Count"
        )
    else:
        # Create placeholders
        language_fig = go.Figure().update_layout(title="No language distribution data available")
        language_time_fig = go.Figure().update_layout(title="No language time series data available")
    
    # Create database distribution and time series with custom colors
    if database_time_series is not None and not database_time_series.empty:
        # Process the database time series data
        database_data = {}
        # Create a color map for databases
        database_color_map = {}
        for i, db in enumerate(database_time_series['database'].unique()):
            database_color_map[db] = px.colors.qualitative.Bold[i % len(px.colors.qualitative.Bold)]
            
            db_df = database_time_series[database_time_series['database'] == db]
            database_data[db] = {
                'x': db_df['date'].tolist(),
                'y': db_df['count'].tolist(),
                'color': database_color_map[db]  # Add color to the data
            }
        
        database_distribution_data = {
            'x': list(database_data.keys()),
            'y': [sum(database_data[db]['y']) for db in database_data.keys()]
        }
        
        # Calculate percentages
        total = sum(database_distribution_data['y'])
        database_distribution_data['y'] = [round(val / total * 100, 1) if total > 0 else 0 for val in database_distribution_data['y']]
        
        # Create database bar chart with custom colors
        database_colors = []
        for i, db in enumerate(database_distribution_data['x']):
            database_colors.append(px.colors.qualitative.Bold[i % len(px.colors.qualitative.Bold)])
        
        database_fig = create_bar_chart(
            x_values=database_distribution_data['x'],
            y_values=database_distribution_data['y'],
            title="Top 10 Databases by Taxonomy Elements",
            color=database_colors  # Use custom colors
        )
        
        database_time_fig = create_multi_line_chart(
            data=database_data,
            title="Taxonomy Elements by Database Over Time",
            x_title="Month",
            y_title="Taxonomy Count"
        )
    else:
        # Create placeholders
        database_fig = go.Figure().update_layout(title="No database distribution data available")
        database_time_fig = go.Figure().update_layout(title="No database time series data available")
    
    # Combine everything into the tab layout
    taxonomy_tab = html.Div([
        about_box,
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=combinations_fig)
            ], width=6),
            
            dbc.Col([
                taxonomy_stats_card
            ], width=6)
        ], className="mb-4"),
        
        html.H5("Taxonomy Elements Over Time", className="mt-4"),
        dcc.Graph(figure=taxonomy_time_fig),
        
        dbc.Row([
            dbc.Col([
                html.H5("By Language", className="mt-4"),
                dcc.Graph(figure=language_fig)
            ], width=6),
            
            dbc.Col([
                html.H5("By Database", className="mt-4"),
                dcc.Graph(figure=database_fig)
            ], width=6)
        ], className="mb-4"),
        
        dcc.Graph(figure=language_time_fig),
        
        dcc.Graph(figure=database_time_fig)
    ])
    
    return taxonomy_tab


def create_keywords_tab(
    keywords_data,
    time_series_data=None,
    language_time_series=None,
    database_time_series=None,
    database_breakdown=None
):
    """
    Create the Keywords tab content.
    
    Args:
        keywords_data: Keywords data dictionary
        time_series_data: Time series data DataFrame
        language_time_series: Language time series data DataFrame
        database_time_series: Database time series data DataFrame
        database_breakdown: Database breakdown data
        
    Returns:
        html.Div: Tab content
    """
    # About box for Keywords tab
    about_box = html.Div([
        html.H5("About Keywords", style={"color": "#13376f"}),
        html.P([
            "Keywords are automatically extracted terms that represent key concepts in the text. "
            "NOTE: Keywords are extracted from ALL chunks in the corpus, not just those with taxonomic "
            "classifications. This provides a comprehensive view of all content themes."
        ]),
        html.Ul([
            html.Li("Unique Keywords: Distinct terms found across all chunks"),
            html.Li("Total Occurrences: Sum of all keyword appearances"),
            html.Li("Coverage: Percentage of ALL chunks containing keywords"),
            html.Li("Top keywords show the most frequent concepts in the corpus")
        ]),
        html.P([
            html.Strong("Important: "),
            "Keyword statistics include both relevant and irrelevant chunks to provide complete corpus coverage."
        ], className="text-info")
    ], className="about-box mb-4")
    # Keywords Statistics card
    keywords_stats_card = html.Div([
        html.Div([
            html.H1(f"{keywords_data['total_unique_keywords']:,}", 
                    style={'text-align': 'center', 'font-size': '48px'}),
            html.Div("Unique Keywords", 
                     style={'text-align': 'center', 'color': '#666'})
        ], className="mb-4"),
        
        html.Hr(),
        
        html.Div([
            html.Div([
                html.Div(f"{keywords_data['total_keyword_occurrences']:,}", className="mb-1", 
                         style={'font-weight': 'bold', 'font-size': '20px'}),
                html.Div("Total Occurrences")
            ], style={'display': 'inline-block', 'width': '48%', 'text-align': 'center'}),
            
            html.Div([
                html.Div(f"{keywords_data['chunks_with_keywords']:,}", className="mb-1", 
                         style={'font-weight': 'bold', 'font-size': '20px'}),
                html.Div("Chunks with Keywords")
            ], style={'display': 'inline-block', 'width': '48%', 'text-align': 'center'})
        ]),
        
        html.Hr(),
        
        html.Div([
            html.Div(f"Keyword Coverage: {keywords_data['keyword_coverage']}%", 
                     style={'font-weight': 'bold', 'text-align': 'center', 'color': '#9c27b0'})
        ]),
        
        html.Hr(),
        
        html.Div([
            html.Div(f"Avg. Keywords per Chunk: {keywords_data['avg_keywords_per_chunk']}", 
                     style={'text-align': 'center'})
        ])
    ], className="card p-3")
    
    # Create pie chart for keyword coverage
    coverage_values = [keywords_data['chunks_with_keywords'], keywords_data['total_chunks'] - keywords_data['chunks_with_keywords']]
    coverage_labels = ["With Keywords", "Without Keywords"]
    
    coverage_fig = create_pie_chart(
        inner_value=keywords_data['total_chunks'],
        inner_label="Total Chunks",
        values=coverage_values,
        labels=coverage_labels,
        colors=['#9c27b0', '#e0e0e0'],  # Purple for with keywords, grey for without
        title="Keyword Coverage"
    )
    
    # Create bar chart for top keywords
    if keywords_data['top_keywords']['labels']:
        # Limit to top 15 for better visibility
        top_15_labels = keywords_data['top_keywords']['labels'][:15]
        top_15_values = keywords_data['top_keywords']['values'][:15]
        
        # Generate unique colors for each keyword using tab20
        # Plotly doesn't have tab20, so we'll define it manually
        tab20_colors = [
            '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
            '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
            '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
            '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5'
        ]
        keyword_colors = [tab20_colors[i % len(tab20_colors)] for i in range(len(top_15_labels))]
        
        top_keywords_fig = go.Figure(go.Bar(
            x=top_15_values,
            y=top_15_labels,
            orientation='h',
            marker_color=keyword_colors,
            text=top_15_values,
            textposition='auto',
            hovertemplate='%{y}<br>Count: %{x:,}<extra></extra>'
        ))
        
        top_keywords_fig.update_layout(
            title="Top 15 Keywords",
            xaxis_title="Occurrences",
            yaxis=dict(autorange="reversed"),
            height=500,
            margin=dict(l=150, r=20, t=40, b=40)
        )
    else:
        top_keywords_fig = go.Figure().update_layout(title="No keyword data available")
    
    # Create distribution chart for keywords per chunk
    if keywords_data['keywords_per_chunk_distribution']['labels']:
        dist_fig = create_bar_chart(
            x_values=keywords_data['keywords_per_chunk_distribution']['labels'],
            y_values=keywords_data['keywords_per_chunk_distribution']['percentages'],
            title="Keywords per Chunk Distribution",
            x_title="Number of Keywords",
            color='#9c27b0'
        )
    else:
        dist_fig = go.Figure().update_layout(title="No distribution data available")
    
    # Language distribution chart
    if keywords_data['by_language']['labels']:
        # Calculate percentages for language distribution
        total_lang_occurrences = sum(keywords_data['by_language']['total_occurrences'])
        lang_percentages = [round(v / total_lang_occurrences * 100, 1) if total_lang_occurrences > 0 else 0 
                           for v in keywords_data['by_language']['total_occurrences']]
        
        # Define custom colors for languages
        language_colors = []
        for lang in keywords_data['by_language']['labels']:
            if lang in THEME_COLORS:
                language_colors.append(THEME_COLORS[lang])
            else:
                language_colors.append(px.colors.qualitative.Pastel[len(language_colors) % len(px.colors.qualitative.Pastel)])
        
        language_fig = create_bar_chart(
            x_values=keywords_data['by_language']['labels'],
            y_values=lang_percentages,
            title="Keywords by Language",
            color=language_colors
        )
    else:
        language_fig = go.Figure().update_layout(title="No language data available")
    
    # Database distribution chart
    if keywords_data['by_database']['labels']:
        # Calculate percentages for database distribution
        total_db_occurrences = sum(keywords_data['by_database']['total_occurrences'])
        db_percentages = [round(v / total_db_occurrences * 100, 1) if total_db_occurrences > 0 else 0 
                         for v in keywords_data['by_database']['total_occurrences']]
        
        # Define custom colors for databases
        database_colors = []
        for i, db in enumerate(keywords_data['by_database']['labels']):
            database_colors.append(px.colors.qualitative.Bold[i % len(px.colors.qualitative.Bold)])
        
        database_fig = create_bar_chart(
            x_values=keywords_data['by_database']['labels'],
            y_values=db_percentages,
            title="Top 10 Databases by Keywords",
            color=database_colors
        )
    else:
        database_fig = go.Figure().update_layout(title="No database data available")
    
    # Create time series chart if available
    if time_series_data is not None and not time_series_data.empty:
        if not pd.api.types.is_datetime64_any_dtype(time_series_data['date']):
            time_series_data['date'] = pd.to_datetime(time_series_data['date'])
            
        keywords_time_fig = create_time_series_chart(
            x_values=time_series_data['date'].tolist(),
            y_values=time_series_data['count'].tolist(),
            title="Keyword Occurrences Over Time",
            color="#9c27b0"  # Purple color
        )
    else:
        keywords_time_fig = go.Figure().update_layout(title="No time series data available")
    
    # Create language time series chart
    if language_time_series is not None and not language_time_series.empty:
        language_data = {}
        language_color_map = {}
        for i, lang in enumerate(language_time_series['language'].unique()):
            if lang in THEME_COLORS:
                language_color_map[lang] = THEME_COLORS[lang]
            else:
                language_color_map[lang] = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                
            lang_df = language_time_series[language_time_series['language'] == lang]
            language_data[lang] = {
                'x': lang_df['date'].tolist(),
                'y': lang_df['count'].tolist(),
                'color': language_color_map[lang]
            }
        
        language_time_fig = create_multi_line_chart(
            data=language_data,
            title="Keyword Occurrences by Language Over Time",
            x_title="Month",
            y_title="Keyword Count"
        )
    else:
        language_time_fig = go.Figure().update_layout(title="No language time series data available")
    
    # Create database time series chart
    if database_time_series is not None and not database_time_series.empty:
        database_data = {}
        database_color_map = {}
        for i, db in enumerate(database_time_series['database'].unique()):
            database_color_map[db] = px.colors.qualitative.Bold[i % len(px.colors.qualitative.Bold)]
            
            db_df = database_time_series[database_time_series['database'] == db]
            database_data[db] = {
                'x': db_df['date'].tolist(),
                'y': db_df['count'].tolist(),
                'color': database_color_map[db]
            }
        
        database_time_fig = create_multi_line_chart(
            data=database_data,
            title="Keyword Occurrences by Database Over Time (Top 5)",
            x_title="Month",
            y_title="Keyword Count"
        )
    else:
        database_time_fig = go.Figure().update_layout(title="No database time series data available")
    
    # Combine everything into the tab layout
    keywords_tab = html.Div([
        about_box,
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=coverage_fig)
            ], width=6),
            
            dbc.Col([
                keywords_stats_card
            ], width=6)
        ], className="mb-4"),
        
        dcc.Graph(figure=top_keywords_fig),
        
        html.H5("Keyword Distribution", className="mt-4"),
        dcc.Graph(figure=dist_fig),
        
        html.H5("Keywords Over Time", className="mt-4"),
        dcc.Graph(figure=keywords_time_fig),
        
        dbc.Row([
            dbc.Col([
                html.H5("By Language", className="mt-4"),
                dcc.Graph(figure=language_fig)
            ], width=6),
            
            dbc.Col([
                html.H5("By Database", className="mt-4"),
                dcc.Graph(figure=database_fig)
            ], width=6)
        ], className="mb-4"),
        
        dcc.Graph(figure=language_time_fig),
        
        dcc.Graph(figure=database_time_fig),
        
        # Add database breakdown charts at the bottom
        create_database_breakdown_charts('keyword', database_breakdown, "Keyword Coverage")
    ])
    
    return keywords_tab


def create_named_entities_tab(
    entities_data,
    time_series_data=None,
    language_time_series=None,
    database_time_series=None,
    database_breakdown=None
):
    """
    Create the Named Entities tab content.
    
    Args:
        entities_data: Named entities data dictionary
        time_series_data: Time series data DataFrame
        language_time_series: Language time series data DataFrame
        database_time_series: Database time series data DataFrame
        database_breakdown: Database breakdown data
        
    Returns:
        html.Div: Tab content
    """
    # Create entity type filter dropdown
    entity_type_options = [{'label': 'All Entity Types', 'value': 'ALL'}]
    if entities_data['entity_types']['labels']:
        for entity_type in entities_data['entity_types']['labels']:
            entity_type_options.append({'label': entity_type, 'value': entity_type})
    
    entity_type_filter = html.Div([
        html.Label("Filter by Entity Type:", style={"font-weight": "bold", "margin-bottom": "5px"}),
        dcc.Dropdown(
            id='entity-type-filter',
            options=entity_type_options,
            value='ALL',
            clearable=False,
            style={'width': '200px'}
        )
    ], className="mb-3")
    
    # About box for Named Entities tab
    about_box = html.Div([
        html.H5("About Named Entities", style={"color": "#13376f"}),
        html.P([
            "Named entities are automatically identified people, places, organizations, and other proper nouns "
            "in the text. NOTE: Like keywords, named entities are extracted from ALL chunks in the corpus, "
            "not just those with taxonomic classifications."
        ]),
        html.Ul([
            html.Li("Entity Types: Categories like PERSON, ORG, GPE (geopolitical entities), LOC, etc."),
            html.Li("Filter by Type: Use the dropdown above to focus on specific entity types"),
            html.Li("Coverage: Percentage of ALL chunks containing named entities"),
            html.Li("Top entities show the most mentioned people, places, and organizations")
        ]),
        html.P([
            html.Strong("Important: "),
            "Entity statistics include both relevant and irrelevant chunks. Use the entity type filter "
            "to explore specific categories of entities."
        ], className="text-info")
    ], className="about-box mb-4")
    
    # Named Entities Statistics card
    entities_stats_card = html.Div([
        html.Div([
            html.H1(f"{entities_data['total_unique_entities']:,}", 
                    style={'text-align': 'center', 'font-size': '48px'}),
            html.Div("Unique Named Entities", 
                     style={'text-align': 'center', 'color': '#666'})
        ], className="mb-4"),
        
        html.Hr(),
        
        html.Div([
            html.Div([
                html.Div(f"{entities_data['total_entity_occurrences']:,}", className="mb-1", 
                         style={'font-weight': 'bold', 'font-size': '20px'}),
                html.Div("Total Occurrences")
            ], style={'display': 'inline-block', 'width': '48%', 'text-align': 'center'}),
            
            html.Div([
                html.Div(f"{entities_data['chunks_with_entities']:,}", className="mb-1", 
                         style={'font-weight': 'bold', 'font-size': '20px'}),
                html.Div("Chunks with Entities")
            ], style={'display': 'inline-block', 'width': '48%', 'text-align': 'center'})
        ]),
        
        html.Hr(),
        
        html.Div([
            html.Div(f"Entity Coverage: {entities_data['entity_coverage']}%", 
                     style={'font-weight': 'bold', 'text-align': 'center', 'color': '#e91e63'})
        ]),
        
        html.Hr(),
        
        html.Div([
            html.Div(f"Avg. Entities per Chunk: {entities_data['avg_entities_per_chunk']}", 
                     className="mb-1", style={'text-align': 'center'}),
            html.Div(f"Entity Types: {entities_data['entity_types_count']:,}", 
                     style={'text-align': 'center'})
        ])
    ], className="card p-3")
    
    # Create pie chart for entity coverage
    coverage_values = [entities_data['chunks_with_entities'], entities_data['total_chunks'] - entities_data['chunks_with_entities']]
    coverage_labels = ["With Entities", "Without Entities"]
    
    coverage_fig = create_pie_chart(
        inner_value=entities_data['total_chunks'],
        inner_label="Total Chunks",
        values=coverage_values,
        labels=coverage_labels,
        colors=['#e91e63', '#e0e0e0'],  # Pink for with entities, grey for without
        title="Named Entity Coverage"
    )
    
    # Create bar chart for top entities
    if entities_data['top_entities']['labels']:
        # Format labels to include entity type
        formatted_labels = []
        for i in range(len(entities_data['top_entities']['labels'])):
            entity_text = entities_data['top_entities']['labels'][i]
            entity_type = entities_data['top_entities']['types'][i] if 'types' in entities_data['top_entities'] else 'N/A'
            formatted_labels.append(f"{entity_text} ({entity_type})")
        
        # Limit to top 15 for better visibility
        top_15_labels = formatted_labels[:15]
        top_15_values = entities_data['top_entities']['values'][:15]
        top_15_types = [entities_data['top_entities']['types'][i] if 'types' in entities_data['top_entities'] else 'N/A' 
                        for i in range(min(15, len(entities_data['top_entities']['types'])))]
        
        # Define base colors for entity types
        entity_type_colors = {
            'PERSON': '#e41a1c',      # Red
            'ORG': '#377eb8',         # Blue  
            'GPE': '#4daf4a',         # Green (Geopolitical entities)
            'LOC': '#984ea3',         # Purple (Locations)
            'DATE': '#ff7f00',        # Orange
            'EVENT': '#ffff33',       # Yellow
            'FAC': '#a65628',         # Brown (Facilities)
            'PRODUCT': '#f781bf',     # Pink
            'NORP': '#999999',        # Gray (Nationalities, religious, political groups)
            'CARDINAL': '#66c2a5',    # Teal
            'ORDINAL': '#fc8d62',     # Light orange
            'TIME': '#8da0cb',        # Light blue
            'MONEY': '#e78ac3',       # Light pink
            'PERCENT': '#a6d854',     # Light green
            'QUANTITY': '#ffd92f',    # Light yellow
            'LANGUAGE': '#e5c494',    # Beige
            'LAW': '#b3b3b3',         # Light gray
        }
        
        # Generate colors with alpha variations for entities of the same type
        entity_colors = []
        type_counts = {}
        
        for i, entity_type in enumerate(top_15_types):
            base_color = entity_type_colors.get(entity_type, '#666666')  # Default gray
            
            # Count occurrences of this type to vary alpha
            if entity_type not in type_counts:
                type_counts[entity_type] = 0
            type_counts[entity_type] += 1
            
            # Convert hex to RGB and add alpha variation
            # Alpha varies from 1.0 to 0.6 based on occurrence
            alpha = 1.0 - (type_counts[entity_type] - 1) * 0.1
            alpha = max(alpha, 0.6)  # Don't go below 0.6 alpha
            
            # Convert hex to rgba
            hex_color = base_color.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            entity_colors.append(f'rgba({r}, {g}, {b}, {alpha})')
        
        top_entities_fig = go.Figure(go.Bar(
            x=top_15_values,
            y=top_15_labels,
            orientation='h',
            marker_color=entity_colors,
            text=top_15_values,
            textposition='auto',
            hovertemplate='%{y}<br>Count: %{x:,}<extra></extra>'
        ))
        
        top_entities_fig.update_layout(
            title="Top 15 Named Entities",
            xaxis_title="Occurrences",
            yaxis=dict(autorange="reversed"),
            height=500,
            margin=dict(l=200, r=20, t=40, b=40)
        )
    else:
        top_entities_fig = go.Figure().update_layout(title="No entity data available")
    
    # Create entity types distribution chart
    if entities_data['entity_types']['labels']:
        # Calculate percentages
        total_type_count = sum(entities_data['entity_types']['counts'])
        type_percentages = [round(v / total_type_count * 100, 1) if total_type_count > 0 else 0 
                           for v in entities_data['entity_types']['counts']]
        
        # Use different colors for different entity types
        type_colors = px.colors.qualitative.Set3[:len(entities_data['entity_types']['labels'])]
        
        entity_types_fig = create_bar_chart(
            x_values=entities_data['entity_types']['labels'],
            y_values=type_percentages,
            title="Distribution by Entity Type",
            x_title="Entity Type",
            color=type_colors
        )
    else:
        entity_types_fig = go.Figure().update_layout(title="No entity type data available")
    
    # Create distribution chart for entities per chunk
    if entities_data['entities_per_chunk_distribution']['labels']:
        dist_fig = create_bar_chart(
            x_values=entities_data['entities_per_chunk_distribution']['labels'],
            y_values=entities_data['entities_per_chunk_distribution']['percentages'],
            title="Entities per Chunk Distribution",
            x_title="Number of Entities",
            color='#e91e63'
        )
    else:
        dist_fig = go.Figure().update_layout(title="No distribution data available")
    
    # Language distribution chart
    if entities_data['by_language']['labels']:
        # Calculate percentages for language distribution
        total_lang_occurrences = sum(entities_data['by_language']['total_occurrences'])
        lang_percentages = [round(v / total_lang_occurrences * 100, 1) if total_lang_occurrences > 0 else 0 
                           for v in entities_data['by_language']['total_occurrences']]
        
        # Define custom colors for languages
        language_colors = []
        for lang in entities_data['by_language']['labels']:
            if lang in THEME_COLORS:
                language_colors.append(THEME_COLORS[lang])
            else:
                language_colors.append(px.colors.qualitative.Pastel[len(language_colors) % len(px.colors.qualitative.Pastel)])
        
        language_fig = create_bar_chart(
            x_values=entities_data['by_language']['labels'],
            y_values=lang_percentages,
            title="Named Entities by Language",
            color=language_colors
        )
    else:
        language_fig = go.Figure().update_layout(title="No language data available")
    
    # Database distribution chart
    if entities_data['by_database']['labels']:
        # Calculate percentages for database distribution
        total_db_occurrences = sum(entities_data['by_database']['total_occurrences'])
        db_percentages = [round(v / total_db_occurrences * 100, 1) if total_db_occurrences > 0 else 0 
                         for v in entities_data['by_database']['total_occurrences']]
        
        # Define custom colors for databases
        database_colors = []
        for i, db in enumerate(entities_data['by_database']['labels']):
            database_colors.append(px.colors.qualitative.Bold[i % len(px.colors.qualitative.Bold)])
        
        database_fig = create_bar_chart(
            x_values=entities_data['by_database']['labels'],
            y_values=db_percentages,
            title="Top 10 Databases by Named Entities",
            color=database_colors
        )
    else:
        database_fig = go.Figure().update_layout(title="No database data available")
    
    # Create time series chart if available
    if time_series_data is not None and not time_series_data.empty:
        if not pd.api.types.is_datetime64_any_dtype(time_series_data['date']):
            time_series_data['date'] = pd.to_datetime(time_series_data['date'])
            
        entities_time_fig = create_time_series_chart(
            x_values=time_series_data['date'].tolist(),
            y_values=time_series_data['count'].tolist(),
            title="Named Entity Occurrences Over Time",
            color="#e91e63"  # Pink color
        )
    else:
        entities_time_fig = go.Figure().update_layout(title="No time series data available")
    
    # Create language time series chart
    if language_time_series is not None and not language_time_series.empty:
        language_data = {}
        language_color_map = {}
        for i, lang in enumerate(language_time_series['language'].unique()):
            if lang in THEME_COLORS:
                language_color_map[lang] = THEME_COLORS[lang]
            else:
                language_color_map[lang] = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                
            lang_df = language_time_series[language_time_series['language'] == lang]
            language_data[lang] = {
                'x': lang_df['date'].tolist(),
                'y': lang_df['count'].tolist(),
                'color': language_color_map[lang]
            }
        
        language_time_fig = create_multi_line_chart(
            data=language_data,
            title="Named Entity Occurrences by Language Over Time",
            x_title="Month",
            y_title="Entity Count"
        )
    else:
        language_time_fig = go.Figure().update_layout(title="No language time series data available")
    
    # Create database time series chart
    if database_time_series is not None and not database_time_series.empty:
        database_data = {}
        database_color_map = {}
        for i, db in enumerate(database_time_series['database'].unique()):
            database_color_map[db] = px.colors.qualitative.Bold[i % len(px.colors.qualitative.Bold)]
            
            db_df = database_time_series[database_time_series['database'] == db]
            database_data[db] = {
                'x': db_df['date'].tolist(),
                'y': db_df['count'].tolist(),
                'color': database_color_map[db]
            }
        
        database_time_fig = create_multi_line_chart(
            data=database_data,
            title="Named Entity Occurrences by Database Over Time (Top 5)",
            x_title="Month",
            y_title="Entity Count"
        )
    else:
        database_time_fig = go.Figure().update_layout(title="No database time series data available")
    
    # Combine everything into the tab layout
    entities_tab = html.Div([
        about_box,
        
        # Add entity type filter
        entity_type_filter,
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=coverage_fig)
            ], width=6),
            
            dbc.Col([
                entities_stats_card
            ], width=6)
        ], className="mb-4"),
        
        dcc.Graph(figure=top_entities_fig),
        
        dcc.Graph(figure=entity_types_fig),
        
        html.H5("Entity Distribution", className="mt-4"),
        dcc.Graph(figure=dist_fig),
        
        html.H5("Named Entities Over Time", className="mt-4"),
        dcc.Graph(figure=entities_time_fig),
        
        dbc.Row([
            dbc.Col([
                html.H5("By Language", className="mt-4"),
                dcc.Graph(figure=language_fig)
            ], width=6),
            
            dbc.Col([
                html.H5("By Database", className="mt-4"),
                dcc.Graph(figure=database_fig)
            ], width=6)
        ], className="mb-4"),
        
        dcc.Graph(figure=language_time_fig),
        
        dcc.Graph(figure=database_time_fig),
        
        # Add database breakdown charts at the bottom
        create_database_breakdown_charts('entity', database_breakdown, "Entity Coverage")
    ])
    
    return entities_tab


def create_sources_tab_layout(db_options: List, min_date: datetime = None, max_date: datetime = None):
    """
    Create the Sources tab layout.
    
    Args:
        db_options: Database options for filters
        min_date: Minimum date for filters
        max_date: Maximum date for filters
        
    Returns:
        html.Div: Sources tab layout
    """
    # DON'T fetch data here! This defeats lazy loading!
    # Data will be fetched in the callback when needed
    
    # Create corpus overview header
    corpus_overview = html.Div([
        html.Div([
            html.H3("Corpus Overview", style={"display": "inline-block", "margin-right": "20px"}),
            dbc.Button(
                "About", 
                id="open-about-sources", 
                color="secondary", 
                size="sm",
                className="ml-auto",
                style={"display": "inline-block"}
            ),
        ], style={"display": "flex", "align-items": "center", "margin-bottom": "20px"}),
        
        # Filter card with custom spinner
        html.Div([
            create_filter_card(
                id_prefix="sources",
                db_options=db_options,
                min_date=min_date,
                max_date=max_date
            ),
            # Add loading spinner that will show when filters are being applied
            dcc.Loading(
                id="sources-loading-spinner",
                type="circle",  # Use circle type which we'll style with CSS
                color="#13376f",
                children=[html.Div(id="sources-loading-output")],
                style={"position": "absolute", "top": "50%", "left": "50%", "transform": "translate(-50%, -50%)"},
                className="radar-loading"
            )
        ], style={"position": "relative"}),
        
        # Last updated info
        html.Div([
            "Data shown here reflects the latest state of the corpus and is updated regularly. Last updated: ",
            html.Span(datetime.now().strftime("%Y-%m-%d %H:%M"), id="sources-last-updated")
        ], className="text-muted mb-4", style={"border": "1px solid #ddd", "padding": "10px", "background": "#f9f9f9"})
    ])
    
    # Create placeholder content for lazy loading
    loading_content = html.Div([
        dcc.Loading(
            type="default",
            children=[
                html.Div([
                    html.H4("Loading data...", className="text-center mb-3"),
                    html.P("This may take a moment on first load. Data will be cached for faster access later.", 
                          className="text-center text-muted")
                ], className="p-5")
            ]
        )
    ])
    
    # Create initial placeholder tabs
    documents_subtab = loading_content
    chunks_subtab = loading_content
    taxonomy_subtab = loading_content  
    keywords_subtab = loading_content
    entities_subtab = loading_content
    
    # Create the sources tab with subtabs
    sources_tab = html.Div([
        corpus_overview,
        
        # Subtabs
        dcc.Tabs([
            dcc.Tab(label="Documents", children=documents_subtab),
            dcc.Tab(label="Chunks", children=chunks_subtab),
            dcc.Tab(label="Taxonomy Combinations", children=taxonomy_subtab),
            dcc.Tab(label="Keywords", children=keywords_subtab),
            dcc.Tab(label="Named Entities", children=entities_subtab)
        ], id="sources-subtabs", className="custom-tabs"),
        
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
    Register callbacks for the Sources tab.
    
    Args:
        app: Dash application instance
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
    
# We'll remove this separate callback and instead incorporate the spinner into the main data loading callback
    @app.callback(
        [
            Output("sources-result-stats", "children"),
            Output("sources-subtabs", "children"),
            Output("sources-loading-output", "children")  # Add output for the spinner
        ],
        [
            Input("sources-filter-button", "n_clicks")
        ],
        [
            State("sources-language-dropdown", "value"),
            State("sources-database-dropdown", "value"),
            State("sources-source-type-dropdown", "value"),
            State("sources-date-range-picker", "start_date"),
            State("sources-date-range-picker", "end_date"),
            State("sources-subtabs", "value")
        ]
    )
    def update_sources_tab(n_clicks, lang_val, db_val, source_type, start_date, end_date, active_tab):
        """
        Update the Sources tab based on filter selections.
        
        Args:
            n_clicks: Number of button clicks
            lang_val: Selected language
            db_val: Selected database
            source_type: Selected source type
            start_date: Start date
            end_date: End date
            active_tab: Currently active tab
            
        Returns:
            tuple: (stats_html, updated_tabs)
        """
        # If the button wasn't clicked, return minimal data for lazy loading
        if not n_clicks:
            # Only fetch corpus stats for the header
            corpus_stats = fetch_corpus_stats()
            stats_html = html.Div([
                html.P(f"Docs: {corpus_stats['docs_count']:,} ({corpus_stats['docs_rel_count']:,} rel) | Chunks: {corpus_stats['chunks_count']:,} ({corpus_stats['chunks_rel_count']:,} rel) | Tax: {corpus_stats['tax_levels']:,} levels | Items: {corpus_stats['items_count']:,}")
            ])
            
            # Create placeholder content for tabs
            loading_content = html.Div([
                html.Div([
                    html.H4("Click 'Apply Filters' to load data", className="text-center mb-3"),
                    html.P("Data will be loaded when you apply filters. This improves initial page load time.", 
                          className="text-center text-muted")
                ], className="p-5")
            ])
            
            # Return placeholder tabs
            updated_tabs = [
                dcc.Tab(label="Documents", children=loading_content),
                dcc.Tab(label="Chunks", children=loading_content),
                dcc.Tab(label="Taxonomy Combinations", children=loading_content),
                dcc.Tab(label="Keywords", children=loading_content),
                dcc.Tab(label="Named Entities", children=loading_content)
            ]
            
            # Return with empty string for the loading spinner output
            return stats_html, updated_tabs, ""
    
        # Create date range tuple if both dates are provided
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
        
        # Fetch filtered data
        taxonomy_data = fetch_taxonomy_combinations(lang_val, db_val, source_type, date_range)
        chunks_data = fetch_chunks_data(lang_val, db_val, source_type, date_range)
        documents_data = fetch_documents_data(lang_val, db_val, source_type, date_range)
        keywords_data = fetch_keywords_data(lang_val, db_val, source_type, date_range)
        named_entities_data = fetch_named_entities_data(lang_val, db_val, source_type, date_range)
        
        # Get time series data
        doc_time_series = fetch_time_series_data('document', lang_val, db_val, source_type, date_range)
        chunk_time_series = fetch_time_series_data('chunk', lang_val, db_val, source_type, date_range)
        taxonomy_time_series = fetch_time_series_data('taxonomy', lang_val, db_val, source_type, date_range)
        keyword_time_series = fetch_time_series_data('keyword', lang_val, db_val, source_type, date_range)
        entity_time_series = fetch_time_series_data('entity', lang_val, db_val, source_type, date_range)
        
        # Get language time series data
        doc_lang_time_series = fetch_language_time_series('document', lang_val, db_val, source_type, date_range)
        chunk_lang_time_series = fetch_language_time_series('chunk', lang_val, db_val, source_type, date_range)
        taxonomy_lang_time_series = fetch_language_time_series('taxonomy', lang_val, db_val, source_type, date_range)
        keyword_lang_time_series = fetch_language_time_series('keyword', lang_val, db_val, source_type, date_range)
        entity_lang_time_series = fetch_language_time_series('entity', lang_val, db_val, source_type, date_range)
        
        # Get database time series data
        doc_db_time_series = fetch_database_time_series('document', lang_val, db_val, source_type, date_range)
        chunk_db_time_series = fetch_database_time_series('chunk', lang_val, db_val, source_type, date_range)
        taxonomy_db_time_series = fetch_database_time_series('taxonomy', lang_val, db_val, source_type, date_range)
        keyword_db_time_series = fetch_database_time_series('keyword', lang_val, db_val, source_type, date_range)
        entity_db_time_series = fetch_database_time_series('entity', lang_val, db_val, source_type, date_range)
        
        # Get database breakdown data with filters
        doc_db_breakdown = fetch_database_breakdown('document', lang_val, db_val, source_type, date_range)
        chunk_db_breakdown = fetch_database_breakdown('chunk', lang_val, db_val, source_type, date_range)
        keyword_db_breakdown = fetch_database_breakdown('keyword', lang_val, db_val, source_type, date_range)
        entity_db_breakdown = fetch_database_breakdown('entity', lang_val, db_val, source_type, date_range)
        
        # Create updated tabs
        documents_subtab = create_documents_tab(
            documents_data, 
            doc_time_series,
            doc_lang_time_series,
            doc_db_time_series,
            doc_db_breakdown
        )
        
        chunks_subtab = create_chunks_tab(
            chunks_data,
            chunk_time_series,
            chunk_lang_time_series,
            chunk_db_time_series,
            chunk_db_breakdown
        )
        
        taxonomy_subtab = create_taxonomy_combinations_tab(
            taxonomy_data,
            taxonomy_time_series,
            taxonomy_lang_time_series,
            taxonomy_db_time_series
        )
        
        keywords_subtab = create_keywords_tab(
            keywords_data,
            keyword_time_series,
            keyword_lang_time_series,
            keyword_db_time_series,
            keyword_db_breakdown
        )
        
        entities_subtab = create_named_entities_tab(
            named_entities_data,
            entity_time_series,
            entity_lang_time_series,
            entity_db_time_series,
            entity_db_breakdown
        )
        
        updated_tabs = [
            dcc.Tab(label="Documents", children=documents_subtab),
            dcc.Tab(label="Chunks", children=chunks_subtab),
            dcc.Tab(label="Taxonomy Combinations", children=taxonomy_subtab),
            dcc.Tab(label="Keywords", children=keywords_subtab),
            dcc.Tab(label="Named Entities", children=entities_subtab)
        ]
        
        # Return with empty string for the loading spinner output
        return stats_html, updated_tabs, ""