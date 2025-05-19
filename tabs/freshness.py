#!/usr/bin/env python
# coding: utf-8

"""
Freshness tab layout and callbacks for the dashboard.
This tab provides analysis of how recently different taxonomic elements, keywords, and named entities 
have been discussed, with burst detection to identify surges in discussion.
"""

import logging
from datetime import datetime, timedelta
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
from database.data_fetchers import get_freshness_data
from database.data_fetchers_freshness import get_burst_data_for_periods, calculate_burst_summaries
from components.cards import create_freshness_explanation_card
from config import FRESHNESS_PERIOD_OPTIONS, FRESHNESS_FILTER_OPTIONS, FRESHNESS_DATATYPE_OPTIONS
from visualizations.timeline import create_timeline_chart, create_freshness_timeline
from visualizations.bursts import (
    create_burst_heatmap, 
    create_burst_summary_chart,
    create_burst_timeline,
    create_burst_comparison_chart
)


def create_freshness_tab_layout() -> html.Div:
    """
    Create the Freshness tab layout with expanded capabilities for multiple data types.
    
    Returns:
        html.Div: Freshness tab layout
    """
    freshness_tab_layout = html.Div([
        # Header with title and About button
        html.Div([
            html.H3("Content Freshness & Burst Analysis", style={"display": "inline-block", "margin-right": "20px"}),
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
            dbc.CardHeader("Freshness & Burst Analysis Controls"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Time Period:"),
                        dcc.RadioItems(
                            id='freshness-period',
                            options=FRESHNESS_PERIOD_OPTIONS,
                            value='month',
                            inline=True
                        ),
                    ], width=4),
                    dbc.Col([
                        html.Label("Select Filter:"),
                        dcc.Dropdown(
                            id='freshness-filter',
                            options=FRESHNESS_FILTER_OPTIONS,
                            value='all'
                        ),
                    ], width=4),
                    dbc.Col([
                        html.Label("Select Data Type:"),
                        dcc.Checklist(
                            id='freshness-data-types',
                            options=FRESHNESS_DATATYPE_OPTIONS,
                            value=['taxonomy', 'keywords', 'named_entities'],
                            inline=True
                        ),
                    ], width=4)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Br(),
                        dbc.Button('Run Analysis', id='freshness-button', color="primary"),
                    ], width=12, style={"text-align": "center"})
                ], className="mt-3")
            ])
        ], className="mb-4"),
        
        # Results section
        html.Div([
            dbc.Tabs([
                # Traditional Freshness Tab
                dbc.Tab([
                    dcc.Loading(
                        id="loading-freshness-chart",
                        type="default",
                        children=[dcc.Graph(id='freshness-chart', style={'height': '700px'})]
                    )
                ], label="Taxonomic Freshness"),
                
                # Burst Analysis Overview Tab
                dbc.Tab([
                    dcc.Loading(
                        id="loading-burst-comparison",
                        type="default",
                        children=[
                            html.Div([
                                html.H5("Top Bursting Elements", className="mt-3"),
                                dcc.Graph(id='burst-comparison-chart', style={'height': '500px'})
                            ])
                        ]
                    )
                ], label="Burst Comparison"),
                
                # Taxonomy Burst Analysis Tab
                dbc.Tab([
                    dcc.Loading(
                        id="loading-taxonomy-burst",
                        type="default",
                        children=[
                            html.Div([
                                html.H5("Taxonomy Element Bursts", className="mt-3"),
                                dcc.Graph(id='taxonomy-burst-chart', style={'height': '600px'}),
                                html.H5("Burst Timeline", className="mt-4"),
                                dcc.Graph(id='taxonomy-burst-timeline', style={'height': '500px'})
                            ])
                        ]
                    )
                ], label="Taxonomy Bursts"),
                
                # Keyword Burst Analysis Tab
                dbc.Tab([
                    dcc.Loading(
                        id="loading-keyword-burst",
                        type="default",
                        children=[
                            html.Div([
                                html.H5("Keyword Bursts", className="mt-3"),
                                dcc.Graph(id='keyword-burst-chart', style={'height': '600px'}),
                                html.H5("Burst Timeline", className="mt-4"),
                                dcc.Graph(id='keyword-burst-timeline', style={'height': '500px'})
                            ])
                        ]
                    )
                ], label="Keyword Bursts"),
                
                # Named Entity Burst Analysis Tab
                dbc.Tab([
                    dcc.Loading(
                        id="loading-entity-burst",
                        type="default",
                        children=[
                            html.Div([
                                html.H5("Named Entity Bursts", className="mt-3"),
                                dcc.Graph(id='entity-burst-chart', style={'height': '600px'}),
                                html.H5("Burst Timeline", className="mt-4"),
                                dcc.Graph(id='entity-burst-timeline', style={'height': '500px'})
                            ])
                        ]
                    )
                ], label="Entity Bursts")
            ], id="freshness-tabs")
        ], id="freshness-results", style={"display": "none"}),
        
        # Explanation card
        create_freshness_explanation_card(),
        
        # Freshness-specific About modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Using the Freshness & Burst Analysis Tab"), 
                           style={"background-color": "#13376f", "color": "white"}),
            dbc.ModalBody([
                html.P([
                    "The Freshness tab helps you identify which topics, keywords, and entities are currently 'hot' in the discourse around the Russian-Ukrainian War."
                ]),
                
                html.H5("How to Use This Tab:", style={"margin-top": "20px", "color": "#13376f"}),
                html.Ol([
                    html.Li([
                        html.Strong("Select Time Period:"), 
                        " Choose whether to analyze data over weeks, months, or quarters."
                    ]),
                    html.Li([
                        html.Strong("Select Filter:"), 
                        " Choose a subset of sources to analyze (All, Russian, Ukrainian, Western, etc.)."
                    ]),
                    html.Li([
                        html.Strong("Select Data Types:"),
                        " Choose which types of data to include in your analysis (taxonomic elements, keywords, named entities)."
                    ]),
                    html.Li([
                        html.Strong("Click Run Analysis:"), 
                        " Generate the freshness and burst analysis."
                    ]),
                    html.Li([
                        html.Strong("Explore Visualization Tabs:"), 
                        " Examine different aspects of the data across the different tabs."
                    ]),
                ]),
                
                html.H5("Understanding the Visualizations:", style={"margin-top": "20px", "color": "#13376f"}),
                html.Ul([
                    html.Li([
                        html.Strong("Taxonomic Freshness:"), 
                        " Traditional freshness analysis for taxonomic elements."
                    ]),
                    html.Li([
                        html.Strong("Burst Comparison:"), 
                        " Compare the top bursting elements across all data types."
                    ]),
                    html.Li([
                        html.Strong("Taxonomy/Keyword/Entity Bursts:"), 
                        " Detailed analysis of bursts for each data type, including heatmaps and timelines."
                    ]),
                ]),
                
                html.H5("About Burst Detection:", style={"margin-top": "20px", "color": "#13376f"}),
                html.P([
                    "Burst detection identifies sudden and significant increases in the frequency of terms or concepts. " +
                    "It helps identify emerging trends, hot topics, and time-sensitive information. " +
                    "The algorithm identifies when the frequency of an element suddenly jumps above its baseline level."
                ]),
                html.Ul([
                    html.Li([
                        html.Strong("High Burst Intensity (Red):"), 
                        " Topics showing significant and sudden increases in discussion."
                    ]),
                    html.Li([
                        html.Strong("Moderate Burst Intensity (Orange/Yellow):"), 
                        " Topics that show notable but less dramatic increases."
                    ]),
                    html.Li([
                        html.Strong("Low/No Burst Intensity (Light):"), 
                        " Topics that maintain steady discussion levels without significant spikes."
                    ]),
                ]),
                
                html.P([
                    html.Strong("Tip:"), 
                    " Compare burst patterns across different data types to identify correlations between concepts, entities, and keywords."
                ], style={"margin-top": "15px", "font-style": "italic"})
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-freshness-about", className="ms-auto", 
                          style={"background-color": "#13376f", "border": "none"})
            ),
        ], id="freshness-about-modal", size="lg", is_open=False)
    ], style={'max-width': '1200px', 'margin': 'auto'})
    
    return freshness_tab_layout


def register_freshness_callbacks(app):
    """
    Register callbacks for the Freshness tab.
    
    Args:
        app: Dash application instance
    """
    # Callback to toggle the Freshness About modal
    @app.callback(
        Output("freshness-about-modal", "is_open"),
        [Input("open-about-freshness", "n_clicks"), Input("close-freshness-about", "n_clicks")],
        [State("freshness-about-modal", "is_open")],
        prevent_initial_call=True
    )
    def toggle_freshness_about_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open
    
    # Main callback for freshness and burst analysis
    @app.callback(
        [
            Output('freshness-results', 'style'),
            # Traditional freshness outputs
            Output('freshness-chart', 'figure'),
            # Burst comparison outputs
            Output('burst-comparison-chart', 'figure'),
            # Taxonomy burst outputs
            Output('taxonomy-burst-chart', 'figure'),
            Output('taxonomy-burst-timeline', 'figure'),
            # Keyword burst outputs
            Output('keyword-burst-chart', 'figure'),
            Output('keyword-burst-timeline', 'figure'),
            # Named entity burst outputs
            Output('entity-burst-chart', 'figure'),
            Output('entity-burst-timeline', 'figure')
        ],
        Input('freshness-button', 'n_clicks'),
        [
            State('freshness-period', 'value'),
            State('freshness-filter', 'value'),
            State('freshness-data-types', 'value')
        ]
    )
    def update_freshness_analysis(n_clicks, period, filter_value, data_types):
        """
        Update freshness and burst analysis based on selected parameters.
        
        Args:
            n_clicks: Number of button clicks
            period: Selected time period
            filter_value: Selected filter value
            data_types: Selected data types
            
        Returns:
            tuple: Multiple outputs for updating visualizations
        """
        logging.info(f"Updating freshness analysis for period: {period}, filter: {filter_value}, data_types: {data_types}")
        
        if not n_clicks:
            # Initial state - no analysis yet
            empty_fig = go.Figure().update_layout(title="No data to display yet")
            return {'display': 'none'}, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
        
        # Calculate date range based on selected period
        end_date = datetime.now()
        if period == 'week':
            start_date = end_date - timedelta(days=7)
            n_periods = 10  # Last 10 weeks
        elif period == 'month':
            start_date = end_date - timedelta(days=30)
            n_periods = 10  # Last 10 months
        else:  # quarter
            start_date = end_date - timedelta(days=90)
            n_periods = 10  # Last 10 quarters
        
        results = {}
        
        # Get traditional freshness data
        if 'taxonomy' in data_types:
            freshness_data = get_freshness_data(start_date, end_date, filter_value)
            
            if 'category' not in freshness_data or freshness_data['category'].empty:
                empty_fig = go.Figure().update_layout(title="No taxonomic data available for selected filters")
                freshness_fig = empty_fig
            else:
                # Create freshness chart
                category_df = freshness_data['category']
                
                # Sort by freshness score
                sorted_df = category_df.sort_values('freshness_score', ascending=False)
                
                freshness_fig = go.Figure()
                
                freshness_fig.add_trace(go.Bar(
                    x=sorted_df['category'],
                    y=sorted_df['freshness_score'],
                    marker_color=sorted_df['freshness_score'],
                    marker_colorscale=[[0, '#FFEB3B'], [1, '#8BC34A']],  # Yellow to Green
                    text=sorted_df['freshness_score'].round(1),
                    textposition='outside'
                ))
                
                freshness_fig.update_layout(
                    title=f"Taxonomic Element Freshness ({period})",
                    xaxis_tickangle=-45,
                    xaxis_title='',
                    yaxis_title='Freshness Score (0-100)',
                    height=600,
                    margin=dict(b=150)  # Extra bottom margin for rotated labels
                )
                
                results['freshness_timeline'] = create_freshness_timeline(
                    category_df,
                    title=f"Taxonomic Element Freshness Timeline ({period})"
                )
        else:
            empty_fig = go.Figure().update_layout(title="Taxonomic elements not selected for analysis")
            freshness_fig = empty_fig
        
        # Get burst data for all selected data types
        burst_data = get_burst_data_for_periods(
            period_type=period,
            n_periods=n_periods,
            filter_value=filter_value,
            data_types=data_types
        )
        
        # Calculate burst summaries
        burst_summaries = calculate_burst_summaries(burst_data)
        
        # Create burst comparison chart
        taxonomy_summary = burst_summaries.get('taxonomy', pd.DataFrame())
        keyword_summary = burst_summaries.get('keywords', pd.DataFrame())
        entity_summary = burst_summaries.get('named_entities', pd.DataFrame())
        
        comparison_fig = create_burst_comparison_chart(
            taxonomy_summary,
            keyword_summary,
            entity_summary,
            title=f"Top Bursting Elements ({period})"
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
                title="Taxonomy Element Burst Intensity by Period"
            )
            
            taxonomy_timeline_fig = create_burst_timeline(
                burst_data['taxonomy'],
                title="Top Taxonomy Elements Burst Timeline"
            )
        else:
            empty_fig = go.Figure().update_layout(title="No taxonomy burst data available or not selected")
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
                title="Top Keywords Burst Timeline"
            )
        else:
            empty_fig = go.Figure().update_layout(title="No keyword burst data available or not selected")
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
                title="Top Named Entities Burst Timeline"
            )
        else:
            empty_fig = go.Figure().update_layout(title="No named entity burst data available or not selected")
            entity_burst_fig = empty_fig
            entity_timeline_fig = empty_fig
        
        return {'display': 'block'}, freshness_fig, comparison_fig, taxonomy_burst_fig, taxonomy_timeline_fig, keyword_burst_fig, keyword_timeline_fig, entity_burst_fig, entity_timeline_fig