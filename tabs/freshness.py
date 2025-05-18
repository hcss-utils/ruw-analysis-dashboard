#!/usr/bin/env python
# coding: utf-8

"""
Freshness tab layout and callbacks for the dashboard.
This tab provides analysis of how recently different taxonomic elements have been discussed.
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
from components.cards import create_freshness_explanation_card
from config import FRESHNESS_PERIOD_OPTIONS, FRESHNESS_FILTER_OPTIONS


def create_freshness_tab_layout() -> html.Div:
    """
    Create the Freshness tab layout.
    
    Returns:
        html.Div: Freshness tab layout
    """
    freshness_tab_layout = html.Div([
        # Header with title and About button
        html.Div([
            html.H3("Taxonomic Freshness", style={"display": "inline-block", "margin-right": "20px"}),
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
                            options=FRESHNESS_PERIOD_OPTIONS,
                            value='week',
                            inline=True
                        ),
                    ], width=6),
                    dbc.Col([
                        html.Label("Select Filter:"),
                        dcc.Dropdown(
                            id='freshness-filter',
                            options=FRESHNESS_FILTER_OPTIONS,
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
        create_freshness_explanation_card(),
        
        # Freshness-specific About modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Using the Freshness Tab"), 
                           style={"background-color": "#13376f", "color": "white"}),
            dbc.ModalBody([
                html.P([
                    "The Freshness tab helps you identify which topics are currently 'hot' in the discourse around the Russian-Ukrainian War."
                ]),
                
                html.H5("How to Use This Tab:", style={"margin-top": "20px", "color": "#13376f"}),
                html.Ol([
                    html.Li([
                        html.Strong("Select Time Period:"), 
                        " Choose whether to analyze freshness over the last week, month, or quarter."
                    ]),
                    html.Li([
                        html.Strong("Select Filter:"), 
                        " Choose a subset of sources to analyze (All, Russian, Ukrainian, Western, etc.)."
                    ]),
                    html.Li([
                        html.Strong("Click Analyze Freshness:"), 
                        " Generate the freshness analysis."
                    ]),
                    html.Li([
                        html.Strong("Explore Visualizations:"), 
                        " Switch between the different visualization tabs to understand freshness from different perspectives."
                    ]),
                ]),
                
                html.H5("Understanding the Visualizations:", style={"margin-top": "20px", "color": "#13376f"}),
                html.Ul([
                    html.Li([
                        html.Strong("Freshness Overview:"), 
                        " Bar chart showing the freshness score of each taxonomic element."
                    ]),
                    html.Li([
                        html.Strong("Freshness Timeline:"), 
                        " Scatter plot showing when each taxonomic element was last mentioned, with size indicating frequency and color indicating freshness."
                    ]),
                    html.Li([
                        html.Strong("Category Drill-Down:"), 
                        " Detailed analysis of freshness for subcategories within a selected category."
                    ]),
                ]),
                
                html.H5("Interpreting Freshness Scores:", style={"margin-top": "20px", "color": "#13376f"}),
                html.Ul([
                    html.Li([
                        html.Strong("High Freshness (Green):"), 
                        " Topics that are being actively discussed in recent content."
                    ]),
                    html.Li([
                        html.Strong("Medium Freshness (Yellow):"), 
                        " Topics that have moderate activity or were prominent recently but are fading."
                    ]),
                    html.Li([
                        html.Strong("Low Freshness (Red):"), 
                        " Topics that haven't been discussed much recently."
                    ]),
                ]),
                
                html.P([
                    html.Strong("Tip:"), 
                    " Compare freshness scores across different time periods to identify emerging trends and fading topics."
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
    @app.callback(
        [
            Output('freshness-results', 'style'),
            Output('freshness-chart', 'figure'),
            Output('freshness-timeline-chart', 'figure'),
            Output('freshness-category-dropdown', 'options'),
            Output('freshness-category-dropdown', 'value')
        ],
        Input('freshness-button', 'n_clicks'),
        [
            State('freshness-period', 'value'),
            State('freshness-filter', 'value')
        ]
    )
    def update_freshness_analysis(n_clicks, period, filter_value):
        """
        Update freshness analysis based on selected period and filter.
        
        Args:
            n_clicks: Number of button clicks
            period: Selected time period
            filter_value: Selected filter value
            
        Returns:
            tuple: Multiple outputs for updating freshness visualizations
        """
        # Import here to avoid circular imports
        from visualizations.timeline import create_timeline_chart
        
        logging.info(f"Updating freshness analysis for period: {period}, filter: {filter_value}")
        
        if not n_clicks:
            # Initial state - no analysis yet
            empty_fig = go.Figure().update_layout(title="No data to display yet")
            return {'display': 'none'}, empty_fig, empty_fig, [], None
        
        # Calculate date range based on selected period
        end_date = datetime.now()
        if period == 'week':
            start_date = end_date - timedelta(days=7)
        elif period == 'month':
            start_date = end_date - timedelta(days=30)
        else:  # quarter
            start_date = end_date - timedelta(days=90)
        
        # Get freshness data
        data = get_freshness_data(start_date, end_date, filter_value)
        
        if 'category' not in data or data['category'].empty:
            empty_fig = go.Figure().update_layout(title="No data available for selected filters")
            return {'display': 'block'}, empty_fig, empty_fig, [], None
        
        # Create main freshness bar chart
        category_df = data['category']
        
        # Default bar chart for freshness
        main_fig = go.Figure()
        
        # Sort by freshness score
        sorted_df = category_df.sort_values('freshness_score', ascending=False)
        
        main_fig.add_trace(go.Bar(
            x=sorted_df['category'],
            y=sorted_df['freshness_score'],
            marker_color=sorted_df['freshness_score'],
            marker_colorscale=[[0, '#FFEB3B'], [1, '#8BC34A']],  # Yellow to Green
            text=sorted_df['freshness_score'].round(1),
            textposition='outside'
        ))
        
        main_fig.update_layout(
            title=f"Taxonomic Element Freshness ({period})",
            xaxis_tickangle=-45,
            xaxis_title='',
            yaxis_title='Freshness Score (0-100)',
            height=600,
            margin=dict(b=150)  # Extra bottom margin for rotated labels
        )
        
        # Create timeline figure
        timeline_fig = create_timeline_chart(
            category_df.rename(columns={'latest_date': 'month', 'freshness_score': 'count'}),
            title=f"Most Recent Taxonomic Elements ({period})"
        )
        
        # Create dropdown options for drill-down
        category_options = [{'label': cat, 'value': cat} for cat in category_df['category']]
        
        return {'display': 'block'}, main_fig, timeline_fig, category_options, None

    @app.callback(
        Output('freshness-drilldown-chart', 'figure'),
        [
            Input('freshness-category-dropdown', 'value'),
            Input('freshness-button', 'n_clicks')
        ],
        [
            State('freshness-period', 'value'),
            State('freshness-filter', 'value')
        ],
        prevent_initial_call=True
    )
    def update_drilldown_chart(selected_category, n_clicks, period, filter_value):
        """
        Update drill-down chart for selected category.
        
        Args:
            selected_category: Selected category for drill-down
            n_clicks: Number of button clicks
            period: Selected time period
            filter_value: Selected filter value
            
        Returns:
            go.Figure: Drill-down chart
        """
        logging.info(f"Updating drill-down for category: {selected_category}")
        
        if not n_clicks or not selected_category:
            return go.Figure().update_layout(title="Select a category from the dropdown to see subcategories")
        
        # Calculate date range based on selected period
        end_date = datetime.now()
        if period == 'week':
            start_date = end_date - timedelta(days=7)
        elif period == 'month':
            start_date = end_date - timedelta(days=30)
        else:  # quarter
            start_date = end_date - timedelta(days=90)
        
        # Get freshness data
        data = get_freshness_data(start_date, end_date, filter_value)
        
        if 'subcategory' not in data or data['subcategory'].empty:
            return go.Figure().update_layout(title="No subcategory data available")
        
        # Filter for the selected category
        subcategory_df = data['subcategory']
        filtered_df = subcategory_df[subcategory_df['category'] == selected_category]
        
        if filtered_df.empty:
            return go.Figure().update_layout(title=f"No subcategories found for '{selected_category}'")
        
        # Sort by freshness score
        sorted_df = filtered_df.sort_values('freshness_score', ascending=False)
        
        # Create the bar chart for subcategories
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=sorted_df['subcategory'],
            y=sorted_df['freshness_score'],
            marker_color=sorted_df['freshness_score'],
            marker_colorscale=[[0, '#FFEB3B'], [1, '#8BC34A']],  # Yellow to Green
            text=sorted_df['freshness_score'].round(1),
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f"Freshness for Subcategories of '{selected_category}'",
            xaxis_tickangle=-45,
            xaxis_title='',
            yaxis_title='Freshness Score (0-100)',
            height=600,
            margin=dict(b=150)  # Extra bottom margin for rotated labels
        )
        
        return fig