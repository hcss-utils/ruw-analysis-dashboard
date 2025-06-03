#!/usr/bin/env python
# coding: utf-8

"""
Explore tab layout and callbacks for the dashboard.
This tab provides exploration of the data via a sunburst chart and detailed view of text chunks.
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
from database.data_fetchers import fetch_category_data, fetch_text_chunks, fetch_timeline_data
from components.layout import create_filter_card, create_pagination_controls, create_download_buttons
from utils.helpers import format_chunk_row, get_unique_filename
from visualizations.sunburst import create_sunburst_chart
from visualizations.timeline import create_timeline_chart


def create_explore_tab_layout(db_options: List, min_date: datetime = None, max_date: datetime = None) -> html.Div:
    """
    Create the Explore tab layout.
    
    Args:
        db_options: Database options for filters
        min_date: Minimum date for filters
        max_date: Maximum date for filters
        
    Returns:
        html.Div: Explore tab layout
    """
    # Try to get initial data for the sunburst chart
    try:
        df_init = fetch_category_data()
        if df_init is None or df_init.empty:
            df_init = pd.DataFrame(columns=['category', 'subcategory', 'sub_subcategory', 'count'])
    except Exception as e:
        logging.error(f"Error fetching initial data for Explore tab: {e}")
        df_init = pd.DataFrame(columns=['category', 'subcategory', 'sub_subcategory', 'count'])

    # Create initial sunburst chart with updated title
    fig_init = create_sunburst_chart(df_init, title="Taxonomic Element Distribution")
    
    # Create pagination controls
    pagination_controls = create_pagination_controls("explore")
    
    # Define the blue color for consistency
    blue_color = "#13376f"  # Dark blue from the color picker
    
    # Create tab layout
    explore_tab_layout = html.Div([
        # Fixed Back to Text Chunks button
        html.A(
            html.Button(
                "↑ Back to Text Chunks", 
                id="back-to-chunks-btn",
                style={
                    "position": "fixed", 
                    "bottom": "20px", 
                    "right": "20px", 
                    "z-index": "9999",
                    "background-color": "#13376f",  # Using the blue_color variable
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
            href="#chunks-selection-title-container"
        ),
        
        # Header with title and About button
        html.Div([
            html.H3("Explore Data", style={"display": "inline-block", "margin-right": "20px"}),
            dbc.Button(
                "About", 
                id="open-about-explore", 
                color="secondary", 
                size="sm",
                className="ml-auto about-button",
                style={"display": "inline-block"}
            ),
        ], style={"display": "flex", "align-items": "center", "margin-bottom": "20px"}),
        
        # Anchor for scrolling to top
        html.Div(id='top'),

        # Filter card
        create_filter_card(
            id_prefix="explore",
            db_options=db_options,
            min_date=min_date,
            max_date=max_date
        ),

        # Sunburst Chart - centered with minimal bottom margin to reduce gap
        html.Div([
            dcc.Loading(
                id="loading-sunburst",
                type="default",  # This will use the global spinner styling
                children=[
                    html.Div(
                        dcc.Graph(
                            id='sunburst-chart', 
                            figure=fig_init, 
                            style={'height': '700px', 'width': '100%', 'max-width': '900px', 'margin': '0 auto'}
                        ),
                        className="sunburst-container"
                    )
                ]
            )
        ], 
        style={'margin-bottom': '0px', 'width': '100%', 'display': 'flex', 'justify-content': 'center'},
        className="sunburst-chart-container"),


        # Timeline chart
        html.Div(id='timeline-container', children=[
            dcc.Loading(
                id="loading-timeline", 
                type="default",
                children=[dcc.Graph(id='timeline-chart', style={'margin-bottom': '10px'})]
            )
        ], style={'display': 'none'}),

        # Timeline caption (sticky)
        html.Div(id='timeline-caption', style={
            'position': 'sticky',
            'top': 0,
            'background': 'white',
            'zIndex': 200,
            'borderBottom': f'2px solid {blue_color}',
            'padding': '10px',
            'display': 'none',
            'text-align': 'center'
        }, className="timeline-caption"),
        
        # Move selection-title-container here to position the spinner right underneath it
        html.Div(
            id='chunks-selection-title-container',
            style={
                'margin-bottom': '10px',
                'scroll-margin-top': '150px'  # Provides space when scrolling to this element
            },
        ),
        html.Div(id='chunks-selection-stats', style={'margin-bottom': '15px', 'text-align': 'center'}),

        # Loading spinner for chunks - use circle type for radar pulse effect
        dcc.Loading(
            id="loading-chunks",
            type="circle",  # Use circle type which we style as radar pulse
            children=[
                # Pagination controls (top)
                pagination_controls['top'],
                
                # Text chunks container
                html.Div(id='text-chunks-container'),
                
                # Pagination controls (bottom)
                pagination_controls['bottom'],
            ],
            color=blue_color,  # Use the same blue color for the spinner
            className="radar-loading"  # Add class for custom styling
        ),

        # Download buttons
        create_download_buttons("explore"),
        

        # Storage components
        dcc.Store(id='filtered-chunks-store'),
        dcc.Store(id='current-page-store', data=0),
        dcc.Store(id='current-selection-store'),  # Store selected level and value
        
        # Hidden div for scrolling
        html.Div(id='dummy-scroll-div', style={'display': 'none'}),
        
        # Create a special modal just for the Explore tab About section
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Using the Sunburst Chart"), 
                           style={"background-color": blue_color, "color": "white"}),
            dbc.ModalBody([
                html.P([
                    "The sunburst chart displays a hierarchical view of taxonomic elements related to the Russian-Ukrainian War, organized in three levels:"
                ]),
                html.Ul([
                    html.Li("Inner ring: Top-level categories"),
                    html.Li("Middle ring: Subcategories"),
                    html.Li("Outer ring: Sub-subcategories")
                ]),
                html.P([
                    html.Strong("Interactive features:"),
                ]),
                html.Ul([
                    html.Li([
                        html.Strong("Hover over segments:"), 
                        " View details including the element name, count, and percentage of the total"
                    ]),
                    html.Li([
                        html.Strong("Click on segments:"), 
                        " Select a taxonomic element to fetch and display related text chunks from the database. ",
                        html.Em("Please wait for the loading spinner to complete, as this may take a moment.")
                    ]),
                    html.Li([
                        html.Strong("Double-click:"), 
                        " Reset the chart to its initial state"
                    ])
                ]),
                html.P([
                    "After selecting a segment, matching text chunks will appear below the chart. Use the pagination controls to browse through all results."
                ])
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-about-explore", className="ms-auto", 
                          style={"background-color": blue_color, "border": "none"})
            ),
        ], id="explore-about-modal", size="lg", is_open=False)
    ], className="dashboard-container", style={'max-width': '100%', 'margin': 'auto'})
    
    return explore_tab_layout


# Register callbacks for the Explore tab
def register_explore_callbacks(app):
    """
    Register callbacks for the Explore tab.
    
    Args:
        app: Dash application instance
    """
    # Callback to update the sunburst chart based on filters
    @app.callback(
        [
            Output('sunburst-chart', 'figure'),
            Output('explore-result-stats', 'children')
        ],
        Input('explore-filter-button', 'n_clicks'),
        [
            State('explore-language-dropdown', 'value'),
            State('explore-database-dropdown', 'value'),
            State('explore-source-type-dropdown', 'value'),
            State('explore-date-range-picker', 'start_date'),
            State('explore-date-range-picker', 'end_date')
        ]
    )
    def update_sunburst(n_clicks, lang_val, db_val, source_type, start_date, end_date):
        """
        Update the sunburst chart based on filter selections.
        
        Args:
            n_clicks: Number of button clicks
            lang_val: Selected language
            db_val: Selected database
            source_type: Selected source type
            start_date: Start date
            end_date: End date
            
        Returns:
            tuple: (fig, stats_html)
        """
        if n_clicks is None:
            # Initial load - use default filters
            df = fetch_category_data()
            if df.empty:
                return go.Figure().update_layout(title="No data available"), "No data found."
        else:
            # Apply filters
            date_range = None
            if start_date and end_date:
                date_range = (start_date, end_date)
            
            df = fetch_category_data(lang_val, db_val, source_type, date_range)
            
            if df.empty:
                return go.Figure().update_layout(title="No data available for selected filters"), "No data found with selected filters."
        
        # Create the sunburst chart with updated title
        fig = create_sunburst_chart(df, title="Taxonomic Element Distribution")
        
        # Return an empty div instead of duplicate stats
        stats = html.Div([])
        
        return fig, stats
    
    # Callback to handle sunburst click and fetch text chunks
    @app.callback(
        [
            Output('filtered-chunks-store', 'data'),
            Output('current-selection-store', 'data'),
            Output('timeline-chart', 'figure'),
            Output('timeline-container', 'style'),
            Output('explore-top-pagination-controls', 'style'),
            Output('explore-bottom-pagination-controls', 'style'),
            Output('explore-download-buttons', 'style'),
            Output('timeline-caption', 'children'),
            Output('timeline-caption', 'style')
        ],
        Input('sunburst-chart', 'clickData'),
        [
            State('explore-language-dropdown', 'value'),
            State('explore-database-dropdown', 'value'),
            State('explore-source-type-dropdown', 'value'),
            State('explore-date-range-picker', 'start_date'),
            State('explore-date-range-picker', 'end_date')
        ]
    )
    def update_on_sunburst_click(clickData, lang_val, db_val, source_type, start_date, end_date):
        """
        Handle clicks on the sunburst chart to display relevant data.
        
        Args:
            clickData: Click data from sunburst chart
            lang_val: Selected language
            db_val: Selected database
            source_type: Selected source type
            start_date: Start date
            end_date: End date
            
        Returns:
            tuple: Multiple outputs for updating the UI
        """
        # Placeholder response for no click
        empty_response = (
            [], None, {}, {'display': 'none'}, {'display': 'none'}, 
            {'display': 'none'}, {'display': 'none'}, "", {'display': 'none'}
        )
        
        if not clickData:
            return empty_response
        
        # Extract selected taxonomic element from clickData
        selected = clickData['points'][0]['label']
        logging.info(f"Selected: {selected}")
        
        # Determine level (category, subcategory, or sub_subcategory)
        temp_df = fetch_category_data(lang_val, db_val, source_type)
        
        if selected in temp_df['category'].unique():
            level = 'category'
        elif selected in temp_df['subcategory'].unique():
            level = 'subcategory'
        elif selected in temp_df['sub_subcategory'].unique():
            level = 'sub_subcategory'
        else:
            logging.error(f"Invalid selection: {selected}")
            return empty_response
        
        # Apply filters
        date_range = None
        if start_date and end_date:
            date_range = (start_date, end_date)
        
        # Fetch text chunks for the selected taxonomic element
        # First get the total count for pagination
        from database.data_fetchers import fetch_text_chunks_count
        total_count = fetch_text_chunks_count(level, selected, lang_val, db_val, source_type, date_range)
        
        # Then fetch only the first page of chunks
        chunks_df = fetch_text_chunks(level, selected, lang_val, db_val, source_type, date_range, page=1, page_size=10)
        
        if chunks_df.empty:
            return [], None, {}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, "No data available for this selection", {'display': 'block'}
        
        # Fetch timeline data
        timeline_df = fetch_timeline_data(level, selected, lang_val, db_val, source_type, date_range)
        timeline_fig = create_timeline_chart(timeline_df, title=f"Timeline for {level}: {selected}")
        
        # Create timeline caption
        timeline_caption = html.Div([
            html.H5(f"Timeline Distribution for {level}: {selected}"),
            html.P(f"Total chunks: {total_count:,}, Time range: {timeline_df['month'].min().strftime('%Y-%m') if not timeline_df.empty else 'N/A'} to {timeline_df['month'].max().strftime('%Y-%m') if not timeline_df.empty else 'N/A'}")
        ])
        
        # Store current selection level, value, and total count
        selection_data = {
            'level': level, 
            'value': selected,
            'total_count': total_count,
            'lang_val': lang_val,
            'db_val': db_val,
            'source_type': source_type,
            'start_date': start_date,
            'end_date': end_date
        }
        
        # Show all components
        show_style = {'display': 'block'}
        pagination_style = {'margin-bottom': '20px', 'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}
        
        logging.info(f"Loaded {len(chunks_df)} chunks for {level}={selected}")
        return chunks_df.to_dict('records'), selection_data, timeline_fig, show_style, pagination_style, pagination_style, show_style, timeline_caption, show_style

    # Callback for pagination and displaying text chunks
    @app.callback(
        [
            Output('chunks-selection-title-container', 'children'),
            Output('chunks-selection-stats', 'children'),
            Output('text-chunks-container', 'children'),
            Output('explore-page-indicator', 'children'),
            Output('explore-page-indicator-bottom', 'children'),
            Output('current-page-store', 'data')
        ],
        [
            Input('filtered-chunks-store', 'data'),
            Input('explore-prev-page-button', 'n_clicks'),
            Input('explore-next-page-button', 'n_clicks'),
            Input('explore-prev-page-button-bottom', 'n_clicks'),
            Input('explore-next-page-button-bottom', 'n_clicks')
        ],
        [
            State('current-selection-store', 'data'),
            State('current-page-store', 'data')
        ]
    )
    def update_selection_and_pagination(
        filtered_data,
        prev_clicks_top,
        next_clicks_top,
        prev_clicks_bottom,
        next_clicks_bottom,
        selection_data,
        current_page
    ):
        """
        Update the selection display and handle pagination.
        
        Args:
            filtered_data: Filtered chunks data
            prev_clicks_top: Previous page button clicks (top)
            next_clicks_top: Next page button clicks (top)
            prev_clicks_bottom: Previous page button clicks (bottom)
            next_clicks_bottom: Next page button clicks (bottom)
            selection_data: Current selection data
            current_page: Current page number
            
        Returns:
            tuple: Updated UI components
        """
        PAGE_SIZE = 10

        if not selection_data or not filtered_data:
            title_placeholder = html.H4("Select a taxonomic element", id="text-chunks-anchor", style={"scroll-margin-top": "100px"})
            stats_placeholder = html.Div("No taxonomic element selected")
            return (title_placeholder, 
                    stats_placeholder, 
                    [],
                    "Page 1", 
                    "Page 1", 
                    0)

        full_df = pd.DataFrame(filtered_data)
        if full_df.empty:
            title_empty = html.H4("No data for selection", id="text-chunks-anchor", style={"scroll-margin-top": "100px"})
            stats_empty = html.Div("No chunks available")
            return (title_empty, 
                    stats_empty, 
                    [],
                    "Page 1", 
                    "Page 1", 
                    0)

        # Get selected taxonomic element info
        level = selection_data.get('level', 'unknown')
        selected = selection_data.get('value', 'unknown')
        
        # Handle pagination
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        # Get total count from selection data
        total_count = selection_data.get('total_count', len(full_df))
        
        # If the data changed (new selection), reset page
        if triggered_id == 'filtered-chunks-store':
            current_page = 0
        else:
            # Check which pagination button was clicked
            if triggered_id in ['explore-prev-page-button', 'explore-prev-page-button-bottom']:
                current_page = max(current_page - 1, 0)
            elif triggered_id in ['explore-next-page-button', 'explore-next-page-button-bottom']:
                max_page = (total_count - 1) // PAGE_SIZE
                current_page = min(current_page + 1, max_page)
        
        # Fetch the specific page of data from the database
        lang_val = selection_data.get('lang_val')
        db_val = selection_data.get('db_val')
        source_type = selection_data.get('source_type')
        start_date = selection_data.get('start_date')
        end_date = selection_data.get('end_date')
        date_range = (start_date, end_date) if start_date and end_date else None
        
        # Fetch only the current page of chunks
        page_df = fetch_text_chunks(
            level, selected, lang_val, db_val, source_type, date_range,
            page=current_page + 1,  # Pages are 1-indexed in the database
            page_size=PAGE_SIZE
        )
        
        # Calculate indices for display
        start_idx = current_page * PAGE_SIZE
        end_idx = min(start_idx + len(page_df), total_count)
        
        # Create title with more emphasis - use the consistent styling from the screenshot
        # This is where the Back to Text Chunks should go - at the top of the loaded chunks
        title = html.H4(f"{level.capitalize()}: {selected}", id="text-chunks-anchor", className="selection-title", style={"scroll-margin-top": "100px"})
        
        # Create stats with better formatting
        stats = html.Div([
            html.P(f"Total chunks: {total_count:,}", style={"font-weight": "bold"}),
            html.P(f"Showing chunks {start_idx+1}-{end_idx} of {total_count:,}")
        ])
        
        # Format chunks for display
        chunk_rows = []
        for i, row in page_df.iterrows():
            try:
                chunk_row = format_chunk_row(row)
                chunk_rows.append(chunk_row)
            except Exception as e:
                logging.error(f"Error formatting chunk {i}: {e}")
                chunk_rows.append(html.Div(f"Error displaying chunk {i}"))
        
        # Create pagination text
        total_pages = (total_count - 1) // PAGE_SIZE + 1
        page_text = f"Page {current_page + 1} of {total_pages}"
        
        return title, stats, chunk_rows, page_text, page_text, current_page
    
    # Callback to toggle the Explore About modal
    @app.callback(
        Output("explore-about-modal", "is_open"),
        [Input("open-about-explore", "n_clicks"), Input("close-about-explore", "n_clicks")],
        [State("explore-about-modal", "is_open")],
        prevent_initial_call=True
    )
    def toggle_explore_about_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open

    # Callback for CSV download
    @app.callback(
        Output("download-dataframe-csv", "data"),
        Input("explore-btn-csv", "n_clicks"),
        State('filtered-chunks-store', 'data'),
        prevent_initial_call=True
    )
    def download_csv(n_clicks, filtered_data):
        """
        Handle CSV download.
        
        Args:
            n_clicks: Button clicks
            filtered_data: Filtered data to download
            
        Returns:
            dict: Download data
        """
        if not filtered_data:
            return dash.no_update
        
        df = pd.DataFrame(filtered_data)
        filename = get_unique_filename("data_export.csv")
        
        # Create a string buffer, write the CSV, and encode as bytes
        from io import StringIO
        str_buffer = StringIO()
        df.to_csv(str_buffer, index=False)
        
        # Use send_string instead of send_data_frame
        return dcc.send_string(str_buffer.getvalue(), filename)

    # Callback for JSON download
    @app.callback(
        Output("download-dataframe-json", "data"),
        Input("explore-btn-json", "n_clicks"),
        State('filtered-chunks-store', 'data'),
        prevent_initial_call=True
    )
    def download_json(n_clicks, filtered_data):
        """
        Handle JSON download.
        
        Args:
            n_clicks: Button clicks
            filtered_data: Filtered data to download
            
        Returns:
            dict: Download data
        """
        if not filtered_data:
            return dash.no_update
        
        filename = get_unique_filename("data_export.json")
        
        # Convert to JSON string
        import json
        json_str = json.dumps(filtered_data)
        
        # Use send_string for JSON
        return dcc.send_string(json_str, filename)