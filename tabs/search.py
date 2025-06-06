#!/usr/bin/env python
# coding: utf-8

"""
Search tab layout and callbacks for the dashboard.
This tab provides search functionality across the database with different search modes.
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import dash
from dash import html, dcc, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

import sys
import os

# Add the project root to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.data_fetchers import fetch_search_category_data, fetch_all_text_chunks_for_search
from utils.helpers import format_chunk_row, get_unique_filename, format_number
from visualizations.sunburst import create_sunburst_chart
from visualizations.timeline import create_timeline_chart
from config import SEARCH_RESULT_LIMIT, THEME_COLORS
from components.layout import create_filter_card


def create_search_tab_layout(db_options: List, min_date: datetime = None, max_date: datetime = None) -> html.Div:
    """
    Create the Search tab layout.
    
    Args:
        db_options: Database options for filters
        min_date: Minimum date for filters
        max_date: Maximum date for filters
        
    Returns:
        html.Div: Search tab layout
    """
    # Define the blue color for consistency
    blue_color = "#13376f"  # Dark blue for all UI elements
    
    search_tab_layout = html.Div([
        # Fixed Back to Text Chunks button (same styling as explore tab)
        html.A(
            html.Button(
                "↑ Back to Text Chunks", 
                id="search-back-to-chunks-btn",
                style={
                    "position": "fixed", 
                    "bottom": "20px", 
                    "right": "20px", 
                    "z-index": "9999",
                    "background-color": blue_color,
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
            href="#search-chunks-anchor"
        ),
        
        # Header with title and About button
        html.Div([
            html.H3("Search Data", style={"display": "inline-block", "margin-right": "20px"}),
            dbc.Button(
                "About", 
                id="open-about-search", 
                color="secondary", 
                size="sm",
                className="ml-auto about-button",
                style={"display": "inline-block"}
            ),
        ], style={"display": "flex", "align-items": "center", "margin-bottom": "20px"}),
        
        # Anchor for scrolling to top
        html.Div(id='search-top'),
        
        # Enhanced Search Card with multiple search modes
        dbc.Card([
            dbc.CardHeader("Search Options"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Search Mode:"),
                        dcc.RadioItems(
                            id='search-mode',
                            options=[
                                {'label': 'Keyword Search', 'value': 'keyword'},
                                {'label': 'Boolean Search', 'value': 'boolean'},
                                {'label': 'Semantic Search (Beta)', 'value': 'semantic'}
                            ],
                            value='keyword',
                            style={'display': 'flex', 'justify-content': 'space-between'}
                        ),
                        dbc.Tooltip(
                            "Choose search mode: Keyword (whole word matching), Boolean (AND, OR, NOT operators), or Semantic (meaning-based)",
                            target="search-mode",
                        )
                    ], width=12)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Search Terms:"),
                        dcc.Input(
                            id='search-input',
                            type='text',
                            placeholder='Enter search term...',
                            style={'width': '100%'}
                        ),
                        html.Div(id='search-syntax-help', style={'font-size': '0.8em', 'margin-top': '5px'})
                    ], width=12)
                ], className="mt-3"),
            ])
        ], className="mb-4"),
        
        # Standard filter card that matches explore tab
        create_filter_card(
            id_prefix="search",
            db_options=db_options,
            min_date=min_date,
            max_date=max_date
        ),
        
        # Separate search button
        html.Div([
            dbc.Button('Search', id='search-button', color="primary", className="mt-2", size="lg"),
        ], style={"text-align": "center", "margin-top": "10px", "margin-bottom": "20px"}),
        
        # Results section - initial information area
        html.Div(id='search-stats-container', className="mt-4", style={"scroll-margin-top": "100px"}),
        
        # Loading message overlay container with background and radar pulse
        html.Div([
            # Semi-transparent background overlay
            html.Div(style={
                'position': 'fixed',
                'top': '0',
                'left': '0',
                'width': '100%',
                'height': '100%',
                'background-color': 'rgba(0, 0, 0, 0.3)',
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
                    html.P("🔍 Search in Progress", 
                           className="text-center mt-4", 
                           style={'color': blue_color, 'font-weight': 'bold', 'font-size': '18px'}),
                    html.P(id='search-progress-message', 
                           className="text-muted text-center", 
                           style={'font-size': '14px'}),
                    html.P("💡 Fun fact: If we printed all searchable text, it would reach the moon and back!", 
                           className="text-info text-center mt-3",
                           style={'font-style': 'italic', 'font-size': '13px'})
                ], style={
                    'background': 'rgba(255, 255, 255, 0.98)',
                    'border': '2px solid ' + blue_color,
                    'border-radius': '12px',
                    'padding': '40px',
                    'box-shadow': '0 4px 20px rgba(0, 0, 0, 0.15)',
                    'max-width': '500px',
                    'margin': '0 auto',
                    'position': 'relative',
                    'z-index': '1001'
                })
            ], style={
                'position': 'fixed',
                'top': '50%',
                'left': '50%',
                'transform': 'translate(-50%, -50%)',
                'z-index': '1001'
            })
        ], id="search-loading-messages", 
           style={
               'display': 'none'
           }),
        
        # Sunburst Chart container - initially hidden
        html.Div([
            html.Div(
                dcc.Graph(
                    id='search-sunburst-chart', 
                    style={'height': '700px', 'width': '100%', 'max-width': '900px', 'margin': '0 auto'}
                ),
                className="sunburst-container"
            )
        ], 
        style={'margin-bottom': '0px', 'width': '100%', 'justify-content': 'center', 'display': 'none'},
        className="sunburst-chart-container", id="search-results-tabs"),
        
        # Add loading spinner for segment clicks - positioned right after the sunburst chart
        dcc.Loading(
            id="loading-search-segment", 
            type="default",
            color=blue_color,  # Match the blue color
            children=[
                html.Div(id="search-segment-loading-indicator", style={"height": "10px"})
            ],
            style={"margin-top": "10px", "margin-bottom": "10px"}
        ),
        
        # Timeline section
        html.Div([
            dcc.Loading(
                id="loading-search-timeline",
                type="default",
                color=blue_color,  # Match the blue color
                children=[dcc.Graph(id='search-timeline-chart', style={'height': '400px'})]
            )
        ], id="search-timeline-container", style={'display': 'none'}),
        
        # Store components
        dcc.Store(id='search-results-store'),
        dcc.Store(id='search-current-page-store', data=0),
        
        # Results containers
        html.Div(id='search-results-header', style={'display': 'none'}),
        
        html.Div(
            id='search-chunks-container', 
            style={'scroll-margin-top': '150px'}  # Provides space when scrolling to this element
        ),
        
        # Pagination controls
        html.Div([
            dbc.Button('Previous Page', id='search-prev-page-button', color="secondary", className="me-2"),
            html.Span("Page 1", id='search-page-indicator', style={'margin': '0 10px'}),
            dbc.Button('Next Page', id='search-next-page-button', color="secondary"),
        ], id='search-pagination-controls', style={'margin-top': '20px', 'display': 'none', 'justify-content': 'center', 'align-items': 'center'}),
        
        # Download buttons
        html.Div([
            dbc.Button("Download CSV", id="search-btn-csv", color="success", className="me-2"),
            dbc.Button("Download JSON", id="search-btn-json", color="success"),
        ], id='search-download-buttons', style={'margin-top': '20px', 'text-align': 'center', 'display': 'none'}),
        
        
        # Search-specific About modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Using the Search Tab"), 
                          style={"background-color": "#13376f", "color": "white"}),
            dbc.ModalBody([
                html.P([
                    "The Search tab allows you to find specific content in the dataset using various search methods."
                ]),
                
                html.H5("How to Use This Tab:", style={"margin-top": "20px", "color": "#13376f"}),
                html.Ol([
                    html.Li([
                        html.Strong("Select Search Mode:"), 
                        " Choose between Keyword Search, Boolean Search, or Semantic Search depending on your needs."
                    ]),
                    html.Li([
                        html.Strong("Enter Search Terms:"), 
                        " Type your search terms in the search box."
                    ]),
                    html.Li([
                        html.Strong("Apply Filters:"), 
                        " Narrow down results by language, database, source type, or date range."
                    ]),
                    html.Li([
                        html.Strong("Explore Results:"), 
                        " View matching chunks, browse through the timeline, and analyze category distribution."
                    ]),
                ]),
                
                html.H5("Search Modes:", style={"margin-top": "20px", "color": "#13376f"}),
                html.Ul([
                    html.Li([
                        html.Strong("Keyword Search:"), 
                        " Finds exact matches of your search terms."
                    ]),
                    html.Li([
                        html.Strong("Boolean Search:"), 
                        " Uses operators like AND, OR, NOT for complex queries (e.g., 'Ukraine AND military NOT peace')."
                    ]),
                    html.Li([
                        html.Strong("Semantic Search:"), 
                        " Uses AI to find content related to your search concept, even if it doesn't use the exact words."
                    ]),
                ]),
                
                html.H5("Understanding Results:", style={"margin-top": "20px", "color": "#13376f"}),
                html.Ul([
                    html.Li([
                        html.Strong("Category Distribution:"), 
                        " Shows how search results are distributed across different taxonomic categories."
                    ]),
                    html.Li([
                        html.Strong("Timeline:"), 
                        " Displays when matching documents were published."
                    ]),
                    html.Li([
                        html.Strong("Text Results:"), 
                        " Shows the actual text chunks with your search terms highlighted."
                    ]),
                ]),
                
                html.P([
                    html.Strong("Tip:"), 
                    " For best results with Boolean search, use parentheses to group terms, e.g., 'Russia AND (economy OR sanctions)'."
                ], style={"margin-top": "15px", "font-style": "italic"})
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-search-about", className="ms-auto", 
                          style={"background-color": "#13376f", "border": "none"})
            ),
        ], id="search-about-modal", size="lg", is_open=False)
    ], style={'max-width': '1200px', 'margin': 'auto'})
    
    return search_tab_layout


def register_search_callbacks(app):
    """
    Register callbacks for the Search tab.
    
    Args:
        app: Dash application instance
    """
    # Callback to toggle the Search About modal
    @app.callback(
        Output("search-about-modal", "is_open"),
        [Input("open-about-search", "n_clicks"), Input("close-search-about", "n_clicks")],
        [State("search-about-modal", "is_open")],
        prevent_initial_call=True
    )
    def toggle_search_about_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open
        
    # Update search syntax help based on mode
    @app.callback(
        Output('search-syntax-help', 'children'),
        Input('search-mode', 'value')
    )
    def update_search_syntax_help(search_mode):
        """
        Update search syntax help text based on selected search mode.
        
        Args:
            search_mode: Selected search mode
            
        Returns:
            html.P: Help text as HTML component
        """
        if search_mode == 'keyword':
            return html.P("Enter words or phrases to match. Whole word matching - searching for 'NATO' won't match 'senator'.", className="text-muted")
        elif search_mode == 'boolean':
            return html.P("Use AND, OR, NOT operators. Example: 'war AND (Ukraine OR Russia) NOT peace'", className="text-muted")
        elif search_mode == 'semantic':
            return html.P("Enter concepts or ideas. Results will include semantically similar content. (Beta feature)", className="text-muted")
        return ""

    # Progress indicator callback - shows loading overlay immediately when search starts
    app.clientside_callback(
        """
        function(n_clicks, search_mode, search_term) {
            if (!n_clicks || !search_term) {
                return [{'display': 'none'}, ''];
            }
            
            let message;
            if (search_mode === 'semantic') {
                message = `(Using AI to understand '${search_term}' and find semantically similar content)`;
            } else if (search_mode === 'boolean') {
                message = `(Processing boolean query: ${search_term})`;
            } else {
                message = `(Looking for exact matches of '${search_term}')`;
            }
            
            return [{'display': 'block'}, message];
        }
        """,
        [
            Output('search-loading-messages', 'style', allow_duplicate=True),
            Output('search-progress-message', 'children', allow_duplicate=True)
        ],
        Input('search-button', 'n_clicks'),
        [
            State('search-mode', 'value'),
            State('search-input', 'value')
        ],
        prevent_initial_call=True
    )

    # Main search functionality
    @app.callback(
        [
            Output('search-sunburst-chart', 'figure'),
            Output('search-timeline-chart', 'figure'),
            Output('search-results-store', 'data'),
            Output('search-stats-container', 'children'),
            Output('search-results-tabs', 'style'),
            Output('search-results-header', 'style'),
            Output('search-pagination-controls', 'style'),
            Output('search-download-buttons', 'style'),
            Output('search-loading-messages', 'style'),
            Output('search-timeline-container', 'style')
        ],
        Input('search-button', 'n_clicks'),
        [
            State('search-mode', 'value'),
            State('search-input', 'value'),
            State('search-language-dropdown', 'value'),
            State('search-database-dropdown', 'value'),
            State('search-source-type-dropdown', 'value'),
            State('search-date-range-picker', 'start_date'),
            State('search-date-range-picker', 'end_date')
        ]
    )
    def update_search_results(n_clicks, search_mode, search_term, lang_val, db_val, source_type, start_date, end_date):
        """
        Update search results based on search criteria and filters.
        
        Args:
            n_clicks: Number of button clicks
            search_mode: Selected search mode
            search_term: Search term
            lang_val: Selected language
            db_val: Selected database
            source_type: Selected source type
            start_date: Start date
            end_date: End date
            
        Returns:
            tuple: Multiple outputs for updating the UI with search results
        """
        # Initialize with empty results
        empty_fig = go.Figure()
        hidden_style = {'display': 'none'}
        loading_hidden = {'display': 'none'}
        
        # If no search has been performed yet, hide everything including the sunburst
        if not n_clicks or not search_term:
            # Return empty content for stats container when no search
            empty_stats = html.Div("Enter a search term and click Search", className="text-center")
            return empty_fig, empty_fig, [], empty_stats, hidden_style, hidden_style, hidden_style, hidden_style, loading_hidden, hidden_style

        # Create date range tuple if both dates are provided
        date_range = None
        if start_date and end_date:
            date_range = (start_date, end_date)

        # 1. Fetch categories for the search term
        df_filtered = fetch_search_category_data(search_mode, search_term, lang_val, db_val, source_type, date_range)
        
        if df_filtered.empty:
            empty_stats = html.Div(f"No results found for '{search_term}'", className="text-center")
            # Hide sunburst when no results
            return empty_fig, empty_fig, [], empty_stats, hidden_style, hidden_style, hidden_style, hidden_style, loading_hidden, hidden_style

        # Create sunburst chart for category distribution - Using the same function that the Explore tab uses
        # This ensures visual consistency across the dashboard
        fig_sunburst = create_sunburst_chart(df_filtered, title=f"Category Distribution for '{search_term}'")

        # Set uniform layout parameters identical to the main sunburst 
        fig_sunburst.update_layout(
            margin=dict(t=50, l=10, r=10, b=10),
            height=700,
            width=700,
            title_x=0.5,
            # Add clickmode to ensure click events are captured properly
            clickmode='event+select'
        )
        
        # Add marker properties to ensure sunburst segments are clickable
        fig_sunburst.update_traces(
            hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percentRoot:.2f}%',
            marker=dict(line=dict(width=0.5, color='#ffffff')),
            selector=dict(type='sunburst')
        )

        # 2. Fetch text chunks that match (with reasonable limit for performance)
        df_chunks = fetch_all_text_chunks_for_search(
            search_mode, search_term, lang_val, db_val, source_type, date_range, 
            limit=500  # Reasonable limit to prevent timeouts while still showing good results
        )
        
        if df_chunks.empty:
            empty_stats = html.Div(f"Search returned category matches but no text chunks for '{search_term}'")
            # Use the correct flex display style for the sunburst container
            sunburst_visible_style = {'margin-bottom': '0px', 'width': '100%', 'justify-content': 'center', 'display': 'flex'}
            return fig_sunburst, empty_fig, [], empty_stats, sunburst_visible_style, hidden_style, hidden_style, hidden_style, loading_hidden, hidden_style

        # Build timeline chart
        df_chunks['date'] = pd.to_datetime(df_chunks['date'], errors='coerce')
        # Filter out invalid dates
        df_chunks_valid_dates = df_chunks.dropna(subset=['date'])
        
        if df_chunks_valid_dates.empty:
            fig_timeline = go.Figure().update_layout(title="No valid dates in search results")
        else:
            df_chunks_valid_dates['month'] = df_chunks_valid_dates['date'].dt.to_period('M').dt.to_timestamp()
            timeline_counts = df_chunks_valid_dates.groupby('month').size().reset_index(name='count')
            
            fig_timeline = create_timeline_chart(
                timeline_counts,
                title="Search Results Timeline"
            )

        # Create stats container with summary information
        search_stats = html.Div([
            dbc.Card([
                dbc.CardHeader(f"Search Results for '{search_term}'"),
                dbc.CardBody([
                    html.P([
                        f"Total results: {format_number(len(df_chunks))} chunks from {format_number(df_chunks['document_id'].nunique())} documents",
                        html.Br(),
                        f"Search mode: {search_mode.capitalize()}",
                        html.Br(),
                        f"Categories: {format_number(df_filtered['category'].nunique())} | Subcategories: {format_number(df_filtered['subcategory'].nunique())}",
                        html.Br(),
                        f"Date range: {df_chunks_valid_dates['date'].min().strftime('%Y-%m-%d') if not df_chunks_valid_dates.empty else 'N/A'} to {df_chunks_valid_dates['date'].max().strftime('%Y-%m-%d') if not df_chunks_valid_dates.empty else 'N/A'}"
                    ])
                ])
            ])
        ], className="mb-4")

        # Show all components
        show_style = {'display': 'block'}
        # Special style for the sunburst chart container - needs to be flex
        sunburst_visible_style = {'margin-bottom': '0px', 'width': '100%', 'justify-content': 'center', 'display': 'flex'}
        pagination_style = {'margin-bottom': '20px', 'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}
        
        logging.info(f"Search results updated. Found {len(df_chunks)} chunks")
        # Make the search-results-header always visible when there are results
        show_header_style = {'display': 'block', 'scroll-margin-top': '100px'}
        return fig_sunburst, fig_timeline, df_chunks.to_dict('records'), search_stats, sunburst_visible_style, show_header_style, pagination_style, show_style, loading_hidden, show_style
        
    # Display search results with pagination
    @app.callback(
        [
            Output('search-chunks-container', 'children'),
            Output('search-page-indicator', 'children'),
            Output('search-current-page-store', 'data'),
            Output('search-segment-loading-indicator', 'children')  # Added to trigger the spinner
        ],
        [
            Input('search-results-store', 'data'),
            Input('search-prev-page-button', 'n_clicks'),
            Input('search-next-page-button', 'n_clicks'),
            Input('search-sunburst-chart', 'clickData')
        ],
        [
            State('search-current-page-store', 'data')
        ]
    )
    def update_search_chunks_display(results_data, prev_clicks, next_clicks, click_data, current_page):
        """
        Display search results with pagination.
        
        Args:
            results_data: Search results data
            prev_clicks: Previous page button clicks
            next_clicks: Next page button clicks
            click_data: Sunburst chart click data 
            current_page: Current page number
            
        Returns:
            tuple: (chunk_rows, page_text, current_page)
        """
        PAGE_SIZE = 10

        if not results_data:
            return [html.Div("No search results to display")], "Page 0 of 0", 0, None

        # Create DataFrame from store data
        df = pd.DataFrame(results_data)
        
        # Handle click context
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        # If the sunburst was clicked, filter the data by the selected category
        if triggered_id == 'search-sunburst-chart' and click_data:
            # Get the selected category from the click data
            selected_category = click_data['points'][0]['label']
            logging.info(f"Search sunburst clicked: {selected_category}")
            
            # Find the level (category, subcategory, or sub_subcategory)
            if selected_category in df['category'].unique():
                level = 'category'
            elif selected_category in df['subcategory'].unique():
                level = 'subcategory'
            elif selected_category in df['sub_subcategory'].unique():
                level = 'sub_subcategory'
            else:
                logging.warning(f"Selected category '{selected_category}' not found in any level")
                return [html.Div("No category found")], "No results", 0, None
            
            # Filter the results based on the selected level and value
            df = df[df[level] == selected_category]
            
            if df.empty:
                logging.warning(f"No results found for {level}: {selected_category}")
                return [html.Div(f"No chunks found for {level}: {selected_category}")], f"0 results", 0, None
            
            # Reset pagination to first page for new selection
            current_page = 0
            
        # Handle regular pagination
        elif triggered_id == 'search-results-store':
            # If new search results, reset page
            current_page = 0
        elif triggered_id == 'search-prev-page-button':
            current_page = max(current_page - 1, 0)
        elif triggered_id == 'search-next-page-button':
            max_page = (len(df) - 1) // PAGE_SIZE
            current_page = min(current_page + 1, max_page)

        # Get current page data
        start_idx = current_page * PAGE_SIZE
        end_idx = min(start_idx + PAGE_SIZE, len(df))
        page_df = df.iloc[start_idx:end_idx]
        
        # Format chunks for display
        chunk_rows = []
        
        # Add anchor div for the Back to Text Chunks button
        anchor = html.Div(id='search-chunks-anchor', style={
            'scroll-margin-top': '120px', 
            'height': '1px'
        })
        chunk_rows.append(anchor)
        
        # Add header when a sunburst segment is clicked - similar to the explore tab's behavior
        if triggered_id == 'search-sunburst-chart' and click_data:
            selected_category = click_data['points'][0]['label']
            level = None
            if selected_category in df['category'].unique():
                level = 'category'
            elif selected_category in df['subcategory'].unique():
                level = 'subcategory'
            elif selected_category in df['sub_subcategory'].unique():
                level = 'sub_subcategory'
                
            if level:
                # Create header with formatting similar to explore tab
                header = html.Div([
                    html.H4(f"{level.capitalize()}: {selected_category}", 
                            className="selection-title", 
                            style={"margin-top": "20px", "margin-bottom": "10px"}),
                    html.Div([
                        html.P(f"Total chunks: {format_number(len(df))}")
                    ], className="selection-stats")
                ], style={"text-align": "center", "margin-bottom": "20px"})
                chunk_rows.append(header)
        
        # Add the chunks
        for i, row in page_df.iterrows():
            try:
                chunk_row = format_chunk_row(row)
                chunk_rows.append(chunk_row)
            except Exception as e:
                logging.error(f"Error formatting search result {i}: {e}")
                chunk_rows.append(html.Div(f"Error displaying result {i}"))
        
        # Create page indicator with detailed information
        total_pages = max(1, (len(df) - 1) // PAGE_SIZE + 1)
        page_text = f"Page {format_number(current_page + 1)} of {format_number(total_pages)} (showing {format_number(len(page_df))} of {format_number(len(df))} chunks)"
        
        return (
            chunk_rows if chunk_rows else [html.Div("No results to display")], 
            page_text,
            current_page,
            None  # Empty content for the loading indicator div
        )

    # Search download handlers
    @app.callback(
        Output("search-download-csv", "data"),
        Input("search-btn-csv", "n_clicks"),
        [
            State('search-results-store', 'data'),
            State('search-input', 'value')
        ],
        prevent_initial_call=True
    )
    def download_search_csv(n_clicks, results_data, search_term):
        """
        Handle CSV download for search results.
        
        Args:
            n_clicks: Button clicks
            results_data: Search results data
            search_term: Search term
            
        Returns:
            dict: Download data
        """
        if not results_data:
            return dash.no_update

        df = pd.DataFrame(results_data)
        search_term_safe = re.sub(r'[^\w]', '_', search_term)
        filename = get_unique_filename(f"search_{search_term_safe}.csv")

        # Create a string buffer, write the CSV, and encode as bytes
        from io import StringIO
        str_buffer = StringIO()
        df.to_csv(str_buffer, index=False)
        
        # Use send_string instead of send_data_frame
        return dcc.send_string(str_buffer.getvalue(), filename)

    @app.callback(
        Output("search-download-json", "data"),
        Input("search-btn-json", "n_clicks"),
        [
            State('search-results-store', 'data'),
            State('search-input', 'value')
        ],
        prevent_initial_call=True
    )
    def download_search_json(n_clicks, results_data, search_term):
        """
        Handle JSON download for search results.
        
        Args:
            n_clicks: Button clicks
            results_data: Search results data
            search_term: Search term
            
        Returns:
            dict: Download data
        """
        if not results_data:
            return dash.no_update

        search_term_safe = re.sub(r'[^\w]', '_', search_term)
        filename = get_unique_filename(f"search_{search_term_safe}.json")

        # Convert to JSON string
        import json
        json_str = json.dumps(results_data)
        
        # Use send_string for JSON
        return dcc.send_string(json_str, filename)