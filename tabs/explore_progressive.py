#!/usr/bin/env python
# coding: utf-8

"""
Progressive loading implementation for the Explore tab.
This module splits the loading into priority and background tasks.
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go


def register_progressive_explore_callbacks(app):
    """
    Register callbacks for progressive loading in the Explore tab.
    
    Priority loading:
    1. Sunburst chart
    2. First batch of chunks
    
    Background loading:
    3. Timeline chart
    4. Additional chunk pages
    5. Statistics
    """
    
    # Priority callback 1: Load sunburst chart ONLY
    @app.callback(
        [
            Output('sunburst-chart', 'figure'),
            Output('explore-result-stats', 'children'),
            Output('sunburst-loading-complete', 'data')  # Signal completion
        ],
        Input('explore-filter-button', 'n_clicks'),
        [
            State('explore-language-dropdown', 'value'),
            State('explore-database-dropdown', 'value'),
            State('explore-source-type-dropdown', 'value'),
            State('explore-date-range-picker', 'start_date'),
            State('explore-date-range-picker', 'end_date')
        ],
        prevent_initial_call=True
    )
    def load_sunburst_priority(n_clicks, lang_val, db_val, source_type, start_date, end_date):
        """Load only the sunburst chart as priority."""
        if not n_clicks:
            return dash.no_update, dash.no_update, False
            
        logging.info("[PRIORITY] Loading sunburst chart")
        
        from database.data_fetchers import fetch_category_data
        from visualizations.sunburst import create_sunburst_chart
        
        # Prepare date range
        date_range = None
        if start_date is not None and end_date is not None:
            date_range = (start_date, end_date)
        
        # Fetch ONLY category data for sunburst
        if lang_val == 'ALL' and db_val == 'ALL' and source_type == 'ALL' and date_range is None:
            df = fetch_category_data()
        else:
            df = fetch_category_data(lang_val, db_val, source_type, date_range)
        
        # Create sunburst
        if df is not None and not df.empty:
            fig = create_sunburst_chart(df, title="Taxonomic Element Distribution")
            total_items = df['count'].sum()
            unique_categories = df['category'].nunique()
            result_text = f"Found {total_items:,} items across {unique_categories} categories"
        else:
            fig = create_sunburst_chart(pd.DataFrame(), title="No Data Found")
            result_text = "No data found for the selected filters"
        
        logging.info("[PRIORITY] Sunburst chart loaded")
        return fig, result_text, True
    
    # Priority callback 2: Load first batch of chunks when sunburst is clicked
    @app.callback(
        [
            Output('chunks-selection-title', 'children'),
            Output('chunks-stats', 'children'),
            Output('explore-chunks-container', 'children'),
            Output('chunks-loading-complete', 'data')
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
    def load_chunks_priority(clickData, lang_val, db_val, source_type, start_date, end_date):
        """Load only the first batch of chunks as priority."""
        if not clickData:
            return "", "", [], False
            
        logging.info("[PRIORITY] Loading first batch of chunks")
        
        from database.data_fetchers import fetch_text_chunks, fetch_text_chunks_count
        from utils.helpers import format_chunk_row
        
        # Extract selection
        selected = clickData['points'][0]['label']
        
        # Determine level based on parent
        level = 'category'  # Default
        parent = clickData['points'][0].get('parent', '')
        if parent and parent != '':
            if parent.count('/') == 0:
                level = 'subcategory'
            else:
                level = 'sub_subcategory'
        
        # Prepare filters
        date_range = None
        if start_date and end_date:
            date_range = (start_date, end_date)
        
        # Get count for stats
        total_count = fetch_text_chunks_count(level, selected, lang_val, db_val, source_type, date_range)
        
        # Fetch ONLY first page of chunks
        chunks_df = fetch_text_chunks(
            level, selected, lang_val, db_val, source_type, date_range, 
            page=1, page_size=10
        )
        
        # Format chunks
        chunk_rows = []
        if chunks_df is not None and not chunks_df.empty:
            for _, row in chunks_df.iterrows():
                chunk_rows.append(format_chunk_row(row))
        
        # Create title and stats
        title = f"Text Chunks for: {selected}"
        stats = f"Showing {len(chunk_rows)} of {total_count:,} chunks (Page 1)"
        
        logging.info(f"[PRIORITY] Loaded {len(chunk_rows)} chunks")
        return title, stats, chunk_rows, True
    
    # Background callback: Load timeline after priority items
    @app.callback(
        [
            Output('timeline-chart', 'figure'),
            Output('timeline-container', 'style'),
            Output('timeline-caption', 'children'),
            Output('timeline-caption', 'style')
        ],
        [
            Input('chunks-loading-complete', 'data'),  # Wait for chunks to load
            Input('sunburst-chart', 'clickData')
        ],
        [
            State('explore-language-dropdown', 'value'),
            State('explore-database-dropdown', 'value'),
            State('explore-source-type-dropdown', 'value'),
            State('explore-date-range-picker', 'start_date'),
            State('explore-date-range-picker', 'end_date')
        ]
    )
    def load_timeline_background(chunks_loaded, clickData, lang_val, db_val, source_type, start_date, end_date):
        """Load timeline chart in the background."""
        if not chunks_loaded or not clickData:
            return {}, {'display': 'none'}, "", {'display': 'none'}
            
        logging.info("[BACKGROUND] Loading timeline chart")
        
        from database.data_fetchers import fetch_timeline_data
        from visualizations.timeline import create_timeline_chart
        
        # Extract selection info
        selected = clickData['points'][0]['label']
        level = 'category'
        parent = clickData['points'][0].get('parent', '')
        if parent and parent != '':
            if parent.count('/') == 0:
                level = 'subcategory'
            else:
                level = 'sub_subcategory'
        
        # Prepare filters
        date_range = None
        if start_date and end_date:
            date_range = (start_date, end_date)
        
        # Fetch timeline data
        timeline_df = fetch_timeline_data(level, selected, lang_val, db_val, source_type, date_range)
        
        if timeline_df is not None and not timeline_df.empty:
            fig = create_timeline_chart(timeline_df, title=f"Timeline for {selected}")
            caption_text = f"Showing timeline for: {selected}"
            caption_style = {
                'position': 'sticky',
                'top': 0,
                'background': 'white',
                'zIndex': 200,
                'borderBottom': '2px solid #13376f',
                'padding': '10px',
                'display': 'block',
                'text-align': 'center'
            }
        else:
            fig = {}
            caption_text = ""
            caption_style = {'display': 'none'}
        
        logging.info("[BACKGROUND] Timeline chart loaded")
        return fig, {'display': 'block'}, caption_text, caption_style
    
    # Background callback: Handle pagination
    @app.callback(
        Output('explore-chunks-container', 'children', allow_duplicate=True),
        [
            Input('explore-prev-page', 'n_clicks'),
            Input('explore-next-page', 'n_clicks')
        ],
        [
            State('current-page-store', 'data'),
            State('filtered-chunks-store', 'data'),
            State('current-selection-store', 'data'),
            State('explore-language-dropdown', 'value'),
            State('explore-database-dropdown', 'value'),
            State('explore-source-type-dropdown', 'value'),
            State('explore-date-range-picker', 'start_date'),
            State('explore-date-range-picker', 'end_date')
        ],
        prevent_initial_call=True
    )
    def handle_pagination_background(prev_clicks, next_clicks, current_page, filtered_data, 
                                   selection_data, lang_val, db_val, source_type, 
                                   start_date, end_date):
        """Handle pagination in the background."""
        logging.info("[BACKGROUND] Loading additional chunk pages")
        
        # Implementation would go here
        # This is just a placeholder to show the structure
        return dash.no_update