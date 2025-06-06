#!/usr/bin/env python
# coding: utf-8

"""
Fast Sources tab with basic stats only - no heavy queries.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.connection import get_engine
from components.layout import create_filter_card
from config import THEME_COLORS
from sqlalchemy import text
from utils.cache import cached


@cached(timeout=3600)
def fetch_basic_stats(lang_val=None, db_val=None, source_type=None, date_range=None):
    """Fetch only basic statistics - fast queries only."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Build filter conditions
            filters = []
            params = {}
            
            if lang_val and lang_val != 'ALL':
                filters.append("ud.language = :lang")
                params['lang'] = lang_val
            
            if db_val and db_val != 'ALL':
                filters.append("ud.database = :db")
                params['db'] = db_val
                
            if date_range and len(date_range) == 2:
                filters.append("ud.date >= :start_date AND ud.date <= :end_date")
                params['start_date'] = date_range[0]
                params['end_date'] = date_range[1]
            
            filter_sql = " AND " + " AND ".join(filters) if filters else ""
            
            # Single efficient query to get all counts
            stats_query = f"""
            WITH doc_stats AS (
                SELECT 
                    COUNT(DISTINCT ud.id) as total_docs,
                    COUNT(DISTINCT CASE WHEN t.id IS NOT NULL THEN ud.id END) as relevant_docs,
                    MIN(ud.date) as earliest_date,
                    MAX(ud.date) as latest_date
                FROM uploaded_document ud
                LEFT JOIN document_section ds ON ud.id = ds.uploaded_document_id
                LEFT JOIN document_section_chunk dsc ON ds.id = dsc.document_section_id
                LEFT JOIN taxonomy t ON dsc.id = t.chunk_id
                WHERE 1=1 {filter_sql}
            ),
            chunk_stats AS (
                SELECT
                    COUNT(DISTINCT dsc.id) as total_chunks,
                    COUNT(DISTINCT CASE WHEN t.id IS NOT NULL THEN dsc.id END) as relevant_chunks
                FROM document_section_chunk dsc
                JOIN document_section ds ON dsc.document_section_id = ds.id
                JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
                LEFT JOIN taxonomy t ON dsc.id = t.chunk_id
                WHERE 1=1 {filter_sql}
            ),
            tax_stats AS (
                SELECT 
                    COUNT(DISTINCT t.category) as categories,
                    COUNT(DISTINCT t.subcategory) as subcategories,
                    COUNT(DISTINCT t.sub_subcategory) as sub_subcategories,
                    COUNT(*) as total_items
                FROM taxonomy t
                JOIN document_section_chunk dsc ON t.chunk_id = dsc.id
                JOIN document_section ds ON dsc.document_section_id = ds.id
                JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
                WHERE 1=1 {filter_sql}
            )
            SELECT 
                d.total_docs,
                d.relevant_docs,
                d.earliest_date,
                d.latest_date,
                c.total_chunks,
                c.relevant_chunks,
                t.categories,
                t.subcategories,
                t.sub_subcategories,
                t.total_items
            FROM doc_stats d, chunk_stats c, tax_stats t
            """
            
            result = pd.read_sql(text(stats_query), conn, params=params)
            
            if not result.empty:
                row = result.iloc[0]
                return {
                    'total_docs': int(row['total_docs']),
                    'relevant_docs': int(row['relevant_docs']),
                    'earliest_date': row['earliest_date'].strftime('%Y-%m-%d') if pd.notna(row['earliest_date']) else 'N/A',
                    'latest_date': row['latest_date'].strftime('%Y-%m-%d') if pd.notna(row['latest_date']) else 'N/A',
                    'total_chunks': int(row['total_chunks']),
                    'relevant_chunks': int(row['relevant_chunks']),
                    'categories': int(row['categories']),
                    'subcategories': int(row['subcategories']),
                    'sub_subcategories': int(row['sub_subcategories']),
                    'total_items': int(row['total_items']),
                    'relevance_rate': round(row['relevant_docs'] / row['total_docs'] * 100, 1) if row['total_docs'] > 0 else 0,
                    'chunk_coverage': round(row['relevant_chunks'] / row['total_chunks'] * 100, 1) if row['total_chunks'] > 0 else 0
                }
            
    except Exception as e:
        logging.error(f"Error fetching basic stats: {e}")
    
    return None


def create_sources_tab_layout(db_options: List, min_date: datetime = None, max_date: datetime = None) -> html.Div:
    """Create the fast Sources tab layout."""
    blue_color = THEME_COLORS.get('western', '#13376f')
    
    sources_tab = html.Div([
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
        
        # Stats container
        html.Div(id='sources-stats-container', className="mt-4"),
        
        # Loading spinner
        dbc.Spinner(
            html.Div(id="sources-content"),
            color="primary",
            spinner_style={"width": "3rem", "height": "3rem"}
        ),
        
        # About modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Using the Sources Tab"), 
                           style={"background-color": "#13376f", "color": "white"}),
            dbc.ModalBody([
                html.P("The Sources tab provides an overview of the corpus used for this dashboard."),
                html.P("Due to the large size of the corpus, this tab shows basic statistics only. "
                       "For detailed analysis, please use the other tabs."),
                html.H5("Statistics Shown:", style={"margin-top": "20px", "color": "#13376f"}),
                html.Ul([
                    html.Li("Total documents and chunks in the corpus"),
                    html.Li("Documents and chunks with taxonomic classifications"),
                    html.Li("Taxonomy hierarchy statistics"),
                    html.Li("Date range of the corpus"),
                    html.Li("Relevance and coverage rates")
                ])
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-sources-about", className="ms-auto", 
                          style={"background-color": "#13376f", "border": "none"})
            ),
        ], id="sources-about-modal", size="lg", is_open=False)
    ])
    
    return sources_tab


def register_sources_tab_callbacks(app):
    """Register callbacks for the fast Sources tab."""
    
    @app.callback(
        Output("sources-about-modal", "is_open"),
        [Input("open-about-sources", "n_clicks"), Input("close-sources-about", "n_clicks")],
        [State("sources-about-modal", "is_open")],
        prevent_initial_call=True
    )
    def toggle_about_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open
    
    @app.callback(
        [Output("sources-stats-container", "children"),
         Output("sources-content", "children")],
        [Input("sources-apply-button", "n_clicks"),
         Input("tabs", "value")],
        [State("sources-language-dropdown", "value"),
         State("sources-database-dropdown", "value"),
         State("sources-source-type-dropdown", "value"),
         State("sources-date-range-picker", "start_date"),
         State("sources-date-range-picker", "end_date")],
        prevent_initial_call=False
    )
    def update_sources(n_clicks, active_tab, lang_val, db_val, source_type, start_date, end_date):
        """Update sources tab with basic statistics only."""
        
        if active_tab != "tab-sources":
            return dash.no_update, dash.no_update
        
        # Process filters
        date_range = None
        if start_date and end_date:
            date_range = (
                start_date.split('T')[0] if 'T' in str(start_date) else start_date,
                end_date.split('T')[0] if 'T' in str(end_date) else end_date
            )
        
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
        
        # Fetch basic stats
        stats = fetch_basic_stats(lang_val, db_val, source_type, date_range)
        
        if not stats:
            error_content = dbc.Alert(
                "Unable to load corpus statistics. Please try again.",
                color="danger"
            )
            return error_content, error_content
        
        # Create stats summary
        stats_summary = html.Div([
            html.P(f"Filters: {' | '.join(filter_desc) if filter_desc else 'None'}", 
                   className="text-muted mb-3"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{stats['total_docs']:,}", className="text-primary"),
                            html.P("Total Documents", className="mb-1"),
                            html.Small(f"{stats['relevant_docs']:,} with taxonomy ({stats['relevance_rate']:.1f}%)", 
                                      className="text-muted")
                        ])
                    ])
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{stats['total_chunks']:,}", className="text-primary"),
                            html.P("Total Chunks", className="mb-1"),
                            html.Small(f"{stats['relevant_chunks']:,} with taxonomy ({stats['chunk_coverage']:.1f}%)", 
                                      className="text-muted")
                        ])
                    ])
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{stats['categories'] + stats['subcategories'] + stats['sub_subcategories']:,}", 
                                    className="text-primary"),
                            html.P("Taxonomy Levels", className="mb-1"),
                            html.Small(f"{stats['total_items']:,} total items", className="text-muted")
                        ])
                    ])
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Date Range", className="text-primary"),
                            html.P(f"{stats['earliest_date']}", className="mb-0"),
                            html.P(f"to {stats['latest_date']}", className="mb-0")
                        ])
                    ])
                ], md=3)
            ])
        ])
        
        # Create content
        content = html.Div([
            dbc.Alert([
                html.H5("ℹ️ Basic Statistics View", className="alert-heading"),
                html.P("This tab shows basic corpus statistics for quick overview."),
                html.P("For detailed analysis of keywords, named entities, and other features, "
                       "please use the Search, Compare, and Explore tabs.", className="mb-0")
            ], color="info", className="mt-4"),
            
            html.Div([
                html.H5("Taxonomy Breakdown", className="mt-4 mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6("Categories", className="text-muted"),
                            html.H3(f"{stats['categories']:,}", className="text-primary")
                        ], className="text-center")
                    ], md=4),
                    dbc.Col([
                        html.Div([
                            html.H6("Subcategories", className="text-muted"),
                            html.H3(f"{stats['subcategories']:,}", className="text-primary")
                        ], className="text-center")
                    ], md=4),
                    dbc.Col([
                        html.Div([
                            html.H6("Sub-subcategories", className="text-muted"),
                            html.H3(f"{stats['sub_subcategories']:,}", className="text-primary")
                        ], className="text-center")
                    ], md=4)
                ])
            ])
        ])
        
        return stats_summary, content