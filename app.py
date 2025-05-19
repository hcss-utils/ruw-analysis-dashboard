#!/usr/bin/env python
# coding: utf-8

"""
Russian-Ukrainian War Data Analysis Dashboard - Refactored Version
-------------------------------------------
This dashboard provides tools to explore, search, and compare data related to the
Russian-Ukrainian War, with enhanced visualizations and performance optimizations.
"""

import logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import os
import sys
from datetime import datetime

# Import Dash components
import dash
import dash_auth
from dash import html, dcc
import dash_bootstrap_components as dbc

# Import application components
from config import VALID_USERNAME_PASSWORD_PAIRS, APP_VERSION, REPORT_PDF_URL, STATIC_HTML_URL
# CHANGE: Explicitly import default dates for use later
from config import DEFAULT_START_DATE, DEFAULT_END_DATE
from database.connection import get_engine, dispose_engine
from database.data_fetchers import fetch_all_databases, fetch_date_range
from tabs.explore import create_explore_tab_layout, register_explore_callbacks
from tabs.search import create_search_tab_layout, register_search_callbacks
from tabs.compare import create_compare_tab_layout, register_compare_callbacks
from tabs.burstiness import create_burstiness_tab_layout, register_burstiness_callbacks
from components.layout import create_header, create_about_modal
from utils.cache import clear_cache

# Define the consistent color for all components - use the exact color from the color picker
THEME_BLUE = "#13376f"  # Dark blue color from the image

# Simple placeholder for the sources tab if there's an issue with the import
def create_simple_sources_tab(db_options, min_date, max_date):
    """Create a simple placeholder for the sources tab"""
    from components.layout import create_filter_card
    
    return html.Div([
        html.H3("Sources", style={"margin-bottom": "20px"}),
        create_filter_card(
            id_prefix="sources",
            db_options=db_options,
            min_date=min_date,
            max_date=max_date
        ),
        html.Div([
            html.H4("Sources Data"),
            html.P("This tab provides analysis of the sources in the dataset."),
            html.P("Use the filters above to explore source distribution and statistics.")
        ], className="mt-4")
    ])

# Simple placeholder for sources callbacks
def register_simple_sources_callbacks(app):
    """Register minimal callbacks for the sources tab"""
    @app.callback(
        dash.Output("sources-result-stats", "children"),
        dash.Input("sources-filter-button", "n_clicks"),
        prevent_initial_call=True
    )
    def update_sources_stats(n_clicks):
        if not n_clicks:
            return dash.no_update
        
        return html.Div("Filters applied")


def create_dash_app() -> dash.Dash:
    """
    Create and configure the Dash application.
    
    Returns:
        dash.Dash: Configured Dash application
    """
    app = dash.Dash(
        __name__, 
        external_stylesheets=[dbc.themes.BOOTSTRAP], 
        assets_folder='static',
        suppress_callback_exceptions=True
    )
    
    app.title = "Russian-Ukrainian War Data Analysis Dashboard"
    app.scripts.config.serve_locally = True
    app.css.config.serve_locally = True
    
    # Add Basic Authentication
    # Comment this out during development
    # auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)
    
    # Fetch initial data
    db_options = []
    min_date, max_date = None, None
    
    try:
        # Get database options for dropdowns
        db_list = fetch_all_databases()
        from config import DATABASE_DISPLAY_MAP
        db_options = [{'label': DATABASE_DISPLAY_MAP.get(db, db), 'value': db} for db in db_list]
        db_options.insert(0, {'label': 'All Databases', 'value': 'ALL'})
        
        # Get date range for date pickers
        db_min_date, max_date = fetch_date_range()
        
        # CHANGE: Force minimum date to be January 1, 2022 regardless of what's in the database
        min_date = DEFAULT_START_DATE
        
        # Only use max_date from database if it's valid
        if max_date is None:
            max_date = DEFAULT_END_DATE
            
        logging.info(f"Initial data loaded: {len(db_options)} databases, date range: {min_date} to {max_date}")
    except Exception as e:
        logging.error(f"Error loading initial data: {e}")
        db_options = [{'label': 'All Databases', 'value': 'ALL'}]
        # CHANGE: Use explicitly imported defaults
        min_date = DEFAULT_START_DATE
        max_date = DEFAULT_END_DATE
    
    # Create the tab layouts
    explore_tab = create_explore_tab_layout(db_options, min_date, max_date)
    search_tab = create_search_tab_layout(db_options, min_date, max_date)
    compare_tab = create_compare_tab_layout(db_options, min_date, max_date)
    burstiness_tab = create_burstiness_tab_layout()
    
    # Try to import the sources tab, but use a placeholder if it fails
    try:
        from tabs.sources import create_sources_tab_layout, register_sources_tab_callbacks
        sources_tab = create_sources_tab_layout(db_options, min_date, max_date)
        sources_callbacks = register_sources_tab_callbacks
    except Exception as e:
        logging.warning(f"Could not import sources tab, using placeholder: {e}")
        sources_tab = create_simple_sources_tab(db_options, min_date, max_date)
        sources_callbacks = register_simple_sources_callbacks
    
    # About modal component
    about_modal = create_about_modal()
    
    # Create the main header
    header = create_header()
    
    # Custom CSS for thinner tabs and responsive layout
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                /* Essential styles for consistent layout */
                body, html {
                    max-width: 100%;
                    overflow-x: hidden;
                }
                
                /* Make tabs much thinner */
                .dash-tab {
                    padding: 4px 12px !important;
                    height: 32px !important;
                    line-height: 24px !important;
                    margin-bottom: -1px;
                }
                
                /* Selected tab styling */
                .dash-tab--selected {
                    font-weight: 500;
                    border-top: 2px solid ''' + THEME_BLUE + ''' !important;
                }
                
                /* Tab content padding */
                .dash-tab-content {
                    padding-top: 10px;
                }
                
                /* Responsive sunburst chart */
                .sunburst-container {
                    display: flex;
                    justify-content: center;
                    width: 100%;
                }
                
                /* Make loading spinner more visible and center it */
                ._dash-loading {
                    position: fixed !important;
                    top: 50% !important;
                    left: 50% !important;
                    transform: translate(-50%, -50%) !important;
                    width: 80px !important;
                    height: 80px !important;
                    border: 8px solid #f3f3f3 !important;
                    border-top: 8px solid ''' + THEME_BLUE + ''' !important;
                    z-index: 9999 !important;
                }
                
                /* Dashboard container for responsive layout */
                .dashboard-container {
                    width: 100%;
                    max-width: 1800px;
                    margin: 0 auto;
                    padding: 0 15px;
                }
                
                /* Ultra-wide screen optimization */
                @media (min-width: 2000px) {
                    .dashboard-container {
                        max-width: 80%;
                    }
                }
                
                /* Compact date picker for fitting in one row */
                .DateInput {
                    width: 85px !important;
                }
                
                .DateInput_input {
                    padding: 2px 8px !important;
                    font-size: 0.8rem !important;
                }
                
                .DateRangePickerInput_arrow {
                    padding: 0 5px !important;
                }
                
                /* Filter card styling to make it more compact */
                .filter-card .card-body {
                    padding: 1rem 1rem 0.5rem 1rem;
                }
                
                /* Make dropdowns more compact */
                .Select-control, .Select-menu-outer {
                    font-size: 0.9rem !important;
                }
                
                /* Button styling to match header */
                .about-button {
                    background-color: ''' + THEME_BLUE + ''' !important;
                    border-color: ''' + THEME_BLUE + ''' !important;
                    color: white !important;
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                }
                
                /* Header styling */
                .app-header {
                    background-color: ''' + THEME_BLUE + ''';
                    color: white;
                    padding: 10px 20px;
                    margin-bottom: 20px;
                    border-radius: 5px;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                }
                
                .app-header h2 {
                    margin: 0;
                    font-weight: 500;
                }
                
                /* Tooltip styling for the sunburst chart */
                .js-plotly-plot .plotly-notifier {
                    font-size: 14px !important;
                }
                
                /* Prettier select message */
                .select-message {
                    font-size: 1.2rem;
                    color: #666;
                    text-align: center;
                    padding: 2rem;
                    font-style: italic;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    border: 1px solid #eee;
                    margin-top: 2rem;
                }
                
                /* About box styling */
                .about-box {
                    background-color: #f9f9f9;
                    border: 1px solid #eee;
                    border-radius: 5px;
                    padding: 1rem;
                    margin: 1rem 0;
                }
                
                .about-box h5 {
                    color: ''' + THEME_BLUE + ''';
                    border-bottom: 1px solid #eee;
                    padding-bottom: 0.5rem;
                    margin-bottom: 1rem;
                }
                
                /* Timeline caption styling - make it stick properly */
                .timeline-caption {
                    position: sticky;
                    top: 0;
                    background: white;
                    z-index: 100;
                    padding: 10px;
                    border-bottom: 2px solid #eee;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                
                /* Selection title styling - match the style from the screenshot */
                .selection-title {
                    color: #2196F3;
                    font-size: 1.5rem;
                    font-weight: 500;
                    margin-top: 1rem;
                    margin-bottom: 0.5rem;
                    text-align: center;
                }
                
                /* Reduce vertical gap between sunburst and title */
                .sunburst-chart-container {
                    margin-bottom: 10px !important;
                }
                
                /* Card header styling */
                .card-header {
                    background-color: ''' + THEME_BLUE + ''' !important;
                    color: white !important;
                }
                
                /* Button styling */
                .btn-primary, .btn-secondary, .btn-success, .btn-info {
                    background-color: ''' + THEME_BLUE + ''' !important;
                    border-color: ''' + THEME_BLUE + ''' !important;
                }
                
                /* Consistent spinner styling */
                .dash-spinner .dash-spinner-inner {
                    border-top-color: ''' + THEME_BLUE + ''' !important;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    # Main layout with tabs
    app.layout = html.Div([
        header,
        dcc.Tabs([
            dcc.Tab(label="Explore", children=explore_tab, className="custom-tab"),
            dcc.Tab(label="Search", children=search_tab, className="custom-tab"),
            dcc.Tab(label="Compare", children=compare_tab, className="custom-tab"),
            dcc.Tab(label="Burstiness", children=burstiness_tab, className="custom-tab"),
            dcc.Tab(label="Sources", children=sources_tab, className="custom-tab"),
        ], className="slimmer-tabs"),
        # About modal
        about_modal,
        # Download components
        dcc.Download(id="download-dataframe-csv"),
        dcc.Download(id="download-dataframe-json"),
        dcc.Download(id="search-download-csv"),
        dcc.Download(id="search-download-json"),
        # Local storage - Cache status
        dcc.Store(id="cache-status", data={"enabled": True})
    ], className="dashboard-container")
    
    # Register callbacks for each tab
    register_explore_callbacks(app)
    register_search_callbacks(app)
    register_compare_callbacks(app)
    register_burstiness_callbacks(app)
    sources_callbacks(app)
    
    # Register about modal callback - MODIFIED: Now only main header About button triggers this
    @app.callback(
        dash.Output("about-modal", "is_open", allow_duplicate=True),
        [
            dash.Input("open-about-main", "n_clicks"),
            dash.Input("close-about", "n_clicks"),
        ],
        [dash.State("about-modal", "is_open")],
        prevent_initial_call=True
    )
    def toggle_about_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open
    
    # Add cache clearing callback
    @app.callback(
        dash.Output("cache-status", "data"),
        dash.Input("clear-cache-button", "n_clicks"),
        dash.State("cache-status", "data"),
        prevent_initial_call=True
    )
    def handle_clear_cache(n_clicks, cache_status):
        if n_clicks:
            clear_cache()
            cache_status["last_cleared"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return cache_status
        return dash.no_update
    
    return app


# Create app at module level for Gunicorn
app = create_dash_app()
# Expose server for Gunicorn
server = app.server

def main():
    """
    Main entry point for the application when run directly.
    """
    try:
        port = int(os.environ.get("PORT", 8051))
        # Use app.run() instead of app.run_server()
        app.run(debug=True, host='0.0.0.0', port=port)
        
    except Exception as e:
        logging.error(f"Error starting application: {e}")
        raise
    finally:
        # Clean up resources when the app exits
        dispose_engine()
        logging.info("Application resources cleaned up")


if __name__ == "__main__":
    main()