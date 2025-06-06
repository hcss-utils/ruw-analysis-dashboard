#!/usr/bin/env python
# coding: utf-8

"""
Minimal test to check Sources tab
"""

import dash
from dash import html, dcc
import logging

logging.basicConfig(level=logging.DEBUG)

# Create a minimal app
app = dash.Dash(__name__)

# Test Sources tab
try:
    from tabs.sources import create_sources_tab_layout, register_sources_tab_callbacks
    
    # Create layout
    sources_tab = create_sources_tab_layout(
        db_options=[{'label': 'All', 'value': 'ALL'}],
        min_date=None,
        max_date=None
    )
    
    app.layout = html.Div([
        html.H1("Test Sources Tab"),
        sources_tab
    ])
    
    # Register callbacks
    register_sources_tab_callbacks(app)
    
    print("✓ Sources tab setup complete")
    
    # Run the app
    app.run_server(debug=True, port=8051)
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()