#!/usr/bin/env python
# coding: utf-8

"""
Card components for the dashboard.
"""

import dash_bootstrap_components as dbc
from dash import html


def create_info_card(title: str, content: str, footer: str = None) -> dbc.Card:
    """
    Create an informational card with title, content, and optional footer.
    
    Args:
        title: Card title
        content: Card content
        footer: Optional footer text
        
    Returns:
        dbc.Card: Card component
    """
    card_content = [
        dbc.CardHeader(title),
        dbc.CardBody([
            html.P(content, className="card-text")
        ])
    ]
    
    if footer:
        card_content.append(dbc.CardFooter(footer))
    
    return dbc.Card(card_content, className="mb-4")


def create_stats_card(title: str, stats_data: dict, color: str = "primary") -> dbc.Card:
    """
    Create a card with statistics.
    
    Args:
        title: Card title
        stats_data: Dictionary of statistics (label -> value)
        color: Card color
        
    Returns:
        dbc.Card: Card component
    """
    # Create stats items
    stats_items = []
    for label, value in stats_data.items():
        stats_items.append(html.P([
            html.Span(f"{label}: ", className="font-weight-bold"),
            html.Span(f"{value}")
        ], className="mb-1"))
    
    return dbc.Card([
        dbc.CardHeader(title),
        dbc.CardBody(stats_items)
    ], color=color, outline=True, className="mb-4")


def create_search_results_card(search_term: str, result_count: int, db_count: int, metadata: dict = None) -> dbc.Card:
    """
    Create a card showing search results metadata.
    
    Args:
        search_term: Search term
        result_count: Number of results
        db_count: Number of databases
        metadata: Additional metadata as dictionary
        
    Returns:
        dbc.Card: Card component
    """
    metadata_items = []
    if metadata:
        for key, value in metadata.items():
            metadata_items.append(
                html.Span([f"{key}: {value}", html.Br()])
            )
    
    return dbc.Card([
        dbc.CardHeader(f"Search Results for '{search_term}'"),
        dbc.CardBody([
            html.P([
                f"Total results: {result_count:,} items from {db_count} documents",
                html.Br(),
                *metadata_items
            ])
        ])
    ], className="mb-4")


def create_comparison_card(title: str, content_list: list, color: str = "info") -> dbc.Card:
    """
    Create a card for comparison analysis.
    
    Args:
        title: Card title
        content_list: List of content items
        color: Card color
        
    Returns:
        dbc.Card: Card component
    """
    return dbc.Card([
        dbc.CardHeader(title),
        dbc.CardBody(content_list)
    ], color=color, outline=True, className="mb-4")


def create_freshness_explanation_card() -> dbc.Card:
    """
    Create a card explaining freshness metrics.
    
    Returns:
        dbc.Card: Card component
    """
    return dbc.Card([
        dbc.CardHeader("Understanding Freshness Metrics"),
        dbc.CardBody([
            html.P([
                "The freshness analysis helps identify which taxonomic elements are most current in the discourse. ",
                "Fresher topics (with higher freshness scores) appear in brighter green, while older or staler topics ",
                "appear in yellow."
            ]),
            html.P([
                "Freshness is calculated based on how recently and how frequently a taxonomic element has been ",
                "mentioned in the selected time period."
            ]),
            html.P([
                "Use the Category Drill-Down tab to explore freshness at deeper levels of the taxonomy hierarchy."
            ])
        ])
    ], className="mt-4")


def create_visualization_guide_card() -> dbc.Card:
    """
    Create a card explaining visualization options.
    
    Returns:
        dbc.Card: Card component
    """
    return dbc.Card([
        dbc.CardHeader("Visualization Guide"),
        dbc.CardBody([
            html.P([
                "Use different visualizations to explore different aspects of the data:"
            ]),
            html.Ul([
                html.Li([html.Strong("Parallel Stacked Bars: "), "Compare percentage distributions within each slice with connecting lines"]),
                html.Li([html.Strong("Radar Chart: "), "See relative proportions of categories across both slices in a single chart"]),
                html.Li([html.Strong("Heatmap: "), "View the percentage point differences between slices with color intensity"]),
                html.Li([html.Strong("Difference in Means: "), "See which slice has higher proportion of each category"]),
                html.Li([html.Strong("Sunburst: "), "Explore hierarchical category structure in both slices"]),
                html.Li([html.Strong("Sankey: "), "Visualize flow of data from slices to categories"])
            ])
        ])
    ], className="mb-4")