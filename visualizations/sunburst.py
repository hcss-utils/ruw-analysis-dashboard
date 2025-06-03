#!/usr/bin/env python
# coding: utf-8

"""
Sunburst visualization helpers for the dashboard.
"""

import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional, Any, Union

import sys
import os

# Add the project root to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import THEME_COLORS
from utils.helpers import hex_to_rgba


def process_data_for_sunburst(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """
    Process DataFrame for sunburst visualization.
    
    Args:
        df: DataFrame with category data
        
    Returns:
        Tuple: (outer_counts, middle_counts, inner_counts, total_count)
    """
    if df.empty:
        # Return empty data structures
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0
    
    outer_counts = df.groupby(['category', 'subcategory', 'sub_subcategory'])['count'].sum().reset_index()
    total_count = outer_counts['count'].sum()

    middle_counts = outer_counts.groupby(['category', 'subcategory'])['count'].sum().reset_index()
    middle_counts['percentage'] = (middle_counts['count'] / total_count * 100).round(2)

    inner_counts = outer_counts.groupby('category')['count'].sum().reset_index()
    inner_counts['percentage'] = (inner_counts['count'] / total_count * 100).round(2)

    return outer_counts, middle_counts, inner_counts, total_count


def create_color_mapping(inner_counts: pd.DataFrame, middle_counts: pd.DataFrame, outer_counts: pd.DataFrame) -> Dict[str, str]:
    """
    Create color mapping with alpha variations.
    
    Args:
        inner_counts: DataFrame with inner counts
        middle_counts: DataFrame with middle counts
        outer_counts: DataFrame with outer counts
        
    Returns:
        Dict[str, str]: Color mapping for all elements
    """
    if inner_counts.empty:
        return {}  # Return empty dict if no data
    
    # Define the consistent blue color
    THEME_BLUE = "#13376f"  # Dark blue from the color picker
    
    # Use colors from THEME_COLORS when category name matches a key
    color_map = {}
    
    # First assign consistent colors for categories
    for idx, row in inner_counts.iterrows():
        category = row['category']
        # Check if the category name directly matches a language code or other key in THEME_COLORS
        if category in THEME_COLORS:
            base_color = THEME_COLORS[category]
        else:
            # Use default Set3 colors if no match
            color_idx = idx % len(px.colors.qualitative.Set3)
            base_color = px.colors.qualitative.Set3[color_idx]
            # Convert from rgb to hex if needed
            if base_color.startswith('rgb'):
                base_color = base_color.lstrip('rgb(').rstrip(')')
                rgb_components = [int(c.strip()) for c in base_color.split(',')]
                base_color = f'#{rgb_components[0]:02x}{rgb_components[1]:02x}{rgb_components[2]:02x}'
        
        color_map[category] = base_color
        
        # Process subcategories with reduced alpha
        sub_mask = middle_counts['category'] == category
        n_subs = sub_mask.sum()
        sub_alphas = np.linspace(0.7, 0.9, max(1, n_subs))
        np.random.seed(42)  # Fixed seed for consistent colors
        np.random.shuffle(sub_alphas)
        
        for sub_idx, (_, sub_row) in enumerate(middle_counts[sub_mask].iterrows()):
            alpha = min(max(sub_alphas[sub_idx % len(sub_alphas)], 0.0), 1.0)
            color_map[sub_row['subcategory']] = hex_to_rgba(base_color, alpha)
            
            # Process sub-subcategories with further reduced alpha
            subsub_mask = outer_counts['subcategory'] == sub_row['subcategory']
            n_subsubs = subsub_mask.sum()
            subsub_alphas = np.linspace(0.4, 0.6, max(1, n_subsubs))
            np.random.shuffle(subsub_alphas)
            
            for subsub_idx, (_, subsub_row) in enumerate(outer_counts[subsub_mask].iterrows()):
                alpha = min(max(subsub_alphas[subsub_idx % len(subsub_alphas)], 0.0), 1.0)
                color_map[subsub_row['sub_subcategory']] = hex_to_rgba(base_color, alpha)
    
    return color_map


def create_sunburst_chart(df: pd.DataFrame, title: str = "Taxonomic Element Distribution") -> go.Figure:
    """
    Create a sunburst chart from a DataFrame.
    
    Args:
        df: DataFrame with category data
        title: Chart title
        
    Returns:
        go.Figure: Plotly Figure object
    """
    # Define the blue color for title to match selection title styling
    BLUE_COLOR = "#2196F3"  # This matches the selection title color in the screenshot
    
    if df.empty:
        # Create an empty figure with the properly styled title
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title={
                'text': "No data available",
                'font': {
                    'size': 24,
                    'color': BLUE_COLOR
                },
                'x': 0.5,
                'y': 0.95
            },
            height=700,
            width=700
        )
        return empty_fig
        
    outer_counts, middle_counts, inner_counts, _ = process_data_for_sunburst(df)
    color_map = create_color_mapping(inner_counts, middle_counts, outer_counts)
    
    # We need to manually apply colors after creating the figure
    fig = px.sunburst(
        outer_counts,
        path=['category', 'subcategory', 'sub_subcategory'],
        values='count'
    )
    
    # Apply the color mapping manually to the traces
    # Get the labels from the figure data
    if fig.data and len(fig.data) > 0:
        labels = fig.data[0].labels
        colors = []
        
        # Map each label to its color from our color_map
        for label in labels:
            if label in color_map:
                colors.append(color_map[label])
            else:
                # Default color if not found
                colors.append('#cccccc')
        
        # Update the trace with our custom colors
        fig.update_traces(marker=dict(colors=colors))
    
    # Update layout for better readability with consistent styling for title
    fig.update_layout(
        margin=dict(t=50, l=10, r=10, b=10),
        height=700,
        width=700,
        title={
            'text': title,
            'font': {
                'size': 24,
                'color': BLUE_COLOR
            },
            'x': 0.5,
            'y': 0.98  # Position the title higher to reduce vertical gap
        }
    )
    
    # Add percentage hover info WITH THOUSANDS SEPARATORS
    # Also update text display to use HTML line breaks
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percentRoot:.2f}%',
        textinfo='label+percent entry',
        insidetextorientation='radial'
    )
    
    return fig