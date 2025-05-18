#!/usr/bin/env python
# coding: utf-8

"""
Freshness visualization helpers for the dashboard.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional


def create_freshness_bar_chart(df: pd.DataFrame, title: str = "Taxonomic Element Freshness") -> go.Figure:
    """
    Create a bar chart showing taxonomic element freshness.
    
    Args:
        df: DataFrame with freshness data
        title: Chart title
        
    Returns:
        go.Figure: Plotly Figure object
    """
    if df.empty:
        return go.Figure().update_layout(title="No data available for freshness")
    
    # Sort by freshness score
    sorted_df = df.sort_values('freshness_score', ascending=False)
    
    # Create the bar chart
    fig = px.bar(
        sorted_df,
        x='category',
        y='freshness_score',
        color='freshness_score',
        color_continuous_scale=[(0, '#FFEB3B'), (1, '#8BC34A')],  # Yellow to Green
        title=title,
        labels={'category': 'Taxonomic Element', 'freshness_score': 'Freshness Score'},
        hover_data=['count', 'latest_date', 'recency_score', 'frequency_score']
    )
    
    # Rotate x-axis labels for better readability
    fig.update_layout(
        xaxis_tickangle=-45,
        xaxis_title='',
        yaxis_title='Freshness Score (0-100)',
        height=600,
        margin=dict(b=150)  # Extra bottom margin for rotated labels
    )
    
    # Add hover template WITH THOUSANDS SEPARATORS
    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>Freshness Score: %{y:.1f}<br>Count: %{customdata[0]:,}<br>Latest Date: %{customdata[1]}<br>Recency: %{customdata[2]:.1f}<br>Frequency: %{customdata[3]:.1f}<extra></extra>'
    )
    
    return fig


def create_freshness_timeline(df: pd.DataFrame, title: str = "Most Recent Taxonomic Elements") -> go.Figure:
    """
    Create a timeline chart showing when taxonomic elements were last mentioned.
    
    Args:
        df: DataFrame with freshness data
        title: Chart title
        
    Returns:
        go.Figure: Plotly Figure object
    """
    if df.empty:
        return go.Figure().update_layout(title="No data available for freshness timeline")
    
    # Ensure date column is properly formatted
    df['latest_date'] = pd.to_datetime(df['latest_date'])
    
    # Sort by date
    sorted_df = df.sort_values('latest_date', ascending=False)
    
    # Create scatter plot with color intensity representing freshness
    fig = px.scatter(
        sorted_df,
        x='latest_date',
        y='category',
        size='count',
        color='freshness_score',
        color_continuous_scale=[(0, '#FFEB3B'), (1, '#8BC34A')],  # Yellow to Green
        title=title,
        labels={
            'latest_date': 'Most Recent Mention',
            'category': 'Taxonomic Element',
            'freshness_score': 'Freshness Score'
        }
    )
    
    fig.update_layout(
        xaxis_title="Date of Most Recent Mention",
        yaxis_title="Taxonomic Element",
        height=600,
        margin=dict(l=150)  # Extra left margin for long category names
    )
    
    # Add hover information WITH THOUSANDS SEPARATORS
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>Date: %{x|%Y-%m-%d}<br>Count: %{marker.size:,}<br>Freshness: %{marker.color:.1f}<extra></extra>'
    )
    
    return fig


def create_freshness_drilldown_chart(
    df: pd.DataFrame, 
    selected_category: str,
    title: str = "Subcategory Freshness"
) -> go.Figure:
    """
    Create a drill-down chart for subcategories freshness.
    
    Args:
        df: DataFrame with subcategory freshness data
        selected_category: Selected parent category
        title: Chart title
        
    Returns:
        go.Figure: Plotly Figure object
    """
    if df.empty:
        return go.Figure().update_layout(title="No data available for subcategory freshness")
    
    # Filter for the selected category
    filtered_df = df[df['category'] == selected_category]
    
    if filtered_df.empty:
        return go.Figure().update_layout(title=f"No subcategories found for '{selected_category}'")
    
    # Sort by freshness score
    sorted_df = filtered_df.sort_values('freshness_score', ascending=False)
    
    # Create the bar chart for subcategories
    fig = px.bar(
        sorted_df,
        x='subcategory',
        y='freshness_score',
        color='freshness_score',
        color_continuous_scale=[(0, '#FFEB3B'), (1, '#8BC34A')],  # Yellow to Green
        title=title,
        labels={'subcategory': 'Subcategory', 'freshness_score': 'Freshness Score'},
        hover_data=['count', 'latest_date', 'recency_score', 'frequency_score']
    )
    
    # Rotate x-axis labels for better readability
    fig.update_layout(
        xaxis_tickangle=-45,
        xaxis_title='',
        yaxis_title='Freshness Score (0-100)',
        height=600,
        margin=dict(b=150)  # Extra bottom margin for rotated labels
    )
    
    # Add hover template WITH THOUSANDS SEPARATORS
    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>Freshness Score: %{y:.1f}<br>Count: %{customdata[0]:,}<br>Latest Date: %{customdata[1]}<br>Recency: %{customdata[2]:.1f}<br>Frequency: %{customdata[3]:.1f}<extra></extra>'
    )
    
    return fig