#!/usr/bin/env python
# coding: utf-8

"""
Timeline visualization helpers for the dashboard.
"""

import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional


def create_timeline_chart(df: pd.DataFrame, title: str = "Timeline Distribution") -> go.Figure:
    """
    Create a timeline chart from a DataFrame.
    
    Args:
        df: DataFrame with timeline data
        title: Chart title
        
    Returns:
        go.Figure: Plotly Figure object
    """
    if df.empty:
        return go.Figure().update_layout(title="No data available for timeline")
    
    # Ensure date column is properly formatted
    df['month'] = pd.to_datetime(df['month'])
    
    # If 'count' column appears more than once, rename it to avoid duplicate column names
    df_cols = df.columns.tolist()
    if df_cols.count('count') > 1:
        # Rename the columns to make them unique
        renamed_cols = []
        count_idx = 0
        for col in df_cols:
            if col == 'count':
                renamed_cols.append(f'count_{count_idx}')
                count_idx += 1
            else:
                renamed_cols.append(col)
        df.columns = renamed_cols
        
        # Use the first 'count' column for the line chart
        count_col = 'count_0'
    else:
        count_col = 'count'
    
    # Create the line chart
    fig = px.line(
        df, 
        x='month', 
        y=count_col,
        title=title,
        markers=True,
        labels={'month': 'Month', count_col: 'Number of Items'}
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Count",
        hovermode="x unified",
        height=400,
        width=1200,  # Increased width for better visualization
        margin=dict(t=50, l=0, r=0, b=20),  # Adjusted margins
        autosize=True  # Allow resizing
    )
    
    # Add hover information WITH THOUSANDS SEPARATORS
    fig.update_traces(
        hovertemplate='<b>%{x|%Y-%m}</b><br>Count: %{y:,}<extra></extra>'
    )
    
    return fig


def create_comparison_timeline(
    df_a: pd.DataFrame, 
    df_b: pd.DataFrame, 
    title: str = "Comparison Timeline", 
    name_a: str = "Group A", 
    name_b: str = "Group B"
) -> go.Figure:
    """
    Create a comparison timeline with two datasets.
    
    Args:
        df_a: DataFrame for the first dataset
        df_b: DataFrame for the second dataset
        title: Chart title
        name_a: Name for the first dataset
        name_b: Name for the second dataset
        
    Returns:
        go.Figure: Plotly Figure object
    """
    if df_a.empty and df_b.empty:
        return go.Figure().update_layout(title="No data available for timeline comparison")
    
    # Create Figure
    fig = go.Figure()
    
    # Add traces for df_a if not empty
    if not df_a.empty:
        # Ensure date column is properly formatted
        df_a['month'] = pd.to_datetime(df_a['month'])
        
        # Check for duplicate count columns
        df_a_cols = df_a.columns.tolist()
        if df_a_cols.count('count') > 1:
            # Rename the columns to make them unique
            renamed_cols = []
            count_idx = 0
            for col in df_a_cols:
                if col == 'count':
                    renamed_cols.append(f'count_{count_idx}')
                    count_idx += 1
                else:
                    renamed_cols.append(col)
            df_a.columns = renamed_cols
            count_col_a = 'count_0'
        else:
            count_col_a = 'count'
            
        fig.add_trace(go.Scatter(
            x=df_a['month'],
            y=df_a[count_col_a],
            mode='lines+markers',
            name=name_a,
            line=dict(width=3),
            marker=dict(size=8),
            hovertemplate='<b>%{x|%Y-%m}</b><br>Count: %{y:,}<extra></extra>'
        ))
    
    # Add traces for df_b if not empty
    if not df_b.empty:
        # Ensure date column is properly formatted
        df_b['month'] = pd.to_datetime(df_b['month'])
        
        # Check for duplicate count columns
        df_b_cols = df_b.columns.tolist()
        if df_b_cols.count('count') > 1:
            # Rename the columns to make them unique
            renamed_cols = []
            count_idx = 0
            for col in df_b_cols:
                if col == 'count':
                    renamed_cols.append(f'count_{count_idx}')
                    count_idx += 1
                else:
                    renamed_cols.append(col)
            df_b.columns = renamed_cols
            count_col_b = 'count_0'
        else:
            count_col_b = 'count'
            
        fig.add_trace(go.Scatter(
            x=df_b['month'],
            y=df_b[count_col_b],
            mode='lines+markers',
            name=name_b,
            line=dict(width=3, dash='dash'),
            marker=dict(size=8),
            hovertemplate='<b>%{x|%Y-%m}</b><br>Count: %{y:,}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Count",
        hovermode="x unified",
        height=500,
        width=1200,
        margin=dict(t=50, l=0, r=0, b=20),
        autosize=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_freshness_timeline(df: pd.DataFrame, title: str = "Freshness Timeline") -> go.Figure:
    """
    Create a timeline chart with freshness score heatmap.
    
    Args:
        df: DataFrame with timeline and freshness data
        title: Chart title
        
    Returns:
        go.Figure: Plotly Figure object
    """
    if df.empty:
        return go.Figure().update_layout(title="No data available for freshness timeline")
    
    # Ensure date column is properly formatted
    df['latest_date'] = pd.to_datetime(df['latest_date'])
    
    # Create scatter plot with color intensity representing freshness
    fig = px.scatter(
        df,
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
        width=1200,
        margin=dict(t=50, l=0, r=0, b=20),
        autosize=True
    )
    
    # Add hover information WITH THOUSANDS SEPARATORS
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>Date: %{x|%Y-%m-%d}<br>Count: %{marker.size:,}<br>Freshness: %{marker.color:.1f}<extra></extra>'
    )
    
    return fig