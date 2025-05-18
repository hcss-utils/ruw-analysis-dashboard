#!/usr/bin/env python
# coding: utf-8

"""
Burst visualization functions for the dashboard.
These functions create visualizations for burst detection results.
"""

import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any

def create_burst_heatmap(
    summary_df: pd.DataFrame,
    title: str = "Burst Intensity Heatmap",
    color_scale=None
) -> go.Figure:
    """
    Create a heatmap visualization of burst intensity by period.
    
    Args:
        summary_df: DataFrame with element, period, and burst intensity
        title: Chart title
        color_scale: Custom color scale for the heatmap
        
    Returns:
        go.Figure: Plotly Figure object
    """
    if summary_df.empty:
        return go.Figure().update_layout(title="No data available for burst heatmap")
    
    if color_scale is None:
        color_scale = [[0, '#f7f7f7'], [0.4, '#ffeb3b'], [0.7, '#ffa000'], [1, '#ff5252']]
    
    # Create pivot table for heatmap
    try:
        # Group by element and period, taking max intensity
        pivot_df = summary_df.pivot_table(
            index='element', 
            columns='period', 
            values='burst_intensity',
            aggfunc='max'
        ).fillna(0)
        
        # Sort rows by average burst intensity
        pivot_df['avg_intensity'] = pivot_df.mean(axis=1)
        pivot_df = pivot_df.sort_values('avg_intensity', ascending=False)
        pivot_df = pivot_df.drop(columns=['avg_intensity'])
        
        # Create heatmap
        fig = px.imshow(
            pivot_df,
            labels=dict(x="Period", y="Element", color="Burst Intensity"),
            x=pivot_df.columns,
            y=pivot_df.index,
            color_continuous_scale=color_scale,
            aspect="auto",
            title=title
        )
        
        # Update layout
        fig.update_layout(
            height=max(400, 30 * len(pivot_df) + 150),  # Dynamic height based on number of elements
            width=900,
            margin=dict(l=200, r=20, t=40, b=100),
            coloraxis_colorbar=dict(
                title="Burst<br>Intensity",
                tickvals=[0, 25, 50, 75, 100],
                ticktext=["0", "25", "50", "75", "100"]
            )
        )
        
        # Improve axis labels and formatting
        fig.update_xaxes(tickangle=-45)
        fig.update_yaxes(tickangle=0, automargin=True)
        
        # Add hover info
        fig.update_traces(
            hovertemplate='<b>%{y}</b><br>Period: %{x}<br>Burst Intensity: %{z:.1f}<extra></extra>'
        )
        
        return fig
        
    except Exception as e:
        logging.error(f"Error creating burst heatmap: {e}")
        return go.Figure().update_layout(title=f"Error creating burst heatmap: {e}")


def create_burst_summary_chart(
    summary_df: pd.DataFrame,
    title: str = "Top Burst Elements",
    top_n: int = 15
) -> go.Figure:
    """
    Create a bar chart showing elements with the highest burst intensity.
    
    Args:
        summary_df: DataFrame with element and max_burst_intensity
        title: Chart title
        top_n: Number of top elements to show
        
    Returns:
        go.Figure: Plotly Figure object
    """
    if summary_df.empty:
        return go.Figure().update_layout(title="No data available for burst summary")
    
    # Sort and take top N elements
    sorted_df = summary_df.sort_values('max_burst_intensity', ascending=False).head(top_n)
    
    # Create color gradient based on intensity
    colors = px.colors.sequential.Reds
    color_scale = [colors[int(i * (len(colors)-1) / 100)] for i in sorted_df['max_burst_intensity']]
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    # Add main bars for max burst intensity
    fig.add_trace(go.Bar(
        y=sorted_df['element'],
        x=sorted_df['max_burst_intensity'],
        orientation='h',
        marker_color=color_scale,
        name='Max Burst Intensity',
        hovertemplate='<b>%{y}</b><br>Max Burst: %{x:.1f}<br>Period: %{customdata[0]}<br>Total Count: %{customdata[1]:,}<extra></extra>',
        customdata=sorted_df[['max_burst_period', 'total_count']]
    ))
    
    # Add markers for average intensity
    fig.add_trace(go.Scatter(
        y=sorted_df['element'],
        x=sorted_df['avg_intensity'],
        mode='markers',
        marker=dict(
            symbol='diamond',
            size=10,
            color='rgba(0, 0, 0, 0.7)'
        ),
        name='Avg Intensity',
        hovertemplate='<b>%{y}</b><br>Avg Intensity: %{x:.1f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Burst Intensity',
        yaxis_title='',
        height=max(400, 25 * len(sorted_df) + 100),
        width=900,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        barmode='overlay'
    )
    
    return fig


def create_burst_timeline(
    burst_data: Dict[str, pd.DataFrame],
    title: str = "Burst Timeline",
    top_n: int = 8
) -> go.Figure:
    """
    Create a multi-line chart showing burst intensity over time for top elements.
    
    Args:
        burst_data: Dictionary mapping elements to DataFrames with burst data
        title: Chart title
        top_n: Number of top elements to show
        
    Returns:
        go.Figure: Plotly Figure object
    """
    if not burst_data:
        return go.Figure().update_layout(title="No data available for burst timeline")
    
    # Create Figure
    fig = go.Figure()
    
    # Calculate average burst intensity across periods for each element
    avg_intensities = {}
    for element, df in burst_data.items():
        if not df.empty and 'burst_intensity' in df.columns and 'period' in df.columns:
            avg_intensities[element] = df.groupby('period')['burst_intensity'].mean().mean()
    
    # Sort elements by average burst intensity and take top N
    top_elements = sorted(avg_intensities.keys(), key=lambda x: avg_intensities[x], reverse=True)[:top_n]
    
    # Add traces for each top element
    for i, element in enumerate(top_elements):
        df = burst_data[element]
        if df.empty or 'period' not in df.columns or 'burst_intensity' not in df.columns:
            continue
        
        # Aggregate by period to get one value per period
        period_data = df.groupby('period').agg({
            'burst_intensity': 'max',
            'count': 'sum'
        }).reset_index()
        
        # Use plotly's color sequences with some transparency
        color_idx = i % 10  # Cycle through 10 colors
        color = px.colors.qualitative.Plotly[color_idx]
        
        # Add line trace
        fig.add_trace(go.Scatter(
            x=period_data['period'],
            y=period_data['burst_intensity'],
            mode='lines+markers',
            name=element,
            line=dict(width=2, color=color),
            marker=dict(size=8, color=color),
            hovertemplate='<b>%{x}</b><br>Element: %{fullData.name}<br>Burst Intensity: %{y:.1f}<br>Count: %{customdata:,}<extra></extra>',
            customdata=period_data['count']
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Time Period',
        yaxis_title='Burst Intensity',
        height=500,
        width=900,
        margin=dict(l=20, r=20, t=40, b=50),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        hovermode="closest"
    )
    
    return fig


def create_burst_comparison_chart(
    taxonomy_summary: pd.DataFrame,
    keyword_summary: pd.DataFrame,
    entity_summary: pd.DataFrame,
    title: str = "Burst Comparison",
    top_n: int = 5
) -> go.Figure:
    """
    Create a comparison chart showing top bursting elements across all types.
    
    Args:
        taxonomy_summary: DataFrame with taxonomy burst summaries
        keyword_summary: DataFrame with keyword burst summaries
        entity_summary: DataFrame with named entity burst summaries
        title: Chart title
        top_n: Number of top elements to show from each type
        
    Returns:
        go.Figure: Plotly Figure object
    """
    # Check if we have any data
    if taxonomy_summary.empty and keyword_summary.empty and entity_summary.empty:
        return go.Figure().update_layout(title="No data available for burst comparison")
    
    fig = go.Figure()
    
    # Process taxonomy data
    if not taxonomy_summary.empty:
        top_taxonomy = taxonomy_summary.sort_values('max_burst_intensity', ascending=False).head(top_n)
        fig.add_trace(go.Bar(
            name='Taxonomy',
            x=[f"T: {elem}" for elem in top_taxonomy['element']],
            y=top_taxonomy['max_burst_intensity'],
            marker_color='#4caf50',  # Green
            hovertemplate='<b>%{x}</b><br>Burst Intensity: %{y:.1f}<br>Period: %{customdata[0]}<br>Count: %{customdata[1]:,}<extra></extra>',
            customdata=top_taxonomy[['max_burst_period', 'total_count']]
        ))
    
    # Process keyword data
    if not keyword_summary.empty:
        top_keywords = keyword_summary.sort_values('max_burst_intensity', ascending=False).head(top_n)
        fig.add_trace(go.Bar(
            name='Keywords',
            x=[f"K: {elem}" for elem in top_keywords['element']],
            y=top_keywords['max_burst_intensity'],
            marker_color='#2196f3',  # Blue
            hovertemplate='<b>%{x}</b><br>Burst Intensity: %{y:.1f}<br>Period: %{customdata[0]}<br>Count: %{customdata[1]:,}<extra></extra>',
            customdata=top_keywords[['max_burst_period', 'total_count']]
        ))
    
    # Process entity data
    if not entity_summary.empty:
        top_entities = entity_summary.sort_values('max_burst_intensity', ascending=False).head(top_n)
        fig.add_trace(go.Bar(
            name='Named Entities',
            x=[f"E: {elem}" for elem in top_entities['element']],
            y=top_entities['max_burst_intensity'],
            marker_color='#ff9800',  # Orange
            hovertemplate='<b>%{x}</b><br>Burst Intensity: %{y:.1f}<br>Period: %{customdata[0]}<br>Count: %{customdata[1]:,}<extra></extra>',
            customdata=top_entities[['max_burst_period', 'total_count']]
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='',
        yaxis_title='Maximum Burst Intensity',
        height=500,
        width=900,
        margin=dict(l=20, r=20, t=40, b=150),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        barmode='group',
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=10)
        )
    )
    
    return fig