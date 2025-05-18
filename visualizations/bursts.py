#!/usr/bin/env python
# coding: utf-8

"""
Burst visualization functions for the dashboard.
These functions create visualizations for burst detection results using CiteSpace-inspired styles.
"""

import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
from datetime import datetime

# Theme colors for consistency
THEME_BLUE = "#13376f"  # Main dashboard theme color
TAXONOMY_COLOR = '#4caf50'  # Green for taxonomy
KEYWORD_COLOR = '#2196f3'   # Blue for keywords
ENTITY_COLOR = '#ff9800'    # Orange for entities

def create_burst_heatmap(
    summary_df: pd.DataFrame,
    title: str = "Burst Intensity Heatmap",
    color_scale=None
) -> go.Figure:
    """
    Create a heatmap visualization of burst intensity by period, inspired by CiteSpace.
    
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
            margin=dict(l=200, r=20, t=40, b=100),
            coloraxis_colorbar=dict(
                title="Burst<br>Intensity",
                tickvals=[0, 25, 50, 75, 100],
                ticktext=["0", "25", "50", "75", "100"]
            ),
            plot_bgcolor='rgb(248, 248, 248)',
            paper_bgcolor='white',
            title_font=dict(size=16, color=THEME_BLUE, family="Arial, sans-serif"),
        )
        
        # Improve axis labels and formatting
        fig.update_xaxes(
            tickangle=-45, 
            showgrid=True,
            gridwidth=1,
            gridcolor='white',
            linecolor='rgb(204, 204, 204)',
            tickfont=dict(size=12)
        )
        
        fig.update_yaxes(
            tickangle=0, 
            automargin=True,
            showgrid=True,
            gridwidth=1,
            gridcolor='white',
            linecolor='rgb(204, 204, 204)',
            tickfont=dict(size=12)
        )
        
        # Add hover info
        fig.update_traces(
            hovertemplate='<b>%{y}</b><br>Period: %{x}<br>Burst Intensity: %{z:.1f}<extra></extra>'
        )
        
        return fig
        
    except Exception as e:
        logging.error(f"Error creating burst heatmap: {e}")
        return go.Figure().update_layout(title=f"Error creating burst heatmap: {e}")


def create_citespace_timeline(
    summary_df: pd.DataFrame,
    title: str = "CiteSpace-style Burst Timeline",
    color_scale=None
) -> go.Figure:
    """
    Create a burst timeline visualization similar to CiteSpace's citation burst diagram.
    
    Args:
        summary_df: DataFrame with element, period, and burst_intensity
        title: Chart title
        color_scale: Custom color scale for the bursts
        
    Returns:
        go.Figure: Plotly Figure object
    """
    if summary_df.empty:
        return go.Figure().update_layout(title="No data available for burst timeline")
    
    if color_scale is None:
        color_scale = px.colors.sequential.Reds
    
    fig = go.Figure()
    
    # Extract unique elements and periods
    elements = summary_df['element'].unique()
    periods = sorted(summary_df['period'].unique())
    
    # Sort elements by average burst intensity
    element_avg_intensity = {}
    for element in elements:
        element_df = summary_df[summary_df['element'] == element]
        element_avg_intensity[element] = element_df['burst_intensity'].mean()
    
    sorted_elements = sorted(element_avg_intensity.keys(), key=lambda x: element_avg_intensity[x], reverse=True)
    
    # For each element, create a horizontal bar showing bursts over time
    for i, element in enumerate(sorted_elements[:15]):  # Limit to top 15 for readability
        element_df = summary_df[summary_df['element'] == element]
        
        # Get color based on max intensity
        max_intensity = element_df['burst_intensity'].max()
        color_idx = min(int(max_intensity / 100 * (len(color_scale) - 1)), len(color_scale) - 1)
        color = color_scale[color_idx]
        
        y_position = len(sorted_elements) - i  # Reverse order for top elements at the top
        
        # Draw baseline (light gray line)
        fig.add_trace(go.Scatter(
            x=periods,
            y=[y_position] * len(periods),
            mode='lines',
            line=dict(color='rgb(220, 220, 220)', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add bursts as thicker colored segments
        for _, row in element_df.iterrows():
            if row['burst_intensity'] > 0:
                period = row['period']
                burst_intensity = row['burst_intensity']
                try:
                    period_idx = periods.index(period)
                    
                    # Calculate color based on intensity
                    relative_intensity = burst_intensity / 100
                    color_idx = min(int(relative_intensity * (len(color_scale) - 1)), len(color_scale) - 1)
                    segment_color = color_scale[color_idx]
                    
                    # Line width based on intensity (min 3, max 12)
                    line_width = 3 + (burst_intensity / 100) * 9
                    
                    # Determine segment endpoints
                    x_start = periods[period_idx]
                    
                    # Handle last period
                    if period_idx < len(periods) - 1:
                        x_end = periods[period_idx + 1]
                    else:
                        # For last period, extend a bit
                        if isinstance(periods[0], str):
                            x_end = periods[-1] + " (next)"
                        else:
                            x_end = periods[-1] + 1
                            
                    # Add a segment
                    fig.add_trace(go.Scatter(
                        x=[x_start, x_end],
                        y=[y_position, y_position],
                        mode='lines',
                        line=dict(color=segment_color, width=line_width),
                        name=element if period_idx == 0 else "",
                        showlegend=period_idx == 0,  # Only show in legend for first period
                        legendgroup=element,
                        hovertemplate=f"<b>{element}</b><br>Period: {period}<br>Burst Intensity: {burst_intensity:.1f}<extra></extra>"
                    ))
                    
                except (ValueError, IndexError) as e:
                    logging.warning(f"Error processing period {period} for element {element}: {e}")
    
    # Add element labels on the left side
    for i, element in enumerate(sorted_elements[:15]):
        y_position = len(sorted_elements) - i
        fig.add_annotation(
            x=periods[0],  # Position at first period
            y=y_position,
            text=element,
            showarrow=False,
            xanchor='right',
            xshift=-10,
            font=dict(size=10)
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Time Period',
        yaxis_title='',
        height=max(400, 30 * len(sorted_elements[:15]) + 100),
        margin=dict(l=200, r=20, t=40, b=50),
        showlegend=False,  # Hide legend as we have element labels
        plot_bgcolor='white',
        paper_bgcolor='white',
        title_font=dict(size=16, color=THEME_BLUE, family="Arial, sans-serif"),
        hovermode="closest",
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgb(248, 248, 248)',
            tickangle=-45
        ),
        yaxis=dict(
            showticklabels=False,  # Hide y-axis labels as we added annotations
            showgrid=True,
            gridwidth=1,
            gridcolor='rgb(248, 248, 248)',
        )
    )
    
    return fig


def create_burst_summary_chart(
    summary_df: pd.DataFrame,
    title: str = "Top Burst Elements",
    top_n: int = 15,
    color: str = None
) -> go.Figure:
    """
    Create a bar chart showing elements with the highest burst intensity,
    with CiteSpace-inspired styling.
    
    Args:
        summary_df: DataFrame with element and max_burst_intensity
        title: Chart title
        top_n: Number of top elements to show
        color: Base color for the bars
        
    Returns:
        go.Figure: Plotly Figure object
    """
    if summary_df.empty:
        return go.Figure().update_layout(title="No data available for burst summary")
    
    # Sort and take top N elements
    sorted_df = summary_df.sort_values('max_burst_intensity', ascending=False).head(top_n)
    
    # Create color gradient based on intensity
    if color is None:
        colors = px.colors.sequential.Reds
    else:
        # Create a custom color scale based on the provided color
        import colorsys
        
        # Helper function to convert hex to RGB without matplotlib
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))
        
        # Convert hex to RGB
        r, g, b = hex_to_rgb(color)
        # Convert RGB to HSV
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        
        # Create variations with different saturation and value
        colors = []
        for i in range(10):
            # Adjust saturation and value based on position
            new_s = min(1.0, s * (0.5 + 0.5 * i/9))
            new_v = min(1.0, v * (0.5 + 0.5 * i/9))
            # Convert back to RGB
            new_r, new_g, new_b = colorsys.hsv_to_rgb(h, new_s, new_v)
            # Convert to hex
            colors.append(f'rgb({int(new_r*255)}, {int(new_g*255)}, {int(new_b*255)})')
    
    # Determine color for each bar based on intensity
    color_scale = []
    for intensity in sorted_df['max_burst_intensity']:
        idx = min(int(intensity / 100 * (len(colors)-1)), len(colors)-1)
        color_scale.append(colors[idx])
    
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
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        barmode='overlay',
        plot_bgcolor='rgb(248, 248, 248)',
        paper_bgcolor='white',
        title_font=dict(size=16, color=THEME_BLUE, family="Arial, sans-serif"),
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='white',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgb(204, 204, 204)'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='white',
        automargin=True
    )
    
    return fig


def create_burst_timeline(
    burst_data: Dict[str, pd.DataFrame],
    title: str = "Burst Timeline",
    top_n: int = 8,
    color_base: str = None
) -> go.Figure:
    """
    Create a multi-line chart showing burst intensity over time for top elements,
    with CiteSpace-inspired styling.
    
    Args:
        burst_data: Dictionary mapping elements to DataFrames with burst data
        title: Chart title
        top_n: Number of top elements to show
        color_base: Base color to use for the lines
        
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
    
    # Use color base or default to a plotly qualitative scale
    if color_base:
        # Create a custom color scale based on the provided base color
        import colorsys
        
        # Helper function to convert hex to RGB without matplotlib
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))
        
        r, g, b = hex_to_rgb(color_base)
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        
        colors = []
        for i in range(top_n):
            # Rotate hue slightly for variety while keeping in same color family
            new_h = (h + (i * 0.05)) % 1.0
            # Vary saturation and value for contrast
            new_s = min(1.0, s * (0.7 + 0.3 * (i % 3)/2))
            new_v = min(1.0, v * (0.7 + 0.3 * (i % 2)))
            
            new_r, new_g, new_b = colorsys.hsv_to_rgb(new_h, new_s, new_v)
            colors.append(f'rgb({int(new_r*255)}, {int(new_g*255)}, {int(new_b*255)})')
    else:
        colors = px.colors.qualitative.Plotly
    
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
        
        # Use color from our color list
        color_idx = i % len(colors)
        color = colors[color_idx]
        
        # Add line trace
        fig.add_trace(go.Scatter(
            x=period_data['period'],
            y=period_data['burst_intensity'],
            mode='lines+markers',
            name=element,
            line=dict(width=3, color=color),
            marker=dict(
                size=10,
                color=color,
                line=dict(width=1, color='white')
            ),
            hovertemplate='<b>%{x}</b><br>Element: %{fullData.name}<br>Burst Intensity: %{y:.1f}<br>Count: %{customdata:,}<extra></extra>',
            customdata=period_data['count']
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Time Period',
        yaxis_title='Burst Intensity',
        height=500,
        margin=dict(l=20, r=20, t=40, b=50),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        hovermode="closest",
        plot_bgcolor='rgb(248, 248, 248)',
        paper_bgcolor='white',
        title_font=dict(size=16, color=THEME_BLUE, family="Arial, sans-serif"),
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='white',
        tickangle=-45
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='white',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgb(204, 204, 204)'
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
    Create a comparison chart showing top bursting elements across all types,
    with CiteSpace-inspired styling.
    
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
            marker_color=TAXONOMY_COLOR,  # Green
            marker_line=dict(width=1, color='white'),
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
            marker_color=KEYWORD_COLOR,  # Blue
            marker_line=dict(width=1, color='white'),
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
            marker_color=ENTITY_COLOR,  # Orange
            marker_line=dict(width=1, color='white'),
            hovertemplate='<b>%{x}</b><br>Burst Intensity: %{y:.1f}<br>Period: %{customdata[0]}<br>Count: %{customdata[1]:,}<extra></extra>',
            customdata=top_entities[['max_burst_period', 'total_count']]
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='',
        yaxis_title='Maximum Burst Intensity',
        height=500,
        margin=dict(l=20, r=20, t=40, b=150),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor=THEME_BLUE,
            borderwidth=1
        ),
        barmode='group',
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=10),
            showgrid=True,
            gridwidth=1,
            gridcolor='white'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='white',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgb(204, 204, 204)'
        ),
        plot_bgcolor='rgb(248, 248, 248)',
        paper_bgcolor='white',
        title_font=dict(size=16, color=THEME_BLUE, family="Arial, sans-serif"),
    )
    
    return fig