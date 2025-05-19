#!/usr/bin/env python
# coding: utf-8

"""
Burst visualization functions for the dashboard.
These functions create visualizations for burst detection results using CiteSpace-inspired styles.
Includes enhanced visualizations for timeline, co-occurrence networks, and predictive analysis.
Supports loading data from both database and CSV files.
"""

import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import networkx as nx
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import os
import glob
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from itertools import combinations
import json

# Theme colors for consistency
THEME_BLUE = "#13376f"  # Main dashboard theme color
TAXONOMY_COLOR = '#4caf50'  # Green for taxonomy
KEYWORD_COLOR = '#2196f3'   # Blue for keywords
ENTITY_COLOR = '#ff9800'    # Orange for entities

# Additional colors for network and timeline visualizations
NETWORK_COLORS = px.colors.qualitative.D3
TIMELINE_BG_COLOR = '#f9f9f9'
LINE_COLOR = '#cccccc'
EVENT_COLOR = '#e41a1c'
PREDICTION_COLOR = '#9467bd'
CONFIDENCE_INTERVAL_COLOR = 'rgba(148, 103, 189, 0.2)'

###########################################
# Data Loading Functions
###########################################

def load_data_from_csv(data_type: str = 'keywords', file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load frequency data from CSV files.
    
    Args:
        data_type: Type of data to load ('keywords', 'named_entities')
        file_path: Optional specific file path, otherwise most recent file will be used
        
    Returns:
        pd.DataFrame: DataFrame with loaded data
    """
    if file_path and os.path.exists(file_path):
        df = pd.read_csv(file_path)
        logging.info(f"Loaded data from specified file: {file_path}")
        return df
    
    # Find the most recent file of the specified type
    pattern = '*'  # Default pattern
    if data_type == 'keywords':
        pattern = 'keyword_frequencies_*.csv'
    elif data_type == 'named_entities':
        pattern = 'named_entity_frequencies_*.csv'
    
    # Get the list of matching files
    matching_files = glob.glob(pattern)
    
    # If no files found, return empty DataFrame
    if not matching_files:
        logging.warning(f"No {data_type} CSV files found with pattern: {pattern}")
        return pd.DataFrame()
    
    # Sort by modification time (most recent first)
    most_recent_file = max(matching_files, key=os.path.getmtime)
    
    # Load the CSV file
    try:
        df = pd.read_csv(most_recent_file)
        logging.info(f"Loaded data from most recent file: {most_recent_file}")
        return df
    except Exception as e:
        logging.error(f"Error loading CSV file {most_recent_file}: {e}")
        return pd.DataFrame()


def prepare_data_for_burst_analysis(df: pd.DataFrame, data_type: str) -> Dict[str, pd.DataFrame]:
    """
    Prepare CSV data for burst analysis by converting it to the expected format.
    
    Args:
        df: DataFrame with frequency data from CSV
        data_type: Type of data ('keywords', 'named_entities')
        
    Returns:
        Dict mapping elements to DataFrames with date and count data
    """
    if df.empty:
        return {}
    
    result = {}
    
    try:
        # Structure depends on the data type
        if data_type == 'keywords':
            # Expected columns: Keyword, Count, Relative Frequency (%)
            for _, row in df.iterrows():
                keyword = row.get('Keyword')
                count = row.get('Count')
                if keyword and count:
                    # Create a simple DataFrame with one row per keyword for current date
                    result[keyword] = pd.DataFrame({
                        'date': [datetime.now().strftime('%Y-%m-%d')],
                        'element': [keyword],
                        'count': [count]
                    })
        
        elif data_type == 'named_entities':
            # Expected columns: Entity Type, Entity Value, Count, Relative Frequency (%)
            for _, row in df.iterrows():
                entity_type = row.get('Entity Type', '')
                entity_value = row.get('Entity Value', '')
                count = row.get('Count')
                if entity_type and entity_value and count:
                    element_name = f"{entity_value} ({entity_type})"
                    result[element_name] = pd.DataFrame({
                        'date': [datetime.now().strftime('%Y-%m-%d')],
                        'element': [element_name],
                        'count': [count]
                    })
    
    except Exception as e:
        logging.error(f"Error preparing {data_type} data for burst analysis: {e}")
    
    return result


def combine_data_sources(db_data: Dict[str, Dict[str, pd.DataFrame]], 
                        csv_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Combine data from database and CSV files into a single structure for visualization.
    
    Args:
        db_data: Data from database queries
        csv_data: Data from CSV files
        
    Returns:
        Dict with combined data
    """
    combined_data = {data_type: {} for data_type in set(db_data.keys()).union(csv_data.keys())}
    
    # Combine data from both sources
    for data_type in combined_data.keys():
        # Add database data if available
        if data_type in db_data:
            for element, df in db_data[data_type].items():
                combined_data[data_type][element] = df.copy()
        
        # Add or update with CSV data if available
        if data_type in csv_data:
            for element, df in csv_data[data_type].items():
                if element in combined_data[data_type]:
                    # Element exists in both sources, append CSV data
                    combined_data[data_type][element] = pd.concat([combined_data[data_type][element], df])
                else:
                    # Element only in CSV data
                    combined_data[data_type][element] = df.copy()
    
    return combined_data

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


def create_enhanced_citespace_timeline(
    summary_df: pd.DataFrame,
    historical_events: Optional[List[Dict[str, Any]]] = None,
    document_links: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    title: str = "Enhanced CiteSpace-style Burst Timeline",
    color_scale=None,
    show_annotations: bool = True
) -> go.Figure:
    """
    Create an enhanced burst timeline visualization similar to CiteSpace's citation burst diagram,
    with added historical events and document links.
    
    Args:
        summary_df: DataFrame with element, period, and burst_intensity
        historical_events: List of dictionaries with historical events information
            (each dict should have 'date', 'event', and optionally 'impact')
        document_links: Dictionary mapping elements to lists of document references
        title: Chart title
        color_scale: Custom color scale for the bursts
        show_annotations: Whether to show annotations for historical events
        
    Returns:
        go.Figure: Plotly Figure object
    """
    if summary_df.empty:
        return go.Figure().update_layout(title="No data available for burst timeline")
    
    if color_scale is None:
        color_scale = px.colors.sequential.Reds
    
    fig = go.Figure()
    
    # Add horizontal reference line to represent timeline
    periods = sorted(summary_df['period'].unique())
    fig.add_trace(go.Scatter(
        x=periods,
        y=[0] * len(periods),
        mode='lines',
        line=dict(color='rgba(0,0,0,0.5)', width=2),
        name="Timeline",
        hoverinfo='skip'
    ))
    
    # Extract unique elements and periods
    elements = summary_df['element'].unique()
    
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
        
        y_position = i+1  # Place each element on its own level
        
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
                    
                    # Document link information
                    hover_extra = ""
                    if document_links and element in document_links and document_links[element]:
                        period_docs = [doc for doc in document_links[element] 
                                      if doc.get('period') == period]
                        if period_docs:
                            hover_extra = f"<br>Documents: {len(period_docs)}"
                    
                    # Add a segment
                    fig.add_trace(go.Scatter(
                        x=[x_start, x_end],
                        y=[y_position, y_position],
                        mode='lines',
                        line=dict(color=segment_color, width=line_width),
                        name=element if period_idx == 0 else "",
                        showlegend=period_idx == 0,  # Only show in legend for first period
                        legendgroup=element,
                        hovertemplate=f"<b>{element}</b><br>Period: {period}<br>Burst Intensity: {burst_intensity:.1f}{hover_extra}<extra></extra>"
                    ))
                    
                    # Add markers for document links (if available)
                    if document_links and element in document_links:
                        period_docs = [doc for doc in document_links[element] 
                                      if doc.get('period') == period]
                        if period_docs:
                            fig.add_trace(go.Scatter(
                                x=[x_start],
                                y=[y_position],
                                mode='markers',
                                marker=dict(
                                    symbol='circle',
                                    size=8,
                                    color='white',
                                    line=dict(color=segment_color, width=2)
                                ),
                                name=f"{element} documents",
                                showlegend=False,
                                hovertemplate=f"<b>{element}</b><br>Period: {period}<br>{len(period_docs)} related documents<extra></extra>"
                            ))
                    
                except (ValueError, IndexError) as e:
                    logging.warning(f"Error processing period {period} for element {element}: {e}")
    
    # Add element labels on the left side
    for i, element in enumerate(sorted_elements[:15]):
        y_position = i+1
        fig.add_annotation(
            x=periods[0],  # Position at first period
            y=y_position,
            text=element,
            showarrow=False,
            xanchor='right',
            xshift=-10,
            font=dict(size=10)
        )
    
    # Add historical events as vertical lines with annotations
    if historical_events and show_annotations:
        for event in historical_events:
            event_period = event.get('period')
            if event_period in periods:
                event_name = event.get('event', 'Event')
                event_impact = event.get('impact', 1.0)  # Impact scale (0.0 to 1.0)
                
                # Add vertical line
                fig.add_shape(
                    type="line",
                    x0=event_period,
                    y0=0,
                    x1=event_period,
                    y1=len(sorted_elements[:15]) + 1,
                    line=dict(
                        color=EVENT_COLOR,
                        width=2,
                        dash="dash"
                    )
                )
                
                # Add annotation
                fig.add_annotation(
                    x=event_period,
                    y=len(sorted_elements[:15]) + 0.5,
                    text=event_name,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=EVENT_COLOR,
                    font=dict(size=10, color=EVENT_COLOR),
                    align="center",
                    textangle=-90,
                    bordercolor=EVENT_COLOR,
                    borderwidth=1,
                    borderpad=4,
                    bgcolor="white",
                    opacity=0.8
                )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Time Period',
        yaxis_title='',
        height=max(500, 30 * len(sorted_elements[:15]) + 100),
        margin=dict(l=180, r=20, t=60, b=50),
        showlegend=False,  # Hide legend as we have element labels
        plot_bgcolor=TIMELINE_BG_COLOR,
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
            range=[0, len(sorted_elements[:15]) + 1.5]  # Leave room for annotations
        )
    )
    
    # Add interactivity instructions
    fig.add_annotation(
        xref='paper',
        yref='paper',
        x=0.5,
        y=1.05,
        text="Hover for details | Click on timeline points for document links",
        showarrow=False,
        font=dict(size=10, color='gray'),
        opacity=0.7
    )
    
    return fig


###########################################
# Predictive Visualizations
###########################################

def create_predictive_visualization(
    burst_data: Dict[str, Dict[str, pd.DataFrame]],
    prediction_periods: int = 2,
    confidence_level: float = 0.9,
    min_periods_for_prediction: int = 4,
    title: str = "Burst Trend Prediction",
    top_n: int = 5
) -> go.Figure:
    """
    Create a predictive visualization that forecasts future burst intensities based on historical trends.
    
    Args:
        burst_data: Dictionary mapping data types to dictionaries of element burst DataFrames
        prediction_periods: Number of future periods to predict
        confidence_level: Confidence level for prediction intervals
        min_periods_for_prediction: Minimum number of periods required for prediction
        title: Chart title
        top_n: Number of top elements to show
        
    Returns:
        go.Figure: Plotly Figure object with predictions
    """
    # Combine all data types and find elements with strong bursts
    all_elements = []
    
    for data_type, elements in burst_data.items():
        for element_name, df in elements.items():
            if df.empty or 'period' not in df.columns or 'burst_intensity' not in df.columns:
                continue
            
            # Only include elements with enough data points for prediction
            if len(df['period'].unique()) < min_periods_for_prediction:
                continue
            
            avg_intensity = df['burst_intensity'].mean()
            max_intensity = df['burst_intensity'].max()
            
            # Qualify the element name with its data type
            if data_type == 'taxonomy':
                qualified_name = f"T: {element_name}"
                color = TAXONOMY_COLOR
            elif data_type == 'keywords':
                qualified_name = f"K: {element_name}"
                color = KEYWORD_COLOR
            else:
                qualified_name = f"E: {element_name}"
                color = ENTITY_COLOR
            
            all_elements.append({
                'element': qualified_name,
                'data_type': data_type,
                'data': df,
                'avg_intensity': avg_intensity,
                'max_intensity': max_intensity,
                'color': color
            })
    
    # Sort by average intensity and take top N
    sorted_elements = sorted(all_elements, key=lambda x: x['avg_intensity'], reverse=True)[:top_n]
    
    if not sorted_elements:
        return go.Figure().update_layout(title="Insufficient data for prediction")
    
    # Get all unique periods from the data to establish the x-axis
    all_periods = set()
    for elem in sorted_elements:
        all_periods.update(elem['data']['period'].unique())
    
    # Sort periods (assuming they can be sorted - strings or dates)
    try:
        all_periods = sorted(all_periods)
    except:
        all_periods = list(all_periods)  # If they can't be sorted, just use as-is
    
    # Create prediction periods
    if isinstance(all_periods[0], str):
        # String periods - handle by appending "+1", "+2" etc.
        next_periods = [f"{all_periods[-1]} +{i+1}" for i in range(prediction_periods)]
    else:
        # Numeric periods - assume they're equally spaced
        step = all_periods[1] - all_periods[0] if len(all_periods) > 1 else 1
        next_periods = [all_periods[-1] + step * (i+1) for i in range(prediction_periods)]
    
    # Prepare figure
    fig = go.Figure()
    
    # Add light vertical grid lines
    for period in all_periods + next_periods:
        fig.add_shape(
            type="line",
            x0=period,
            y0=0,
            x1=period,
            y1=100,  # Max intensity is 100
            line=dict(
                color="rgba(200,200,200,0.5)",
                width=1
            ),
            layer="below"
        )
    
    # Add prediction zone background
    fig.add_shape(
        type="rect",
        x0=all_periods[-1],
        y0=0,
        x1=next_periods[-1],
        y1=100,  # Max intensity is 100
        fillcolor="rgba(230,230,230,0.5)",
        line=dict(width=0),
        layer="below"
    )
    
    # Add "Prediction Zone" annotation
    fig.add_annotation(
        x=(all_periods[-1] + next_periods[-1]) / 2,
        y=95,
        text="Prediction Zone",
        showarrow=False,
        font=dict(size=12, color="gray"),
        opacity=0.7,
        bgcolor="rgba(255,255,255,0.7)"
    )
    
    # For each element, create a line with prediction
    for elem_data in sorted_elements:
        element = elem_data['element']
        df = elem_data['data']
        color = elem_data['color']
        
        # Group by period to get max intensity per period
        period_data = df.groupby('period')['burst_intensity'].max().reset_index()
        
        # Ensure data for all periods (fill missing with zeros)
        full_period_data = {period: 0 for period in all_periods}
        for _, row in period_data.iterrows():
            full_period_data[row['period']] = row['burst_intensity']
        
        # Create X and Y values for plotting
        x_values = list(all_periods)
        y_values = [full_period_data[period] for period in all_periods]
        
        # Perform linear regression for prediction
        X = np.arange(len(x_values)).reshape(-1, 1)  # Convert to numpy array for regression
        y = np.array(y_values)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Make predictions for future periods
        future_X = np.arange(len(x_values), len(x_values) + prediction_periods).reshape(-1, 1)
        predictions = model.predict(future_X)
        
        # Apply bounds to predictions (0-100%)
        predictions = np.clip(predictions, 0, 100)
        
        # Calculate confidence intervals
        if len(x_values) >= 3:  # Need at least 3 points for meaningful confidence interval
            # Calculate prediction error
            y_pred = model.predict(X)
            residuals = y - y_pred
            mse = np.mean(residuals**2)
            std_error = np.sqrt(mse)
            
            # Calculate t-statistic for confidence level
            from scipy import stats
            t_value = stats.t.ppf((1 + confidence_level) / 2, len(x_values) - 2)
            
            # Calculate confidence intervals for predictions
            conf_interval = t_value * std_error * np.sqrt(1 + 1/len(x_values) + 
                                                       (future_X - np.mean(X))**2 / np.sum((X - np.mean(X))**2))
            
            lower_bound = predictions - conf_interval
            upper_bound = predictions + conf_interval
            
            # Apply bounds to confidence intervals
            lower_bound = np.clip(lower_bound, 0, 100)
            upper_bound = np.clip(upper_bound, 0, 100)
        else:
            # Simple confidence intervals if not enough data
            lower_bound = predictions * 0.7
            upper_bound = predictions * 1.3
            upper_bound = np.clip(upper_bound, 0, 100)
        
        # Add historical data line
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines+markers',
            name=element,
            line=dict(color=color, width=2),
            marker=dict(size=8, color=color, line=dict(width=1, color='white')),
            hovertemplate='<b>%{x}</b><br>Element: %{fullData.name}<br>Burst Intensity: %{y:.1f}<extra></extra>'
        ))
        
        # Add prediction line (dashed)
        fig.add_trace(go.Scatter(
            x=next_periods,
            y=predictions,
            mode='lines+markers',
            name=f"{element} (predicted)",
            line=dict(color=color, width=2, dash='dash'),
            marker=dict(size=8, color=color, symbol='diamond', line=dict(width=1, color='white')),
            hovertemplate='<b>%{x}</b><br>Element: %{fullData.name}<br>Predicted Intensity: %{y:.1f}<extra></extra>',
            showlegend=False
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=next_periods + next_periods[::-1],
            y=np.concatenate([upper_bound, lower_bound[::-1]]),
            fill='toself',
            fillcolor=color.replace(')', ', 0.2)').replace('rgb', 'rgba'),
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Time Period',
        yaxis_title='Burst Intensity',
        height=600,
        margin=dict(l=20, r=20, t=50, b=50),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        hovermode="closest",
        plot_bgcolor=TIMELINE_BG_COLOR,
        paper_bgcolor='white',
        title_font=dict(size=16, color=THEME_BLUE, family="Arial, sans-serif"),
        xaxis=dict(showgrid=False, tickangle=-45),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='white',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgb(204, 204, 204)',
            range=[0, 100]
        )
    )
    
    # Add correlation annotation
    fig.add_annotation(
        xref='paper',
        yref='paper',
        x=0.01,
        y=0.01,
        text="Note: Predictions based on linear trends. Shaded areas show confidence intervals.",
        showarrow=False,
        font=dict(size=10, color="gray"),
        align="left",
        bgcolor="rgba(255,255,255,0.7)"
    )
    
    return fig


###########################################
# Historical Events and Document Linking
###########################################

def load_historical_events(file_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load historical events from a JSON file to annotate visualizations.
    
    Args:
        file_path: Path to the JSON file containing historical events
        
    Returns:
        List of dictionaries containing historical event data
    """
    default_events = [
        {
            "date": "2022-02-24",
            "period": "Feb 2022",  # Will be matched with period labels in visualizations
            "event": "Russian Invasion of Ukraine Begins",
            "impact": 1.0,  # Scale from 0.0 to 1.0 for importance
            "description": "Russia launches a full-scale invasion of Ukraine."
        },
        {
            "date": "2022-04-03",
            "period": "Apr 2022",
            "event": "Bucha Massacre Revealed",
            "impact": 0.9,
            "description": "Discovery of civilian killings in Bucha after Russian withdrawal."
        },
        {
            "date": "2022-09-21",
            "period": "Sep 2022",
            "event": "Russian Mobilization",
            "impact": 0.8,
            "description": "Russia announces partial military mobilization."
        },
        {
            "date": "2023-06-06",
            "period": "Jun 2023",
            "event": "Kakhovka Dam Collapse",
            "impact": 0.7,
            "description": "Massive flooding after the collapse of the Kakhovka Dam."
        },
        {
            "date": "2023-08-23",
            "period": "Aug 2023",
            "event": "Wagner Group Leader Death",
            "impact": 0.6,
            "description": "Yevgeny Prigozhin reportedly killed in plane crash."
        }
    ]
    
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                events = json.load(f)
            logging.info(f"Loaded {len(events)} historical events from {file_path}")
            return events
        except Exception as e:
            logging.error(f"Error loading historical events from {file_path}: {e}")
            return default_events
    else:
        logging.info("Using default historical events")
        return default_events


def prepare_document_links(burst_data: Dict[str, Dict[str, pd.DataFrame]], 
                         document_ids: Optional[Dict[str, List[int]]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Prepare document links for interactive visualization elements.
    
    Args:
        burst_data: Dictionary mapping data types to dictionaries of element burst DataFrames
        document_ids: Dictionary mapping elements to lists of document IDs (optional)
        
    Returns:
        Dict mapping elements to lists of document references
    """
    document_links = {}
    
    # If document IDs are provided, use them to link
    if document_ids:
        for data_type, elements in burst_data.items():
            for element_name, df in elements.items():
                if element_name in document_ids:
                    doc_links = []
                    for period in df['period'].unique():
                        period_df = df[df['period'] == period]
                        
                        # Determine how many documents to link for this period
                        # (up to 5 per period for the demo)
                        num_docs = min(5, len(document_ids[element_name]))
                        
                        for i in range(num_docs):
                            doc_id = document_ids[element_name][i % len(document_ids[element_name])]
                            doc_links.append({
                                'period': period,
                                'document_id': doc_id,
                                'title': f"Document {doc_id} related to {element_name}",
                                'url': f"#document-{doc_id}"
                            })
                    
                    document_links[element_name] = doc_links
    else:
        # Without specific document IDs, create sample document links
        for data_type, elements in burst_data.items():
            for element_name, df in elements.items():
                if not df.empty and 'period' in df.columns:
                    doc_links = []
                    for period in df['period'].unique():
                        period_df = df[df['period'] == period]
                        burst_intensity = period_df['burst_intensity'].max()
                        
                        # Only create links for periods with significant bursts
                        if burst_intensity >= 50:
                            # Number of sample documents proportional to burst intensity
                            num_docs = max(1, int(burst_intensity / 20))
                            
                            for i in range(num_docs):
                                doc_id = 1000 + hash(f"{element_name}_{period}_{i}") % 9000
                                doc_links.append({
                                    'period': period,
                                    'document_id': doc_id,
                                    'title': f"Document about {element_name} in {period}",
                                    'url': f"#document-{doc_id}"
                                })
                    
                    if doc_links:
                        document_links[element_name] = doc_links
    
    return document_links


###########################################
# Comprehensive Analysis View
###########################################

def create_comprehensive_burst_analysis(
    burst_data: Dict[str, Dict[str, pd.DataFrame]],
    include_prediction: bool = True,
    include_network: bool = True,
    include_historical_events: bool = True,
    prediction_periods: int = 2,
    title: str = "Comprehensive Burst Analysis",
    document_ids: Optional[Dict[str, List[int]]] = None,
    historical_events_file: Optional[str] = None
) -> Dict[str, go.Figure]:
    """
    Create a comprehensive set of burst analysis visualizations that include all enhanced features.
    
    Args:
        burst_data: Dictionary mapping data types to dictionaries of element burst DataFrames
        include_prediction: Whether to include predictive visualization
        include_network: Whether to include co-occurrence network
        include_historical_events: Whether to include historical events
        prediction_periods: Number of periods to predict in predictive visualization
        title: Base title for the visualizations
        document_ids: Dictionary mapping elements to document IDs for linking
        historical_events_file: Path to historical events JSON file
        
    Returns:
        Dict mapping visualization names to Plotly Figure objects
    """
    result = {}
    
    # Prepare document links if needed
    document_links = None
    if document_ids:
        document_links = prepare_document_links(burst_data, document_ids)
    
    # Load historical events if enabled
    historical_events = None
    if include_historical_events:
        historical_events = load_historical_events(historical_events_file)
    
    # Create combined summary DataFrame for timeline visualization
    timeline_data = []
    for data_type, elements in burst_data.items():
        prefix = "T: " if data_type == 'taxonomy' else "K: " if data_type == 'keywords' else "E: "
        for element, df in elements.items():
            if not df.empty and 'period' in df.columns and 'burst_intensity' in df.columns:
                qualified_name = prefix + element
                for _, row in df.iterrows():
                    timeline_data.append({
                        'element': qualified_name,
                        'period': row['period'],
                        'burst_intensity': row['burst_intensity']
                    })
    
    # Create enhanced CiteSpace timeline
    if timeline_data:
        timeline_df = pd.DataFrame(timeline_data)
        result['timeline'] = create_enhanced_citespace_timeline(
            timeline_df,
            historical_events=historical_events,
            document_links=document_links,
            title=f"{title} - Enhanced Timeline",
            show_annotations=include_historical_events
        )
    
    # Create co-occurrence network if enabled
    if include_network:
        result['network'] = create_co_occurrence_network(
            burst_data,
            min_burst_intensity=20.0,
            min_periods=2,
            min_strength=0.3,
            title=f"{title} - Co-occurrence Network"
        )
    
    # Create predictive visualization if enabled
    if include_prediction:
        result['prediction'] = create_predictive_visualization(
            burst_data,
            prediction_periods=prediction_periods,
            confidence_level=0.9,
            min_periods_for_prediction=4,
            title=f"{title} - Trend Prediction",
            top_n=5
        )
    
    return result


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


###########################################
# Co-occurrence Network Visualization
###########################################

def create_co_occurrence_network(
    burst_data: Dict[str, Dict[str, pd.DataFrame]],
    min_burst_intensity: float = 20.0,
    min_periods: int = 2,
    min_strength: float = 0.3,
    title: str = "Burst Co-occurrence Network"
) -> go.Figure:
    """
    Create a network visualization of co-occurring burst elements.
    
    Args:
        burst_data: Dictionary mapping data types to dictionaries of element burst DataFrames
        min_burst_intensity: Minimum burst intensity to consider as a significant burst
        min_periods: Minimum number of periods where bursts must co-occur
        min_strength: Minimum co-occurrence strength to include in the network
        title: Chart title
        
    Returns:
        go.Figure: Plotly Figure object
    """
    # Extract all elements with bursts above threshold
    element_bursts = {}
    
    for data_type, elements in burst_data.items():
        for element_name, df in elements.items():
            if df.empty or 'burst_intensity' not in df.columns or 'period' not in df.columns:
                continue
                
            # Create a qualified name with data type prefix
            if data_type == 'taxonomy':
                prefix = "T"
            elif data_type == 'keywords':
                prefix = "K"
            else:
                prefix = "E"
                
            qualified_name = f"{prefix}:{element_name}"
            
            # Record periods with significant bursts
            burst_periods = set()
            for _, row in df.iterrows():
                if row['burst_intensity'] >= min_burst_intensity:
                    burst_periods.add(row['period'])
                    
            if burst_periods:
                element_bursts[qualified_name] = burst_periods
    
    # Find co-occurrences
    co_occurrences = {}
    
    for elem1, elem2 in combinations(element_bursts.keys(), 2):
        # Find common burst periods
        common_periods = element_bursts[elem1].intersection(element_bursts[elem2])
        
        if len(common_periods) >= min_periods:
            # Calculate Jaccard similarity as co-occurrence strength
            union_periods = element_bursts[elem1].union(element_bursts[elem2])
            strength = len(common_periods) / len(union_periods)
            
            if strength >= min_strength:
                # Store co-occurrence with its strength
                co_occurrences[(elem1, elem2)] = {
                    'strength': strength,
                    'common_periods': list(common_periods),
                    'num_common_periods': len(common_periods)
                }
    
    # If no co-occurrences found, return empty chart
    if not co_occurrences:
        return go.Figure().update_layout(title="No significant co-occurrences found")
    
    # Create network using NetworkX for layout
    G = nx.Graph()
    
    # Add nodes with data type information
    for node in set([elem for pair in co_occurrences for elem in pair]):
        data_type, name = node.split(':', 1)
        G.add_node(node, data_type=data_type, name=name)
    
    # Add edges with weights
    for (source, target), data in co_occurrences.items():
        G.add_edge(source, target, weight=data['strength'], 
                  periods=data['common_periods'],
                  count=data['num_common_periods'])
    
    # Use a spring layout for the network
    pos = nx.spring_layout(G, seed=42)
    
    # Prepare node colors by data type
    node_color_map = {'T': TAXONOMY_COLOR, 'K': KEYWORD_COLOR, 'E': ENTITY_COLOR}
    node_colors = [node_color_map[G.nodes[node]['data_type']] for node in G.nodes()]
    
    # Calculate node sizes based on degree
    node_sizes = [10 + 5 * G.degree(node) for node in G.nodes()]
    
    # Prepare the figure
    edge_traces = []
    
    # Create edge traces
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G.edges[edge]['weight']
        width = 1 + 5 * weight  # Scale line width by weight
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=width, color='rgba(150,150,150,0.5)'),
            hoverinfo='none',
            mode='lines'
        )
        edge_traces.append(edge_trace)
    
    # Create node trace
    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        mode='markers',
        marker=dict(
            showscale=False,
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white')
        ),
        text=[f"{G.nodes[node]['name']} ({G.nodes[node]['data_type']})" for node in G.nodes()],
        hovertemplate='<b>%{text}</b><br>Connections: %{marker.size}<extra></extra>'
    )
    
    # Combine all traces
    fig = go.Figure(data=edge_traces + [node_trace])
    
    # Create legend traces for data types
    legend_traces = [
        go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color=TAXONOMY_COLOR),
            name='Taxonomy Elements'
        ),
        go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color=KEYWORD_COLOR),
            name='Keywords'
        ),
        go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color=ENTITY_COLOR),
            name='Named Entities'
        )
    ]
    
    # Add legend traces
    for trace in legend_traces:
        fig.add_trace(trace)
    
    # Update layout
    fig.update_layout(
        title=title,
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        height=600,
        plot_bgcolor=TIMELINE_BG_COLOR,
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        annotations=[
            dict(
                text="<b>Node size</b>: Number of connections<br><b>Edge width</b>: Co-occurrence strength",
                showarrow=False,
                x=0.5,
                y=-0.2,
                xref="paper",
                yref="paper",
                font=dict(size=10)
            )
        ]
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