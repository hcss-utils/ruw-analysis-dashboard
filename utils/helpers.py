#!/usr/bin/env python
# coding: utf-8

"""
Utility functions for the dashboard.
"""

import re
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union

import dash_bootstrap_components as dbc
from dash import html


def get_unique_filename(base_name: str) -> str:
    """
    Generate a unique filename with timestamp.
    
    Args:
        base_name: Base filename
        
    Returns:
        str: Unique filename with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(base_name)
    return f"{name}_{timestamp}{ext}"


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """
    Convert hex color to rgba with alpha.
    
    Args:
        hex_color: Hex color code
        alpha: Alpha value (0-1)
        
    Returns:
        str: RGBA color string
    """
    hex_color = hex_color.lstrip('#')
    if hex_color.startswith('rgba'):
        return hex_color

    if not re.match(r"[0-9a-fA-F]{6}", hex_color):
        hex_color = '000000'

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return f'rgba({r},{g},{b},{alpha})'


def format_chunk_row(row: Dict[str, Any]) -> html.Div:
    """
    Format a chunk row for display.
    
    Args:
        row: Row data as a dictionary
        
    Returns:
        html.Div: Formatted chunk row
    """
    try:
        # Add thousands separators to count values in the metadata
        document_id = row.get('document_id', 'N/A')
        if isinstance(document_id, int):
            document_id = f"{document_id:,}"
            
        metadata = html.P([
            html.B("Document ID: "), f"{document_id} | ",
            html.B("Database: "), f"{row.get('database', 'N/A')} | ",
            html.B("Heading: "), f"{row.get('heading_title', 'N/A')} | ",
            html.B("Date: "), f"{row.get('date', 'N/A')} | ",
            html.B("Author: "), f"{row.get('author', 'N/A')}"
        ], style={'margin-bottom': '5px'})

        chunk_text = html.Div([
            html.P(row.get('chunk_text', 'No text available'))
        ], style={'padding': '10px', 'border': '1px solid lightgray',
                  'margin-bottom': '10px', 'width': '100%'})

        # Reasoning formatting
        reasoning_text = row.get('reasoning', '')
        formatted_reasoning = []
        if reasoning_text:
            reasoning_text = re.sub(r'# Reasoning', '', reasoning_text).strip()
            sections = re.split(r'(\d+\..*?:)', reasoning_text)
            sections = [s.strip() for s in sections if s.strip()]

            for section in sections:
                match = re.match(r'(\d+\.)(.*?):(.*)', section, re.DOTALL)
                if match:
                    number, bold_content, rest_content = match.groups()
                    formatted_reasoning.append(html.P(f"{number} {bold_content}:"))
                    rest_content = rest_content.strip()
                    if rest_content:
                        bullet_items = [
                            html.P("• " + item.strip())
                            for item in re.split(r'(?<!\d\.)\\s*-\\s*', rest_content)
                            if item.strip()
                        ]
                        formatted_reasoning.extend(bullet_items)
                else:
                    formatted_reasoning.append(html.P(section))
                    
        chunk_reasoning = html.Div(
            formatted_reasoning or [html.P("No reasoning available")],
            style={'padding': '10px', 'border': '1px solid lightgray',
                   'margin-bottom': '10px', 'width': '100%'}
        )

        # Two column layout for text and reasoning
        row_layout = html.Div([
            html.Div([metadata], style={'border-bottom': '1px solid #ddd', 'padding-bottom': '8px', 'margin-bottom': '10px'}),
            dbc.Row([
                dbc.Col([
                    html.H6("Text", style={'border-bottom': '1px solid #eee', 'padding-bottom': '5px'}),
                    chunk_text
                ], width=6),
                dbc.Col([
                    html.H6("Reasoning", style={'border-bottom': '1px solid #eee', 'padding-bottom': '5px'}),
                    chunk_reasoning
                ], width=6)
            ])
        ], style={'margin-bottom': '20px', 'border-bottom': '2px solid #ccc', 'padding-bottom': '15px'})

        return row_layout
    except Exception as e:
        logging.error(f"Error formatting chunk row: {e}")
        return html.Div("Error displaying this chunk")


def create_comparison_text(
    df_a: Union[List, 'pd.DataFrame'],
    df_b: Union[List, 'pd.DataFrame'],
    viz_type: str,
    slice_a_name: str = "Russian",
    slice_b_name: str = "Western"
) -> html.Div:
    """
    Create textual analysis of comparison data.
    
    Args:
        df_a: Data for slice A
        df_b: Data for slice B
        viz_type: Visualization type
        slice_a_name: Name for slice A
        slice_b_name: Name for slice B
        
    Returns:
        html.Div: Formatted comparison text
    """
    import pandas as pd
    
    # Convert to DataFrame if necessary
    if not isinstance(df_a, pd.DataFrame):
        df_a = pd.DataFrame(df_a)
    if not isinstance(df_b, pd.DataFrame):
        df_b = pd.DataFrame(df_b)
    
    # Process data to get top categories
    cat_a = df_a.groupby('category')['count'].sum().reset_index()
    cat_b = df_b.groupby('category')['count'].sum().reset_index()
    
    total_a = cat_a['count'].sum()
    total_b = cat_b['count'].sum()
    
    if total_a == 0 or total_b == 0:
        return html.Div([
            html.H5("Insufficient Data for Comparison"),
            html.P("One or both slices have no data. Please adjust your filters and try again.")
        ])
    
    # Calculate percentages
    cat_a['percentage'] = (cat_a['count'] / total_a * 100).round(1)
    cat_b['percentage'] = (cat_b['count'] / total_b * 100).round(1)
    
    # Merge for comparison
    merged = pd.merge(
        cat_a[['category', 'percentage']],
        cat_b[['category', 'percentage']],
        on='category',
        how='outer',
        suffixes=(f'_{slice_a_name.lower()}', f'_{slice_b_name.lower()}')
    ).fillna(0)
    
    # Sort by absolute difference
    merged['diff'] = merged[f'percentage_{slice_b_name.lower()}'] - merged[f'percentage_{slice_a_name.lower()}']
    merged = merged.sort_values('diff', key=abs, ascending=False)
    
    # Get top 5 differences
    top_5 = merged.head(5)
    
    # Create analysis text WITH THOUSANDS SEPARATORS
    text_parts = [
        html.H5("Key Differences in Category Distribution"),
        html.P(f"{slice_a_name} data: {total_a:,} items, {slice_b_name} data: {total_b:,} items"),
        html.P(f"Top 5 differences in percentage points ({slice_b_name} - {slice_a_name}):"),
        html.Ul([
            html.Li([
                html.B(f"{row['category']}: "),
                f"{row['diff']:.1f} pp difference ",
                html.Span(f"({slice_b_name}: {row[f'percentage_{slice_b_name.lower()}']:.1f}%, {slice_a_name}: {row[f'percentage_{slice_a_name.lower()}']:.1f}%)")
            ]) 
            for _, row in top_5.iterrows()
        ])
    ]
    
    # Add visualization-specific insights
    if viz_type == 'radar':
        text_parts.append(html.P(f"The radar chart shows each category's relative prominence within its respective slice. Larger values indicate categories that make up a greater percentage of that slice's content."))
    elif viz_type == 'parallel':
        text_parts.append(html.P(f"The parallel view shows how categories are distributed within each slice, with connecting lines to help track differences."))
    elif viz_type == 'diff_means':
        text_parts.append(html.P(f"The diverging bars show which categories have higher representation in {slice_a_name} content (left) versus {slice_b_name} content (right)."))
    
    return html.Div(text_parts)