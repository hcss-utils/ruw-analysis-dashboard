#!/usr/bin/env python
# coding: utf-8

"""
Comparison visualization helpers for the dashboard.
"""

import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any, Union

import sys
import os

# Add the project root to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import THEME_COLORS
from utils.helpers import hex_to_rgba


def _detect_cross_language_data(df_a: pd.DataFrame, df_b: pd.DataFrame) -> bool:
    """
    Detect if the two datasets are from different languages based on category overlap.
    
    Args:
        df_a: DataFrame A
        df_b: DataFrame B
        
    Returns:
        bool: True if datasets appear to be from different languages
    """
    if df_a.empty or df_b.empty:
        return False
        
    # Get unique categories from each dataset
    cats_a = set(df_a['category'].unique())
    cats_b = set(df_b['category'].unique())
    
    # Calculate overlap
    overlap = cats_a.intersection(cats_b)
    total_cats = len(cats_a.union(cats_b))
    
    # If less than 10% overlap, consider it cross-language
    overlap_ratio = len(overlap) / total_cats if total_cats > 0 else 0
    
    logging.info(f"Category overlap: {len(overlap)}/{total_cats} ({overlap_ratio:.1%})")
    
    return overlap_ratio < 0.1


def create_comparison_plot(
    df_a: Union[List, pd.DataFrame], 
    df_b: Union[List, pd.DataFrame], 
    plot_type: str = 'sunburst', 
    slice_a_name: str = "Russian", 
    slice_b_name: str = "Western"
) -> Tuple[go.Figure, go.Figure]:
    """
    Create comparison plots based on specified type.
    
    Args:
        df_a: Data for Slice A
        df_b: Data for Slice B
        plot_type: Type of visualization to create
        slice_a_name: Custom name for Slice A
        slice_b_name: Custom name for Slice B
    
    Returns:
        tuple: (fig_a, fig_b) two plotly figures or (fig, fig) for single-figure types
    """
    logging.info(f"create_comparison_plot called with plot_type={plot_type}")
    
    # Convert to DataFrame if necessary
    if not isinstance(df_a, pd.DataFrame):
        df_a = pd.DataFrame(df_a)
        logging.info(f"Converted list to DataFrame A: shape={df_a.shape}")
    if not isinstance(df_b, pd.DataFrame):
        df_b = pd.DataFrame(df_b)
        logging.info(f"Converted list to DataFrame B: shape={df_b.shape}")
    
    logging.info(f"DataFrame A shape: {df_a.shape}, columns: {df_a.columns.tolist() if not df_a.empty else 'empty'}")
    logging.info(f"DataFrame B shape: {df_b.shape}, columns: {df_b.columns.tolist() if not df_b.empty else 'empty'}")
    
    # Empty dataframe check
    if df_a.empty or df_b.empty:
        logging.warning(f"Empty dataframe detected! df_a.empty={df_a.empty}, df_b.empty={df_b.empty}")
        empty_fig = go.Figure().update_layout(title="No data available for comparison")
        return empty_fig, empty_fig
    
    # Check if this appears to be cross-language data
    is_cross_language = _detect_cross_language_data(df_a, df_b)
    
    # Process data for categories (usually the first level in the taxonomy)
    cat_a = df_a.groupby('category')['count'].sum().reset_index()
    cat_b = df_b.groupby('category')['count'].sum().reset_index()
    
    logging.info(f"Grouped categories A: {len(cat_a)} categories")
    logging.info(f"Grouped categories B: {len(cat_b)} categories")
    if not cat_a.empty:
        logging.info(f"Categories A sample:\n{cat_a.head()}")
    if not cat_b.empty:
        logging.info(f"Categories B sample:\n{cat_b.head()}")
    
    # For visualizations that need category percentages within each slice
    total_a = cat_a['count'].sum()
    total_b = cat_b['count'].sum()
    
    logging.info(f"Total counts: A={total_a}, B={total_b}")
    
    cat_a['percentage'] = (cat_a['count'] / total_a * 100).round(1) if total_a > 0 else 0
    cat_b['percentage'] = (cat_b['count'] / total_b * 100).round(1) if total_b > 0 else 0
    
    # For cross-language comparisons, use appropriate visualization
    if is_cross_language and plot_type in ['diff_means', 'radar', 'parallel']:
        # For cross-language, these visualizations don't make sense with different categories
        # Use side-by-side bar charts instead
        return _create_cross_language_comparison(cat_a, cat_b, slice_a_name, slice_b_name)
    
    # Choose visualization type
    if plot_type == 'parallel':
        return _create_parallel_bars(cat_a, cat_b, slice_a_name, slice_b_name)
    elif plot_type == 'radar':
        return _create_radar_chart(cat_a, cat_b, slice_a_name, slice_b_name)
    elif plot_type == 'sankey':
        return _create_sankey_diagram(cat_a, cat_b, slice_a_name, slice_b_name)
    elif plot_type == 'heatmap':
        return _create_heatmap_comparison(cat_a, cat_b, slice_a_name, slice_b_name)
    elif plot_type == 'diff_means':
        return _create_diff_means_chart(cat_a, cat_b, slice_a_name, slice_b_name)
    else:
        # Default to sunburst
        # Importing here to avoid circular imports
        from visualizations.sunburst import create_sunburst_chart
        fig_a = create_sunburst_chart(df_a, title=f"{slice_a_name} Categories")
        fig_b = create_sunburst_chart(df_b, title=f"{slice_b_name} Categories")
        return fig_a, fig_b


def _create_parallel_bars(
    cat_a: pd.DataFrame, 
    cat_b: pd.DataFrame, 
    slice_a_name: str, 
    slice_b_name: str
) -> Tuple[go.Figure, go.Figure]:
    """
    Create parallel stacked bars comparison.
    
    Args:
        cat_a: Category data for slice A
        cat_b: Category data for slice B
        slice_a_name: Name for slice A
        slice_b_name: Name for slice B
        
    Returns:
        Tuple[go.Figure, go.Figure]: Same figure repeated (for API consistency)
    """
    # Get common categories to ensure consistent colors
    all_categories = sorted(list(set(cat_a['category']).union(set(cat_b['category']))))
    
    # Start with the stacked percentage data preparation
    subset_sums_a = cat_a.set_index('category')[['count']]
    subset_sums_b = cat_b.set_index('category')[['count']]
    
    # Calculate percentages for each slice
    subset_percentages_a = (subset_sums_a / subset_sums_a.sum()) * 100
    subset_percentages_b = (subset_sums_b / subset_sums_b.sum()) * 100
    
    # Create mapping for subset positions
    subset_map = {slice_a_name: 0, slice_b_name: 1}
    
    # Create figure
    fig = go.Figure()
    
    # Set up colors for consistency, using THEME_COLORS where possible
    colors = []
    for cat in all_categories:
        # Check if category name matches a language code in THEME_COLORS
        if cat in THEME_COLORS:
            colors.append(THEME_COLORS[cat])
        else:
            # Use from qualitative palette
            colors.append(px.colors.qualitative.Set3[len(colors) % len(px.colors.qualitative.Set3)])
    
    # Create a color map for all categories
    color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(all_categories)}
    
    # Bar parameters
    bar_width = 0.8
    
    # Create data frames with all categories for consistent ordering
    all_cat_a = pd.DataFrame({
        'category': all_categories,
        'percentage': [subset_percentages_a.loc[cat, 'count'] if cat in subset_percentages_a.index else 0 
                       for cat in all_categories]
    }).sort_values('percentage', ascending=False)
    
    all_cat_b = pd.DataFrame({
        'category': all_categories,
        'percentage': [subset_percentages_b.loc[cat, 'count'] if cat in subset_percentages_b.index else 0 
                       for cat in all_categories]
    })
    
    # Reorder cat_b to match cat_a's order
    all_cat_b = all_cat_b.set_index('category').loc[all_cat_a['category']].reset_index()
    
    # Add bars for Slice A - stacked approach
    cumulative_a = 0
    a_levels = []
    for _, row in all_cat_a.iterrows():
        cat = row['category']
        perc = row['percentage'] / 100  # Convert to 0-1 scale
        fig.add_trace(go.Bar(
            x=[subset_map[slice_a_name]],
            y=[perc],
            name=cat,
            marker_color=color_map[cat],
            offset=0,
            width=bar_width,
            base=cumulative_a,
            hoverinfo='text',
            hovertext=f"{cat}: {perc*100:.1f}%",
            showlegend=True if cumulative_a == 0 else False  # Only show legend for first segment
        ))
        a_levels.append((cat, cumulative_a, cumulative_a + perc))
        cumulative_a += perc
    
    # Add bars for Slice B - stacked approach with same order as A
    cumulative_b = 0
    b_levels = []
    for _, row in all_cat_b.iterrows():
        cat = row['category']
        perc = row['percentage'] / 100  # Convert to 0-1 scale
        fig.add_trace(go.Bar(
            x=[subset_map[slice_b_name]],
            y=[perc],
            name=cat,
            marker_color=color_map[cat],
            offset=0,
            width=bar_width,
            base=cumulative_b,
            hoverinfo='text',
            hovertext=f"{cat}: {perc*100:.1f}%",
            showlegend=False  # Avoid duplicate legend entries
        ))
        b_levels.append((cat, cumulative_b, cumulative_b + perc))
        cumulative_b += perc
    
    # Add connecting dotted lines between the same categories
    x_a_edge = subset_map[slice_a_name] + bar_width / 2
    x_b_edge = subset_map[slice_b_name] - bar_width / 2
    
    # Bottom line at y=0
    fig.add_trace(go.Scatter(
        x=[x_a_edge, x_b_edge],
        y=[0, 0],
        mode='lines',
        line=dict(dash='dot', color='grey', width=1),
        showlegend=False
    ))
    
    # Connect matching levels with dotted lines
    cat_dict_a = {cat: (start, end) for cat, start, end in a_levels}
    cat_dict_b = {cat: (start, end) for cat, start, end in b_levels}
    
    for cat in all_categories:
        if cat in cat_dict_a and cat in cat_dict_b:
            # Connect the top of each matching category bar
            _, a_end = cat_dict_a[cat]
            _, b_end = cat_dict_b[cat]
            
            fig.add_trace(go.Scatter(
                x=[x_a_edge, x_b_edge],
                y=[a_end, b_end],
                mode='lines',
                line=dict(dash='dot', color='grey', width=1),
                showlegend=False
            ))
    
    # Update layout with improved aesthetics
    fig.update_layout(
        title=f"Percentage Distribution: {slice_a_name} vs {slice_b_name}",
        xaxis=dict(
            tickmode='array',
            tickvals=list(subset_map.values()),
            ticktext=list(subset_map.keys()),
            title=""
        ),
        yaxis=dict(
            tickformat='.0%',  # Format as percentage
            range=[0, 1.05],   # Ensure full range with a little padding
            title="Percentage within Slice"
        ),
        barmode='stack',
        bargap=0.15,
        legend_title="Categories",
        height=700,
        width=1200,  # Increased width
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Add annotation explaining the visualization
    fig.add_annotation(
        text="This chart shows the percentage distribution within each slice.<br>Connecting lines help track the same category across both groups.",
        xref="paper", yref="paper",
        x=0.5, y=1.05,
        showarrow=False,
        font=dict(size=12),
        align="center",
        bordercolor="gray",
        borderwidth=1,
        borderpad=4,
        bgcolor="white"
    )
    
    return fig, fig  # Return the same figure twice for consistency


def _create_radar_chart(
    cat_a: pd.DataFrame, 
    cat_b: pd.DataFrame, 
    slice_a_name: str, 
    slice_b_name: str
) -> Tuple[go.Figure, go.Figure]:
    """
    Create radar chart comparison.
    
    Args:
        cat_a: Category data for slice A
        cat_b: Category data for slice B
        slice_a_name: Name for slice A
        slice_b_name: Name for slice B
        
    Returns:
        Tuple[go.Figure, go.Figure]: Same figure repeated (for API consistency)
    """
    # Get common categories to ensure both sides use the same scale
    common_categories = sorted(list(set(cat_a['category']).union(set(cat_b['category']))))
    
    # Create combined radar chart
    fig = go.Figure()
    
    # Prepare data for both slices
    # Map percentage values to all categories (fill missing with 0)
    full_data_a = {}
    full_data_b = {}
    
    for cat in common_categories:
        row_a = cat_a[cat_a['category'] == cat]
        row_b = cat_b[cat_b['category'] == cat]
        
        full_data_a[cat] = row_a['percentage'].values[0] if not row_a.empty else 0
        full_data_b[cat] = row_b['percentage'].values[0] if not row_b.empty else 0
    
    # Add trace for slice A
    fig.add_trace(go.Scatterpolar(
        r=list(full_data_a.values()),
        theta=list(full_data_a.keys()),
        fill='toself',
        name=slice_a_name,
        line_color=THEME_COLORS['russian'],
        hovertemplate='%{theta}<br>%{r:.1f}%<extra></extra>'
    ))
    
    # Add trace for slice B
    fig.add_trace(go.Scatterpolar(
        r=list(full_data_b.values()),
        theta=list(full_data_b.keys()),
        fill='toself',
        name=slice_b_name,
        line_color=THEME_COLORS['western'],
        hovertemplate='%{theta}<br>%{r:.1f}%<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(full_data_a.values()), max(full_data_b.values())) * 1.1],
                ticksuffix='%'
            )
        ),
        showlegend=True,
        title=f"Category Distribution Comparison: {slice_a_name} vs {slice_b_name}",
        height=600,
        width=1200  # Increased width
    )
    
    # Add annotation explaining the chart
    fig.add_annotation(
        text="This radar chart shows the percentage distribution of categories within each slice.<br>Larger values mean the category occupies a greater portion of that slice.",
        xref="paper", yref="paper",
        x=0.5, y=-0.1,
        showarrow=False,
        font=dict(size=12),
        align="center"
    )
    
    return fig, fig  # Return the same figure twice for consistency


def _create_sankey_diagram(
    cat_a: pd.DataFrame, 
    cat_b: pd.DataFrame, 
    slice_a_name: str, 
    slice_b_name: str
) -> Tuple[go.Figure, go.Figure]:
    """
    Create Sankey diagram comparison.
    
    Args:
        cat_a: Category data for slice A
        cat_b: Category data for slice B
        slice_a_name: Name for slice A
        slice_b_name: Name for slice B
        
    Returns:
        Tuple[go.Figure, go.Figure]: Same figure repeated (for API consistency)
    """
    common_categories = list(set(cat_a['category']).union(set(cat_b['category'])))
    
    # Set up colors using THEME_COLORS where possible
    colors = []
    for cat in common_categories:
        if cat in THEME_COLORS:
            colors.append(THEME_COLORS[cat])
        else:
            colors.append(px.colors.qualitative.Set3[len(colors) % len(px.colors.qualitative.Set3)])
    
    # Prepare sources, targets, and values
    sources = []  # 0 for slice_a, 1 for slice_b
    targets = []  # indices 2+ for categories
    values = []   # count values
    
    # Node labels list
    node_labels = [slice_a_name, slice_b_name] + common_categories
    
    # Map categories to target indices (starting from 2)
    cat_to_idx = {cat: i+2 for i, cat in enumerate(common_categories)}
    
    # Add links from slice_a to categories
    for _, row in cat_a.iterrows():
        sources.append(0)  # Slice A
        targets.append(cat_to_idx[row['category']])
        values.append(row['count'])
        
    # Add links from slice_b to categories
    for _, row in cat_b.iterrows():
        sources.append(1)  # Slice B
        targets.append(cat_to_idx[row['category']])
        values.append(row['count'])
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=[THEME_COLORS['russian'], THEME_COLORS['western']] + colors,
            hovertemplate='%{label}<br>Total: %{value:,}<extra></extra>'
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            hovertemplate='%{source.label} → %{target.label}<br>Value: %{value:,}<extra></extra>'
        )
    )])
    
    fig.update_layout(
        title=f"Category Distribution Flow - {slice_a_name} vs {slice_b_name}",
        font_size=12,
        height=800,
        width=1500  # Increased width for full screen usage
    )
    
    # Add annotation explaining the sankey diagram
    fig.add_annotation(
        text="This Sankey diagram shows the flow of data from each slice to categories.<br>Width of links represents the number of items in each category.",
        xref="paper", yref="paper",
        x=0.5, y=-0.05,
        showarrow=False,
        font=dict(size=12),
        align="center"
    )
    
    return fig, fig  # Return the same figure twice


def _create_heatmap_comparison(
    cat_a: pd.DataFrame, 
    cat_b: pd.DataFrame, 
    slice_a_name: str, 
    slice_b_name: str
) -> Tuple[go.Figure, go.Figure]:
    """
    Create heatmap comparison.
    
    Args:
        cat_a: Category data for slice A
        cat_b: Category data for slice B
        slice_a_name: Name for slice A
        slice_b_name: Name for slice B
        
    Returns:
        Tuple[go.Figure, go.Figure]: Two complementary figures
    """
    # 1. Merge datasets for direct comparison
    # Create a full dataframe with all categories and their percentages
    merged = pd.merge(
        cat_a[['category', 'percentage', 'count']],
        cat_b[['category', 'percentage', 'count']],
        on='category',
        how='outer',
        suffixes=(f'_{slice_a_name.lower()}', f'_{slice_b_name.lower()}')
    ).fillna(0)
    
    # Calculate the difference in percentage points
    merged[f'diff_pp'] = merged[f'percentage_{slice_b_name.lower()}'] - merged[f'percentage_{slice_a_name.lower()}']
    
    # Calculate fold change where applicable
    merged[f'fold_change'] = merged[f'count_{slice_b_name.lower()}'] / merged[f'count_{slice_a_name.lower()}'].replace(0, float('nan'))
    
    # Sort by absolute difference for the heatmap
    merged = merged.sort_values('diff_pp', key=abs, ascending=False)
    
    # 2. Create main heatmap showing the percentage point differences
    tooltip_text = merged.apply(
        lambda x: f"{slice_a_name}: {x[f'percentage_{slice_a_name.lower()}']:.1f}% ({x[f'count_{slice_a_name.lower()}']:,} items)<br>"
                  f"{slice_b_name}: {x[f'percentage_{slice_b_name.lower()}']:.1f}% ({x[f'count_{slice_b_name.lower()}']:,} items)<br>"
                  f"Diff: {x['diff_pp']:.1f} percentage points", 
        axis=1
    ).tolist()
    
    fig = go.Figure(data=go.Heatmap(
        z=merged['diff_pp'].values.reshape(-1, 1),
        y=merged['category'],
        x=['Percentage Point Difference'],
        colorscale='RdBu_r',  # Red for negative (more in A), Blue for positive (more in B)
        zmid=0,  # Center color scale at zero
        text=tooltip_text,
        hoverinfo='text+y',
        colorbar=dict(
            title=f"{slice_b_name} - {slice_a_name} (pp)",
            ticksuffix=" pp"
        )
    ))
    
    fig.update_layout(
        title=f"Category Distribution Difference ({slice_b_name} - {slice_a_name})",
        height=800,
        width=600
    )
    
    # 3. Create side-by-side bar chart for second figure
    fig2 = make_subplots(
        rows=1, 
        cols=2, 
        subplot_titles=(f"{slice_a_name} (%)", f"{slice_b_name} (%)"),
        shared_yaxes=True
    )
    
    fig2.add_trace(
        go.Bar(
            y=merged['category'],
            x=merged[f'percentage_{slice_a_name.lower()}'],
            orientation='h',
            name=slice_a_name,
            marker_color=THEME_COLORS['russian'],
            text=merged[f'percentage_{slice_a_name.lower()}'].apply(lambda x: f"{x:.1f}%"),
            textposition='auto',
            hovertemplate='%{y}: %{x:.1f}% (%{text})<br>Count: %{customdata:,}<extra></extra>',
            customdata=merged[f'count_{slice_a_name.lower()}']
        ),
        row=1, col=1
    )
    
    fig2.add_trace(
        go.Bar(
            y=merged['category'],
            x=merged[f'percentage_{slice_b_name.lower()}'],
            orientation='h',
            name=slice_b_name,
            marker_color=THEME_COLORS['western'],
            text=merged[f'percentage_{slice_b_name.lower()}'].apply(lambda x: f"{x:.1f}%"),
            textposition='auto',
            hovertemplate='%{y}: %{x:.1f}% (%{text})<br>Count: %{customdata:,}<extra></extra>',
            customdata=merged[f'count_{slice_b_name.lower()}']
        ),
        row=1, col=2
    )
    
    fig2.update_layout(
        title="Category Distribution by Percentage (%)",
        height=800,
        width=1200,  # Increased width
        bargap=0.2,
        xaxis_title="Percentage (%)",
        xaxis2_title="Percentage (%)"
    )
    
    # Add annotation explaining the heatmap
    fig2.add_annotation(
        text="These charts show the percentage distribution of categories within each slice.<br>"
             "The heatmap shows the difference in percentage points between slices.",
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        showarrow=False,
        font=dict(size=12),
        align="center"
    )
    
    return fig, fig2


def _create_diff_means_chart(
    cat_a: pd.DataFrame, 
    cat_b: pd.DataFrame, 
    slice_a_name: str, 
    slice_b_name: str
) -> Tuple[go.Figure, go.Figure]:
    """
    Create diverging bar chart comparison.
    
    Args:
        cat_a: Category data for slice A
        cat_b: Category data for slice B
        slice_a_name: Name for slice A
        slice_b_name: Name for slice B
        
    Returns:
        Tuple[go.Figure, go.Figure]: Two complementary figures
    """
    logging.info(f"_create_diff_means_chart: cat_a shape={cat_a.shape}, cat_b shape={cat_b.shape}")
    
    # Prepare data - merge and calculate differences
    merged = pd.merge(
        cat_a[['category', 'percentage', 'count']],
        cat_b[['category', 'percentage', 'count']],
        on='category',
        how='outer',
        suffixes=(f'_{slice_a_name.lower()}', f'_{slice_b_name.lower()}')
    ).fillna(0)
    
    logging.info(f"Merged dataframe shape: {merged.shape}")
    if not merged.empty:
        logging.info(f"Merged columns: {merged.columns.tolist()}")
        logging.info(f"Merged sample:\n{merged.head()}")
    
    # Calculate difference (B - A)
    merged['diff'] = merged[f'percentage_{slice_b_name.lower()}'] - merged[f'percentage_{slice_a_name.lower()}']
    
    # Sort by absolute difference for better visualization
    merged = merged.sort_values('diff', key=abs, ascending=False)
    
    # Create the diverging bar chart
    fig = go.Figure()
    
    # Add bars - use slice colors from THEME_COLORS
    fig.add_trace(go.Bar(
        y=merged['category'],
        x=merged['diff'],
        orientation='h',
        marker_color=[THEME_COLORS['russian'] if x < 0 else THEME_COLORS['western'] for x in merged['diff']],
        text=merged.apply(
            lambda x: f"{x['diff']:.1f}pp<br>({slice_a_name}: {x[f'percentage_{slice_a_name.lower()}']:.1f}%)<br>({slice_b_name}: {x[f'percentage_{slice_b_name.lower()}']:.1f}%)",
            axis=1
        ),
        textposition='outside',
        hoverinfo='text',
        # Add hover template with count values using thousands separators
        hovertemplate=merged.apply(
            lambda x: f"{x['category']}<br>Diff: {x['diff']:.1f} pp<br>{slice_a_name}: {x[f'percentage_{slice_a_name.lower()}']:.1f}% ({x[f'count_{slice_a_name.lower()}']:,} items)<br>{slice_b_name}: {x[f'percentage_{slice_b_name.lower()}']:.1f}% ({x[f'count_{slice_b_name.lower()}']:,} items)<extra></extra>",
            axis=1
        ).tolist()
    ))
    
    # Define annotations for the directional arrow
    # Vertical position for annotation elements
    y_line_level = -0.18  # Y position of horizontal line
    y_text_level = y_line_level - 0.06  # Y position for text labels
    
    # Horizontal positions
    x_center = 0.5    # Center point
    x_left_end = 0.15  # Left boundary
    x_right_end = 0.85  # Right boundary
    
    # Arrow shaft length in pixels
    arrow_shaft_len_px = (x_center - x_left_end) * 400
    
    # Text positions
    x_left_text_pos = (x_left_end + x_center) / 2
    x_right_text_pos = (x_right_end + x_center) / 2
    
    # Create annotations
    annotations = [
        # Left Arrowhead (points left)
        dict(
            x=x_left_end, y=y_line_level,
            xref='paper', yref='paper',
            showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=1, arrowcolor='black',
            ax=arrow_shaft_len_px,  # Tail offset to the right
            ay=0,
            text=''
        ),
        # Right Arrowhead (points right)
        dict(
            x=x_right_end, y=y_line_level,
            xref='paper', yref='paper',
            showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=1, arrowcolor='black',
            ax=-arrow_shaft_len_px,  # Tail offset to the left
            ay=0,
            text=''
        ),
        # Text: More A
        dict(
            x=x_left_text_pos, y=y_text_level,
            xref='paper', yref='paper',
            text=f'More {slice_a_name}',
            showarrow=False,
            font=dict(size=11, color='black'),
            align='center'
        ),
        # Text: More B
        dict(
            x=x_right_text_pos, y=y_text_level,
            xref='paper', yref='paper',
            text=f'More {slice_b_name}',
            showarrow=False,
            font=dict(size=11, color='black'),
            align='center'
        )
    ]
    
    # Define shapes (center tick and horizontal line)
    tick_height = 0.015
    shapes = [
        # Center tick
        dict(
            type='line', layer='below',
            xref='paper', yref='paper',
            x0=x_center, y0=y_line_level - tick_height,
            x1=x_center, y1=y_line_level + tick_height,
            line=dict(color='black', width=1.5)
        ),
        # Main horizontal line
        dict(
            type='line', layer='below',
            xref='paper', yref='paper',
            x0=x_left_end, y0=y_line_level,
            x1=x_right_end, y1=y_line_level,
            line=dict(color='black', width=1)
        )
    ]
    
    # Update layout
    fig.update_layout(
        title=f"Difference in Percentage Points ({slice_b_name} - {slice_a_name})",
        xaxis_title="Percentage Points Difference",
        annotations=annotations,
        shapes=shapes,
        margin=dict(b=120),
        height=800,
        width=1500  # Increased width for full screen
    )
    
    # Add zero reference line
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")
    
    # Add explanatory annotation
    fig.add_annotation(
        text="This chart shows the difference in percentage points between slices.<br>"
             f"Bars extending left indicate categories with higher percentage in {slice_a_name}.<br>"
             f"Bars extending right indicate categories with higher percentage in {slice_b_name}.",
        xref="paper", yref="paper",
        x=0.5, y=-0.25,
        showarrow=False,
        font=dict(size=12),
        align="center",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1,
        borderpad=4
    )
    
    # Create a complementary figure showing the actual percentages
    fig2 = go.Figure()
    
    # Create a grouped bar chart with percentages
    fig2.add_trace(go.Bar(
        y=merged['category'],
        x=merged[f'percentage_{slice_a_name.lower()}'],
        name=slice_a_name,
        marker_color=THEME_COLORS['russian'],
        orientation='h',
        text=merged[f'percentage_{slice_a_name.lower()}'].apply(lambda x: f"{x:.1f}%"),
        textposition='inside',
        hovertemplate='%{y}<br>%{x:.1f}%<br>Count: %{customdata:,}<extra></extra>',
        customdata=merged[f'count_{slice_a_name.lower()}']
    ))
    
    fig2.add_trace(go.Bar(
        y=merged['category'],
        x=merged[f'percentage_{slice_b_name.lower()}'],
        name=slice_b_name,
        marker_color=THEME_COLORS['western'],
        orientation='h',
        text=merged[f'percentage_{slice_b_name.lower()}'].apply(lambda x: f"{x:.1f}%"),
        textposition='inside',
        hovertemplate='%{y}<br>%{x:.1f}%<br>Count: %{customdata:,}<extra></extra>',
        customdata=merged[f'count_{slice_b_name.lower()}']
    ))
    
    fig2.update_layout(
        title=f"Category Percentages by Slice",
        xaxis_title="Percentage (%)",
        barmode='group',
        height=800,
        width=1500,  # Increased width
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig, fig2


def _create_cross_language_comparison(
    cat_a: pd.DataFrame, 
    cat_b: pd.DataFrame, 
    slice_a_name: str, 
    slice_b_name: str
) -> Tuple[go.Figure, go.Figure]:
    """
    Create comparison visualization for cross-language data where categories don't overlap.
    
    Args:
        cat_a: Category data for slice A
        cat_b: Category data for slice B
        slice_a_name: Name for slice A
        slice_b_name: Name for slice B
        
    Returns:
        Tuple[go.Figure, go.Figure]: Side-by-side bar charts
    """
    # Sort by count descending and take top 15
    cat_a_top = cat_a.nlargest(15, 'count')
    cat_b_top = cat_b.nlargest(15, 'count')
    
    # Create side-by-side bar charts
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Top {slice_a_name} Items", f"Top {slice_b_name} Items"),
        shared_yaxes=False,
        horizontal_spacing=0.15
    )
    
    # Add bars for slice A
    fig.add_trace(
        go.Bar(
            y=cat_a_top['category'],
            x=cat_a_top['count'],
            orientation='h',
            name=slice_a_name,
            marker_color=THEME_COLORS.get('russian', '#1f77b4'),
            text=cat_a_top.apply(lambda x: f"{x['count']:,} ({x['percentage']:.1f}%)", axis=1),
            textposition='auto',
            hovertemplate='%{y}<br>Count: %{x:,}<br>Percentage: %{text}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add bars for slice B
    fig.add_trace(
        go.Bar(
            y=cat_b_top['category'],
            x=cat_b_top['count'],
            orientation='h',
            name=slice_b_name,
            marker_color=THEME_COLORS.get('western', '#ff7f0e'),
            text=cat_b_top.apply(lambda x: f"{x['count']:,} ({x['percentage']:.1f}%)", axis=1),
            textposition='auto',
            hovertemplate='%{y}<br>Count: %{x:,}<br>Percentage: %{text}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f"Top Items Comparison: {slice_a_name} vs {slice_b_name}",
        height=600,
        width=1400,
        showlegend=False,
        margin=dict(l=200, r=50, t=80, b=50)
    )
    
    fig.update_xaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(tickmode='linear', row=1, col=1)
    fig.update_yaxes(tickmode='linear', row=1, col=2)
    
    # Add annotation explaining the visualization
    fig.add_annotation(
        text="Note: Items from different languages are shown separately as they have no direct overlap.<br>"
             "The percentages shown are relative to each dataset's total.",
        xref="paper", yref="paper",
        x=0.5, y=-0.1,
        showarrow=False,
        font=dict(size=12),
        align="center",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1,
        borderpad=4
    )
    
    # Create a summary comparison figure
    summary_data = [
        {'Metric': 'Total Items', slice_a_name: len(cat_a), slice_b_name: len(cat_b)},
        {'Metric': 'Total Count', slice_a_name: cat_a['count'].sum(), slice_b_name: cat_b['count'].sum()},
        {'Metric': 'Top Item Count', slice_a_name: cat_a_top.iloc[0]['count'] if not cat_a_top.empty else 0, 
         slice_b_name: cat_b_top.iloc[0]['count'] if not cat_b_top.empty else 0},
        {'Metric': 'Top Item %', slice_a_name: cat_a_top.iloc[0]['percentage'] if not cat_a_top.empty else 0, 
         slice_b_name: cat_b_top.iloc[0]['percentage'] if not cat_b_top.empty else 0}
    ]
    
    summary_df = pd.DataFrame(summary_data)
    
    fig2 = go.Figure()
    
    # Add bars for summary metrics
    for col in [slice_a_name, slice_b_name]:
        fig2.add_trace(go.Bar(
            x=summary_df['Metric'],
            y=summary_df[col],
            name=col,
            text=summary_df[col].apply(lambda x: f"{x:,.0f}" if x > 100 else f"{x:.1f}"),
            textposition='auto'
        ))
    
    fig2.update_layout(
        title="Summary Statistics Comparison",
        xaxis_title="Metric",
        yaxis_title="Value",
        height=400,
        width=800,
        barmode='group'
    )
    
    return fig, fig2