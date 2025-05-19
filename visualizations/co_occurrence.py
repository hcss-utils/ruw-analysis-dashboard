#!/usr/bin/env python
# coding: utf-8

"""
Burst co-occurrence network visualization functions.
These functions create network visualizations showing relationships between elements 
that burst at similar times, integrating concordance data to enhance relationship understanding.
"""

import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any, Union, Tuple
import networkx as nx
import json
from datetime import datetime
from sqlalchemy import text

# Add the project root to the path to import modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.burst_detection import detect_burst_co_occurrences, generate_co_occurrence_network
from database.connection import get_engine
from utils.cache import cached

# Theme colors for consistency
THEME_BLUE = "#13376f"  # Main dashboard theme color
TAXONOMY_COLOR = '#4caf50'  # Green for taxonomy
KEYWORD_COLOR = '#2196f3'   # Blue for keywords
ENTITY_COLOR = '#ff9800'    # Orange for entities

# Additional colors for network visualizations
NODE_COLORS = {
    'taxonomy': TAXONOMY_COLOR,
    'keywords': KEYWORD_COLOR, 
    'named_entities': ENTITY_COLOR,
    'T': TAXONOMY_COLOR,
    'K': KEYWORD_COLOR,
    'E': ENTITY_COLOR
}

@cached(timeout=600)
def fetch_concordance_data(
    element1: str,
    element2: str,
    max_results: int = 100,
    language: Optional[str] = None,
    database: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None
) -> pd.DataFrame:
    """
    Fetch concordance data for a pair of elements that burst together.
    This retrieves documents where both elements co-occur.
    
    Args:
        element1: First element name (with data type prefix, e.g., "T:Military Action")
        element2: Second element name (with data type prefix, e.g., "K:offensive")
        max_results: Maximum number of results to return
        language: Optional language filter
        database: Optional database filter
        date_range: Optional date range filter as (start_date, end_date)
        
    Returns:
        pd.DataFrame: DataFrame with concordance data
    """
    # Extract data types and element names
    data_type1, name1 = element1.split(':', 1)
    data_type2, name2 = element2.split(':', 1)
    
    # Map data type codes to actual data types
    data_type_map = {'T': 'taxonomy', 'K': 'keywords', 'E': 'named_entities'}
    data_type1 = data_type_map.get(data_type1, data_type1)
    data_type2 = data_type_map.get(data_type2, data_type2)
    
    try:
        # Build query based on data types
        params = {'name1': name1, 'name2': name2, 'max_results': max_results}
        
        # Start building the query
        if data_type1 == 'taxonomy' and data_type2 == 'taxonomy':
            # Both are taxonomy elements
            query = """
            SELECT 
                ud.document_id,
                ud.title,
                ud.author,
                ud.date,
                ud.database,
                ud.language,
                ds.heading_title,
                dsc.content AS chunk_text
            FROM taxonomy t1
            JOIN taxonomy t2 ON t1.chunk_id = t2.chunk_id
            JOIN document_section_chunk dsc ON t1.chunk_id = dsc.id
            JOIN document_section ds ON dsc.document_section_id = ds.id
            JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
            WHERE 
                (t1.category = :name1 OR t1.subcategory = :name1 OR t1.sub_subcategory = :name1)
                AND (t2.category = :name2 OR t2.subcategory = :name2 OR t2.sub_subcategory = :name2)
                AND t1.id != t2.id
            """
        elif data_type1 == 'taxonomy' or data_type2 == 'taxonomy':
            # One is taxonomy, the other is keyword or entity
            if data_type1 == 'taxonomy':
                tax_name = name1
                other_type = data_type2
                other_name = name2
            else:
                tax_name = name2
                other_type = data_type1
                other_name = name1
                
            # Decide which table to join based on other type
            if other_type == 'keywords':
                other_table = "document_keyword"
                other_field = "keyword"
            else:  # Named entities
                other_table = "document_entity"
                other_field = "entity_value"
                
            # Build the query
            query = f"""
            SELECT 
                ud.document_id,
                ud.title,
                ud.author,
                ud.date,
                ud.database,
                ud.language,
                ds.heading_title,
                dsc.content AS chunk_text
            FROM taxonomy t
            JOIN document_section_chunk dsc ON t.chunk_id = dsc.id
            JOIN document_section ds ON dsc.document_section_id = ds.id
            JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
            JOIN {other_table} other ON ud.id = other.document_id
            WHERE 
                (t.category = :name1 OR t.subcategory = :name1 OR t.sub_subcategory = :name1)
                AND other.{other_field} = :name2
            """
            
            # Swap the parameter values if needed
            if data_type1 != 'taxonomy':
                params['name1'], params['name2'] = params['name2'], params['name1']
                
        else:
            # Both are keywords or entities or a mix
            if data_type1 == 'keywords' and data_type2 == 'keywords':
                # Both are keywords
                query = """
                SELECT 
                    ud.document_id,
                    ud.title,
                    ud.author,
                    ud.date,
                    ud.database,
                    ud.language,
                    ds.heading_title,
                    string_agg(dsc.content, ' ') AS chunk_text
                FROM document_keyword dk1
                JOIN document_keyword dk2 ON dk1.document_id = dk2.document_id
                JOIN uploaded_document ud ON dk1.document_id = ud.id
                JOIN document_section ds ON ud.id = ds.uploaded_document_id
                JOIN document_section_chunk dsc ON ds.id = dsc.document_section_id
                WHERE 
                    dk1.keyword = :name1
                    AND dk2.keyword = :name2
                    AND dk1.id != dk2.id
                GROUP BY ud.document_id, ud.title, ud.author, ud.date, ud.database, ud.language, ds.heading_title
                """
            elif data_type1 == 'named_entities' and data_type2 == 'named_entities':
                # Both are named entities
                query = """
                SELECT 
                    ud.document_id,
                    ud.title,
                    ud.author,
                    ud.date,
                    ud.database,
                    ud.language,
                    ds.heading_title,
                    string_agg(dsc.content, ' ') AS chunk_text
                FROM document_entity de1
                JOIN document_entity de2 ON de1.document_id = de2.document_id
                JOIN uploaded_document ud ON de1.document_id = ud.id
                JOIN document_section ds ON ud.id = ds.uploaded_document_id
                JOIN document_section_chunk dsc ON ds.id = dsc.document_section_id
                WHERE 
                    de1.entity_value = :name1
                    AND de2.entity_value = :name2
                    AND de1.id != de2.id
                GROUP BY ud.document_id, ud.title, ud.author, ud.date, ud.database, ud.language, ds.heading_title
                """
            else:
                # Mix of keyword and entity
                if data_type1 == 'keywords':
                    kw_name = name1
                    ent_name = name2
                else:
                    kw_name = name2
                    ent_name = name1
                    
                query = """
                SELECT 
                    ud.document_id,
                    ud.title,
                    ud.author,
                    ud.date,
                    ud.database,
                    ud.language,
                    ds.heading_title,
                    string_agg(dsc.content, ' ') AS chunk_text
                FROM document_keyword dk
                JOIN document_entity de ON dk.document_id = de.document_id
                JOIN uploaded_document ud ON dk.document_id = ud.id
                JOIN document_section ds ON ud.id = ds.uploaded_document_id
                JOIN document_section_chunk dsc ON ds.id = dsc.document_section_id
                WHERE 
                    dk.keyword = :kw_name
                    AND de.entity_value = :ent_name
                GROUP BY ud.document_id, ud.title, ud.author, ud.date, ud.database, ud.language, ds.heading_title
                """
                params = {'kw_name': kw_name, 'ent_name': ent_name, 'max_results': max_results}
        
        # Add common filters
        if language:
            query += " AND ud.language = :language"
            params['language'] = language
            
        if database:
            query += " AND ud.database = :database"
            params['database'] = database
            
        if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
            query += " AND ud.date BETWEEN :start_date AND :end_date"
            params['start_date'] = date_range[0]
            params['end_date'] = date_range[1]
            
        # Add limit and order
        query += " ORDER BY ud.date DESC LIMIT :max_results"
        
        # Execute query
        engine = get_engine()
        return pd.read_sql(text(query), engine, params=params)
        
    except Exception as e:
        logging.error(f"Error fetching concordance data for {element1} and {element2}: {e}")
        return pd.DataFrame()

def create_co_occurrence_network(
    burst_data: Dict[str, Dict[str, pd.DataFrame]],
    min_burst_intensity: float = 20.0,
    min_periods: int = 2,
    min_strength: float = 0.3,
    title: str = "Burst Co-occurrence Network",
    highlight_elements: Optional[List[str]] = None,
    include_cascade_patterns: bool = False
) -> go.Figure:
    """
    Create a network visualization of co-occurring burst elements.
    
    Args:
        burst_data: Dictionary mapping data types to dictionaries of element burst DataFrames
        min_burst_intensity: Minimum burst intensity to consider as a significant burst
        min_periods: Minimum number of periods where bursts must co-occur
        min_strength: Minimum co-occurrence strength to include in the network
        title: Chart title
        highlight_elements: Optional list of elements to highlight in the network
        include_cascade_patterns: Whether to include cascade pattern information
        
    Returns:
        go.Figure: Plotly Figure object with interactive network
    """
    # Detect co-occurrences
    co_occurrences = detect_burst_co_occurrences(
        burst_data,
        min_burst_intensity=min_burst_intensity,
        min_periods=min_periods
    )
    
    # If no co-occurrences found, return empty chart
    if not co_occurrences:
        return go.Figure().update_layout(title="No significant co-occurrences found")
    
    # Generate network graph structure
    network_data = generate_co_occurrence_network(co_occurrences, min_strength=min_strength)
    
    # If no nodes in network, return empty chart
    if not network_data['nodes']:
        return go.Figure().update_layout(
            title=f"No co-occurrences found with strength >= {min_strength}",
            annotations=[dict(
                text=f"Try lowering the minimum strength threshold (current: {min_strength})",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )]
        )
    
    # Create network using NetworkX for layout
    G = nx.Graph()
    
    # Add nodes with data type information
    for node in network_data['nodes']:
        G.add_node(node['id'], 
                   data_type=node['data_type'],
                   label=node['label'],
                   highlighted=node['id'] in (highlight_elements or []))
    
    # Add edges with weights
    for edge in network_data['edges']:
        G.add_edge(edge['source'], edge['target'], 
                   weight=edge['weight'],
                   periods=edge['periods'],
                   count=edge['count'])
    
    # Use a spring layout for the network
    pos = nx.spring_layout(G, seed=42, k=0.3)
    
    # Prepare node colors by data type
    node_colors = [NODE_COLORS.get(G.nodes[node]['data_type'], '#888888') for node in G.nodes()]
    
    # Adjust colors for highlighted nodes
    if highlight_elements:
        for i, node in enumerate(G.nodes()):
            if G.nodes[node]['highlighted']:
                # Brighten the node color
                node_colors[i] = 'rgba(255,255,0,1)'  # Bright yellow for highlighted nodes
    
    # Calculate node sizes based on degree and importance
    node_sizes = []
    for node in G.nodes():
        # Base size on degree
        size = 10 + 5 * G.degree(node) 
        # Increase size if highlighted
        if highlight_elements and node in highlight_elements:
            size *= 1.5
        node_sizes.append(size)
    
    # Prepare the figure
    edge_traces = []
    
    # Create edge traces with hover info
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G.edges[edge]['weight']
        periods = G.edges[edge]['periods']
        count = G.edges[edge]['count']
        
        # Format periods for display
        periods_str = ", ".join(str(p) for p in periods)
        
        # Scale line width by weight and adjust opacity
        width = 1 + 5 * weight
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=width, color=f'rgba(150,150,150,{max(0.3, weight)})', dash='solid'),
            mode='lines',
            hoverinfo='text',
            hovertext=f"Connection strength: {weight:.2f}<br>Common periods: {periods_str}<br>Shared bursts: {count}",
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Create node trace with hover info
    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        mode='markers+text',
        marker=dict(
            showscale=False,
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white')
        ),
        text=[G.nodes[node]['label'] for node in G.nodes()],
        textposition="top center",
        hoverinfo='text',
        hovertext=[f"{G.nodes[node]['label']} ({G.nodes[node]['data_type']})<br>Connections: {G.degree(node)}" 
                  for node in G.nodes()],
        showlegend=False
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
        height=700,
        plot_bgcolor='rgba(240,240,240,0.5)',
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
    
    # Add interactive notes about clicking
    fig.add_annotation(
        text="Click on nodes to explore related documents<br>Hover for details about connections",
        showarrow=False,
        x=0.5,
        y=1.05,
        xref="paper",
        yref="paper",
        font=dict(size=10),
        align="center",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#13376f",
        borderwidth=1
    )
    
    return fig

def create_enhanced_co_occurrence_network(
    burst_data: Dict[str, Dict[str, pd.DataFrame]],
    min_burst_intensity: float = 20.0,
    min_periods: int = 2,
    min_strength: float = 0.3,
    title: str = "Enhanced Burst Co-occurrence Network",
    include_community_detection: bool = True,
    include_centrality_metrics: bool = True,
    show_labels: bool = True
) -> go.Figure:
    """
    Create an enhanced network visualization with advanced network analysis features.
    
    Args:
        burst_data: Dictionary mapping data types to dictionaries of element burst DataFrames
        min_burst_intensity: Minimum burst intensity to consider as a significant burst
        min_periods: Minimum number of periods where bursts must co-occur
        min_strength: Minimum co-occurrence strength to include in the network
        title: Chart title
        include_community_detection: Whether to color nodes by community
        include_centrality_metrics: Whether to size nodes by centrality
        show_labels: Whether to show node labels
        
    Returns:
        go.Figure: Plotly Figure object with enhanced network visualization
    """
    # Detect co-occurrences
    co_occurrences = detect_burst_co_occurrences(
        burst_data,
        min_burst_intensity=min_burst_intensity,
        min_periods=min_periods
    )
    
    # If no co-occurrences found, return empty chart
    if not co_occurrences:
        return go.Figure().update_layout(title="No significant co-occurrences found")
    
    # Generate network graph structure
    network_data = generate_co_occurrence_network(co_occurrences, min_strength=min_strength)
    
    # If no nodes in network, return empty chart
    if not network_data['nodes']:
        return go.Figure().update_layout(
            title=f"No co-occurrences found with strength >= {min_strength}",
            annotations=[dict(
                text=f"Try lowering the minimum strength threshold (current: {min_strength})",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )]
        )
    
    # Create network using NetworkX for layout and analysis
    G = nx.Graph()
    
    # Add nodes with data type information
    for node in network_data['nodes']:
        G.add_node(node['id'], 
                   data_type=node['data_type'],
                   label=node['label'])
    
    # Add edges with weights
    for edge in network_data['edges']:
        G.add_edge(edge['source'], edge['target'], 
                   weight=edge['weight'],
                   periods=edge['periods'],
                   count=edge['count'])
    
    # Community detection (if requested)
    if include_community_detection and len(G.nodes()) > 2:
        try:
            import community as community_louvain
            communities = community_louvain.best_partition(G)
            nx.set_node_attributes(G, communities, 'community')
        except ImportError:
            # Fallback if python-louvain not installed
            logging.warning("python-louvain not installed. Using connected components instead.")
            communities = {node: i for i, component in enumerate(nx.connected_components(G)) 
                          for node in component}
            nx.set_node_attributes(G, communities, 'community')
    else:
        # Assign all to same community if detection not requested
        nx.set_node_attributes(G, {node: 0 for node in G.nodes()}, 'community')
    
    # Calculate centrality metrics (if requested)
    if include_centrality_metrics:
        # Eigenvector centrality
        try:
            eigen_centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
            nx.set_node_attributes(G, eigen_centrality, 'eigen_centrality')
        except:
            # Fallback to degree centrality if eigenvector fails to converge
            logging.warning("Eigenvector centrality failed to converge. Using degree centrality.")
            degree_centrality = nx.degree_centrality(G)
            nx.set_node_attributes(G, degree_centrality, 'eigen_centrality')
        
        # Betweenness centrality
        betweenness = nx.betweenness_centrality(G, weight='weight')
        nx.set_node_attributes(G, betweenness, 'betweenness')
    else:
        # Use degree as fallback
        nx.set_node_attributes(G, {node: G.degree(node) for node in G.nodes()}, 'eigen_centrality')
    
    # Use a force-directed layout with more space
    pos = nx.spring_layout(G, seed=42, k=0.4, iterations=100)
    
    # Prepare node colors based on community or data type
    if include_community_detection:
        # Get unique communities and map to colors
        communities = nx.get_node_attributes(G, 'community')
        unique_communities = sorted(set(communities.values()))
        import plotly.express as px
        community_colors = px.colors.qualitative.G10
        # Create color map
        color_map = {comm: community_colors[i % len(community_colors)] 
                    for i, comm in enumerate(unique_communities)}
        # Set node colors
        node_colors = [color_map[G.nodes[node]['community']] for node in G.nodes()]
    else:
        # Use data type colors
        node_colors = [NODE_COLORS.get(G.nodes[node]['data_type'], '#888888') for node in G.nodes()]
    
    # Calculate node sizes based on centrality or degree
    if include_centrality_metrics:
        # Scale eigen_centrality values to reasonable marker sizes
        centrality_values = [G.nodes[node]['eigen_centrality'] for node in G.nodes()]
        max_centrality = max(centrality_values) if centrality_values else 1
        min_size, max_size = 10, 50
        node_sizes = [min_size + (max_size - min_size) * (val / max_centrality) for val in centrality_values]
    else:
        # Scale by degree
        node_sizes = [10 + 2 * G.degree(node) for node in G.nodes()]
    
    # Prepare edge traces with custom hover info
    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = data.get('weight', 1.0)
        periods = data.get('periods', [])
        count = data.get('count', 0)
        
        # Format periods for display
        periods_str = ", ".join(str(p) for p in periods)
        
        # Scale line width by weight and adjust opacity
        width = 1 + 7 * weight
        
        # Custom curved edges for better visibility
        edge_trace = go.Scatter(
            x=[x0, (x0+x1)/2 + (y0-y1)/8, x1, None],  # Add slight curve using control point
            y=[y0, (y0+y1)/2 + (x1-x0)/8, y1, None],
            line=dict(width=width, color=f'rgba(150,150,150,{max(0.3, weight)})', dash='solid'),
            mode='lines',
            hoverinfo='text',
            hovertext=f"<b>{G.nodes[u]['label']} — {G.nodes[v]['label']}</b><br>" +
                      f"Connection strength: {weight:.2f}<br>" +
                      f"Common periods: {periods_str}<br>" +
                      f"Shared bursts: {count}",
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Prepare node trace with hover info
    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        mode='markers+text' if show_labels else 'markers',
        marker=dict(
            showscale=False,
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white')
        ),
        text=[G.nodes[node]['label'] for node in G.nodes()],
        textposition="top center",
        textfont=dict(size=10, color='black'),
        hoverinfo='text',
        hovertext=[f"<b>{G.nodes[node]['label']}</b> ({G.nodes[node]['data_type']})<br>" +
                   f"Connections: {G.degree(node)}<br>" +
                   (f"Centrality: {G.nodes[node]['eigen_centrality']:.3f}<br>" if include_centrality_metrics else "") +
                   (f"Community: {G.nodes[node]['community']}" if include_community_detection else "")
                   for node in G.nodes()],
        showlegend=False
    )
    
    # Combine all traces
    fig = go.Figure(data=edge_traces + [node_trace])
    
    # Create legend for communities or data types
    if include_community_detection:
        # Add legend for communities
        for comm in unique_communities:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=10, color=color_map[comm]),
                name=f'Community {comm}',
                showlegend=True
            ))
    else:
        # Legend for data types
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
        height=700,
        plot_bgcolor='rgba(240,240,240,0.5)',
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
                text="<b>Node size</b>: " +
                     ("Centrality" if include_centrality_metrics else "Number of connections") +
                     "<br><b>Edge width</b>: Co-occurrence strength" +
                     "<br><b>Node color</b>: " +
                     ("Community" if include_community_detection else "Data type"),
                showarrow=False,
                x=0.5,
                y=-0.2,
                xref="paper",
                yref="paper",
                font=dict(size=10)
            )
        ]
    )
    
    # Add interactive notes
    fig.add_annotation(
        text="Click on nodes to explore related documents<br>Double-click node to highlight its connections",
        showarrow=False,
        x=0.5,
        y=1.05,
        xref="paper",
        yref="paper",
        font=dict(size=10),
        align="center",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#13376f",
        borderwidth=1
    )
    
    return fig

def create_temporal_co_occurrence_network(
    burst_data: Dict[str, Dict[str, pd.DataFrame]],
    min_burst_intensity: float = 20.0,
    min_periods: int = 2,
    min_strength: float = 0.3,
    title: str = "Temporal Burst Co-occurrence Network",
    selected_period: Optional[str] = None,
    animation_frame: bool = False
) -> go.Figure:
    """
    Create a temporal network visualization showing how co-occurrences evolve over time.
    
    Args:
        burst_data: Dictionary mapping data types to dictionaries of element burst DataFrames
        min_burst_intensity: Minimum burst intensity to consider as a significant burst
        min_periods: Minimum number of periods where bursts must co-occur
        min_strength: Minimum co-occurrence strength to include in the network
        title: Chart title
        selected_period: Optional period to focus on (if not animating)
        animation_frame: Whether to create an animated visualization across periods
        
    Returns:
        go.Figure: Plotly Figure object with temporal network visualization
    """
    # Detect co-occurrences
    co_occurrences = detect_burst_co_occurrences(
        burst_data,
        min_burst_intensity=min_burst_intensity,
        min_periods=min_periods
    )
    
    # If no co-occurrences found, return empty chart
    if not co_occurrences:
        return go.Figure().update_layout(title="No significant co-occurrences found")
    
    # Get all periods from the data
    all_periods = set()
    for data_type, elements in burst_data.items():
        for element, df in elements.items():
            if 'period' in df.columns:
                all_periods.update(df['period'].unique())
    
    # Sort periods
    try:
        all_periods = sorted(all_periods)
    except:
        all_periods = list(all_periods)
    
    # Use selected period or default to last period
    if selected_period and selected_period in all_periods:
        active_period = selected_period
    else:
        active_period = all_periods[-1] if all_periods else None
    
    # If no periods found, return empty chart
    if not active_period:
        return go.Figure().update_layout(title="No period data available")
    
    # Generate network graph structure
    network_data = generate_co_occurrence_network(co_occurrences, min_strength=min_strength)
    
    # If no nodes in network, return empty chart
    if not network_data['nodes']:
        return go.Figure().update_layout(
            title=f"No co-occurrences found with strength >= {min_strength}",
            annotations=[dict(
                text=f"Try lowering the minimum strength threshold (current: {min_strength})",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )]
        )
    
    # Create network using NetworkX for layout
    G = nx.Graph()
    
    # Add nodes with data type information
    for node in network_data['nodes']:
        G.add_node(node['id'], 
                   data_type=node['data_type'],
                   label=node['label'])
    
    # Add edges with weights
    for edge in network_data['edges']:
        G.add_edge(edge['source'], edge['target'], 
                   weight=edge['weight'],
                   periods=edge['periods'],
                   count=edge['count'])
    
    # Create a consistent layout for all frames
    pos = nx.spring_layout(G, seed=42, k=0.3)
    
    # If not animating, create a single frame for the selected period
    if not animation_frame:
        # Filter edges for this period
        active_edges = []
        for u, v, data in G.edges(data=True):
            if active_period in data.get('periods', []):
                active_edges.append((u, v, data))
        
        # Get active nodes (with at least one edge in this period)
        active_nodes = set()
        for u, v, _ in active_edges:
            active_nodes.add(u)
            active_nodes.add(v)
        
        # Prepare node colors by data type
        node_colors = [NODE_COLORS.get(G.nodes[node]['data_type'], '#888888') if node in active_nodes 
                      else 'rgba(200,200,200,0.3)' for node in G.nodes()]
        
        # Prepare node sizes - larger for active nodes
        node_sizes = [15 + 5 * G.degree(node) if node in active_nodes else 8 for node in G.nodes()]
        
        # Create edge traces for active edges
        edge_traces = []
        for u, v, data in active_edges:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            weight = data.get('weight', 1.0)
            periods = data.get('periods', [])
            count = data.get('count', 0)
            
            # Format periods for display
            periods_str = ", ".join(str(p) for p in periods)
            
            # Scale line width by weight
            width = 1 + 5 * weight
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=width, color=f'rgba(150,150,150,{max(0.5, weight)})', dash='solid'),
                mode='lines',
                hoverinfo='text',
                hovertext=f"Connection strength: {weight:.2f}<br>Common periods: {periods_str}<br>Shared bursts: {count}",
                showlegend=False
            )
            edge_traces.append(edge_trace)
            
        # Add inactive edges (faded)
        for u, v, data in G.edges(data=True):
            if active_period not in data.get('periods', []):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_trace = go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    line=dict(width=1, color='rgba(200,200,200,0.2)', dash='dot'),
                    mode='lines',
                    hoverinfo='skip',
                    showlegend=False
                )
                edge_traces.append(edge_trace)
        
        # Create node trace
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            marker=dict(
                showscale=False,
                size=node_sizes,
                color=node_colors,
                line=dict(width=1, color='white')
            ),
            text=[G.nodes[node]['label'] for node in G.nodes()],
            textposition="top center",
            textfont=dict(
                size=10, 
                color=['black' if node in active_nodes else 'rgba(150,150,150,0.5)' for node in G.nodes()]
            ),
            hoverinfo='text',
            hovertext=[f"{G.nodes[node]['label']} ({G.nodes[node]['data_type']})<br>{'Active in this period' if node in active_nodes else 'Inactive in this period'}" 
                      for node in G.nodes()],
            showlegend=False
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
            ),
            go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=10, color='rgba(200,200,200,0.3)'),
                name='Inactive Elements'
            )
        ]
        
        # Add legend traces
        for trace in legend_traces:
            fig.add_trace(trace)
        
        # Update layout
        fig.update_layout(
            title=f"{title} - {active_period}",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            height=700,
            plot_bgcolor='rgba(240,240,240,0.5)',
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
                    text="<b>Node size</b>: Number of connections<br><b>Edge width</b>: Co-occurrence strength<br><b>Highlighted</b>: Active in this period",
                    showarrow=False,
                    x=0.5,
                    y=-0.2,
                    xref="paper",
                    yref="paper",
                    font=dict(size=10)
                )
            ]
        )
        
        # Add period selection dropdown
        period_buttons = []
        for period in all_periods:
            period_buttons.append(
                dict(
                    method="relayout",
                    args=[{"title": f"{title} - {period}"}],
                    label=str(period)
                )
            )
        
        fig.update_layout(
            updatemenus=[{
                "buttons": period_buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.1,
                "y": 1.15,
            }]
        )
        
        # Label for the dropdown
        fig.add_annotation(
            x=0.1,
            y=1.2,
            xref="paper",
            yref="paper",
            text="Select Period:",
            showarrow=False,
            font=dict(size=12, color="black")
        )
        
    else:
        # Create animated visualization (this is a more complex case)
        # Implementation depends on specific requirements and may need
        # more sophisticated approach with Plotly's frame-based animation
        
        # For now, return the single frame visualization with a note
        fig = go.Figure().update_layout(
            title="Animated temporal network visualization not implemented",
            annotations=[dict(
                text="Please use the period selection dropdown to view different time periods",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )]
        )
    
    return fig