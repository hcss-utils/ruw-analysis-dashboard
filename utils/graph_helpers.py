"""
Helper functions for creating graphs with proper loading states.
"""

from dash import dcc, html


def create_loading_graph(figure, graph_id=None, config=None, **kwargs):
    """
    Create a graph wrapped in a loading component to ensure proper loading state tracking.
    
    Args:
        figure: Plotly figure object
        graph_id: Optional ID for the graph
        config: Optional config dict for the graph
        **kwargs: Additional keyword arguments for dcc.Graph
        
    Returns:
        dcc.Loading: Loading component containing the graph
    """
    # Default config if not provided
    if config is None:
        config = {'displayModeBar': False, 'responsive': True}
    
    # Ensure responsive is True
    if 'responsive' not in config:
        config['responsive'] = True
    
    # Create graph component
    graph = dcc.Graph(
        figure=figure,
        config=config,
        **kwargs
    )
    
    # Add ID if provided
    if graph_id:
        graph.id = graph_id
    
    # Wrap in loading component
    return dcc.Loading(
        children=graph,
        type="none",  # We use "none" because our custom JS handles the visual
        className="graph-loading-wrapper"
    )


def wrap_graphs_with_loading(component):
    """
    Recursively wrap all dcc.Graph components with loading indicators.
    
    Args:
        component: Dash component or list of components
        
    Returns:
        Component with graphs wrapped in loading
    """
    # Handle None
    if component is None:
        return None
    
    # Handle lists
    if isinstance(component, list):
        return [wrap_graphs_with_loading(item) for item in component]
    
    # Handle dcc.Graph
    if hasattr(component, '__class__') and component.__class__.__name__ == 'Graph':
        # Extract properties
        figure = getattr(component, 'figure', None)
        graph_id = getattr(component, 'id', None)
        config = getattr(component, 'config', None)
        
        # Get other properties
        kwargs = {}
        for prop in ['className', 'style', 'responsive']:
            if hasattr(component, prop):
                kwargs[prop] = getattr(component, prop)
        
        return create_loading_graph(figure, graph_id, config, **kwargs)
    
    # Handle components with children
    if hasattr(component, 'children'):
        # Recursively process children
        wrapped_children = wrap_graphs_with_loading(component.children)
        
        # Create new component with wrapped children
        try:
            # Try to create a new instance of the same type
            new_component = component.__class__(children=wrapped_children)
            
            # Copy over other properties
            for prop in dir(component):
                if not prop.startswith('_') and prop != 'children':
                    try:
                        value = getattr(component, prop)
                        if not callable(value):
                            setattr(new_component, prop, value)
                    except:
                        pass
            
            return new_component
        except:
            # If we can't recreate, just update children
            component.children = wrapped_children
            return component
    
    # Return unchanged if not a graph or container
    return component