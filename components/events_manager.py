#!/usr/bin/env python
# coding: utf-8

"""
Events manager component for managing historical events in timelines.
Provides functionality to add, edit, delete, load, and save events for enhanced
timeline visualizations. Integrates with burst timeline visualization.
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State, ALL, MATCH
import dash_bootstrap_components as dbc
import pandas as pd

# Default historical events (matches the ones defined in burstiness.py)
DEFAULT_HISTORICAL_EVENTS = [
    {
        "date": "2022-02-24",
        "period": "Feb 2022",
        "event": "Russian Invasion of Ukraine Begins",
        "impact": 1.0,
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

class EventsManager:
    """
    Class to manage historical events for enhanced timeline visualizations.
    Provides methods to add, edit, delete, load, and save events.
    """

    def __init__(self, events_path: Optional[str] = None):
        """
        Initialize the events manager with default events or from a file.
        
        Args:
            events_path: Optional path to a JSON file containing events
        """
        self.events_path = events_path
        self.events = self.load_events(events_path) if events_path else DEFAULT_HISTORICAL_EVENTS.copy()
        
    def load_events(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load historical events from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing events
            
        Returns:
            List of dictionaries containing historical event data
        """
        if not os.path.exists(file_path):
            logging.warning(f"Events file {file_path} not found. Using default events.")
            return DEFAULT_HISTORICAL_EVENTS.copy()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                events = json.load(f)
            logging.info(f"Loaded {len(events)} historical events from {file_path}")
            
            # Validate event structure
            validated_events = []
            for event in events:
                if 'event' in event and 'period' in event:
                    # Ensure required fields are present
                    if 'date' not in event:
                        event['date'] = event.get('period', "")
                    if 'impact' not in event:
                        event['impact'] = 0.5
                    if 'description' not in event:
                        event['description'] = ""
                    validated_events.append(event)
                else:
                    logging.warning(f"Skipping invalid event: {event}")
            
            return validated_events
        except Exception as e:
            logging.error(f"Error loading historical events from {file_path}: {e}")
            return DEFAULT_HISTORICAL_EVENTS.copy()
    
    def save_events(self, file_path: Optional[str] = None) -> bool:
        """
        Save historical events to a JSON file.
        
        Args:
            file_path: Path to save the JSON file (defaults to self.events_path)
            
        Returns:
            bool: True if successful, False otherwise
        """
        path = file_path or self.events_path
        if not path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"historical_events_{timestamp}.json"
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.events, f, indent=2)
            logging.info(f"Saved {len(self.events)} historical events to {path}")
            return True
        except Exception as e:
            logging.error(f"Error saving historical events to {path}: {e}")
            return False
    
    def add_event(self, event: Dict[str, Any]) -> bool:
        """
        Add a new historical event.
        
        Args:
            event: Dictionary containing event data
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not event.get('event') or not event.get('period'):
            return False
        
        # Ensure required fields
        if 'date' not in event:
            event['date'] = event.get('period', "")
        if 'impact' not in event:
            event['impact'] = 0.5
        if 'description' not in event:
            event['description'] = ""
        
        self.events.append(event)
        return True
    
    def update_event(self, index: int, event: Dict[str, Any]) -> bool:
        """
        Update an existing historical event.
        
        Args:
            index: Index of the event to update
            event: New event data
            
        Returns:
            bool: True if successful, False otherwise
        """
        if index < 0 or index >= len(self.events):
            return False
        
        if not event.get('event') or not event.get('period'):
            return False
        
        # Ensure required fields
        if 'date' not in event:
            event['date'] = event.get('period', "")
        if 'impact' not in event:
            event['impact'] = 0.5
        if 'description' not in event:
            event['description'] = ""
        
        self.events[index] = event
        return True
    
    def delete_event(self, index: int) -> bool:
        """
        Delete a historical event.
        
        Args:
            index: Index of the event to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        if index < 0 or index >= len(self.events):
            return False
        
        self.events.pop(index)
        return True
    
    def get_events(self) -> List[Dict[str, Any]]:
        """
        Get all historical events.
        
        Returns:
            List of dictionaries containing historical event data
        """
        return self.events
    
    def clear_events(self) -> None:
        """Clear all events."""
        self.events = []


def create_events_manager_layout(id_prefix: str) -> html.Div:
    """
    Create the events manager layout component.
    
    Args:
        id_prefix: Prefix for component IDs
        
    Returns:
        html.Div: Events manager layout component
    """
    return html.Div([
        # Control buttons row
        dbc.Row([
            dbc.Col([
                html.Label("Historical Events Management"),
                dbc.Checklist(
                    id=f"{id_prefix}-include-events",
                    options=[{'label': 'Show Historical Events on Timeline', 'value': 'show_events'}],
                    value=['show_events'],
                    switch=True,
                ),
            ], width=6),
            dbc.Col([
                html.Div([
                    dbc.Button(
                        "Add New Event", 
                        id=f"{id_prefix}-add-event-btn", 
                        color="primary", 
                        size="sm",
                        className="me-2"
                    ),
                    dbc.Button(
                        "Import Events", 
                        id=f"{id_prefix}-import-events-btn", 
                        color="success", 
                        size="sm",
                        className="me-2"
                    ),
                    dbc.Button(
                        "Export Events", 
                        id=f"{id_prefix}-export-events-btn", 
                        color="success", 
                        size="sm"
                    ),
                ], className="d-flex justify-content-end", style={"marginTop": "20px"})
            ], width=6),
        ]),
        
        # Events table
        dbc.Row([
            dbc.Col([
                html.Div(
                    id=f"{id_prefix}-events-table-container",
                    children=[
                        # This will be populated by callback
                        html.Div(id=f"{id_prefix}-events-table")
                    ],
                    style={"maxHeight": "300px", "overflowY": "auto", "marginTop": "10px"}
                ),
            ], width=12),
        ]),
        
        # Event edit/add modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(id=f"{id_prefix}-event-modal-title")),
            dbc.ModalBody([
                dbc.Form([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Event Name"),
                            dbc.Input(id=f"{id_prefix}-event-name", type="text", placeholder="Enter event name"),
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Event Date"),
                            dbc.Input(id=f"{id_prefix}-event-date", type="date"),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Period"),
                            dbc.Input(id=f"{id_prefix}-event-period", type="text", placeholder="e.g., Feb 2023"),
                        ], width=6),
                    ], className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Event Description"),
                            dbc.Textarea(id=f"{id_prefix}-event-description", placeholder="Describe the event"),
                        ], width=12),
                    ], className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Event Impact (0.1-1.0)"),
                            dcc.Slider(
                                id=f'{id_prefix}-event-impact',
                                min=0.1,
                                max=1.0,
                                step=0.1,
                                value=0.5,
                                marks={i/10: str(i/10) for i in range(1, 11)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                        ], width=12),
                    ], className="mt-3"),
                    # Hidden input to store event index for editing
                    dbc.Input(id=f"{id_prefix}-event-index", type="hidden", value="-1"),
                ]),
                dbc.Alert(
                    "Please fill in all required fields",
                    id=f"{id_prefix}-event-alert",
                    color="danger",
                    dismissable=True,
                    is_open=False,
                    className="mt-3"
                ),
            ]),
            dbc.ModalFooter([
                dbc.Button(
                    "Close", id=f"{id_prefix}-close-event-modal", className="me-2"
                ),
                dbc.Button(
                    "Save Event", id=f"{id_prefix}-save-event", color="success"
                )
            ]),
        ], id=f"{id_prefix}-event-modal", is_open=False),
        
        # Import modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Import Events")),
            dbc.ModalBody([
                html.P("Select a JSON file containing historical events to import:"),
                dcc.Upload(
                    id=f'{id_prefix}-upload-events',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select a File')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px 0'
                    },
                    multiple=False
                ),
                dbc.Alert(
                    id=f"{id_prefix}-import-alert",
                    dismissable=True,
                    is_open=False,
                    className="mt-3"
                ),
            ]),
            dbc.ModalFooter([
                dbc.Button("Close", id=f"{id_prefix}-close-import-modal", className="ms-auto")
            ]),
        ], id=f"{id_prefix}-import-modal", is_open=False),
        
        # Hidden download component for exports
        dcc.Download(id=f"{id_prefix}-download-events"),
    ])


def register_events_manager_callbacks(app, events_manager: EventsManager, id_prefix: str) -> None:
    """
    Register callbacks for the events manager component.
    
    Args:
        app: Dash application instance
        events_manager: EventsManager instance
        id_prefix: Prefix for component IDs
    """
    # Callback to show/hide the event modal
    @app.callback(
        [
            Output(f"{id_prefix}-event-modal", "is_open"),
            Output(f"{id_prefix}-event-modal-title", "children"),
            Output(f"{id_prefix}-event-name", "value"),
            Output(f"{id_prefix}-event-date", "value"),
            Output(f"{id_prefix}-event-period", "value"),
            Output(f"{id_prefix}-event-description", "value"),
            Output(f"{id_prefix}-event-impact", "value"),
            Output(f"{id_prefix}-event-index", "value"),
        ],
        [
            Input(f"{id_prefix}-add-event-btn", "n_clicks"),
            Input(f"{id_prefix}-close-event-modal", "n_clicks"),
            Input(f"{id_prefix}-save-event", "n_clicks"),
            Input({"type": f"{id_prefix}-edit-event", "index": ALL}, "n_clicks"),
        ],
        [
            State(f"{id_prefix}-event-modal", "is_open"),
            State({"type": f"{id_prefix}-edit-event", "index": ALL}, "id"),
        ],
        prevent_initial_call=True
    )
    def toggle_event_modal(add_clicks, close_clicks, save_clicks, edit_clicks, is_open, edit_button_ids):
        ctx = dash.callback_context
        if not ctx.triggered:
            return is_open, "Edit Event", "", "", "", "", 0.5, "-1"
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Adding a new event
        if button_id == f"{id_prefix}-add-event-btn":
            return True, "Add New Event", "", "", "", "", 0.5, "-1"
        
        # Editing an existing event
        if "index" in button_id:
            try:
                button_data = json.loads(button_id)
                event_index = button_data['index']
                if 0 <= event_index < len(events_manager.events):
                    event = events_manager.events[event_index]
                    return (
                        True, 
                        "Edit Event", 
                        event.get('event', ''),
                        event.get('date', ''),
                        event.get('period', ''),
                        event.get('description', ''),
                        event.get('impact', 0.5),
                        str(event_index)
                    )
            except Exception as e:
                logging.error(f"Error getting event data for editing: {e}")
        
        # Closing the modal
        if button_id in [f"{id_prefix}-close-event-modal", f"{id_prefix}-save-event"]:
            return False, "Edit Event", "", "", "", "", 0.5, "-1"
        
        return is_open, "Edit Event", "", "", "", "", 0.5, "-1"
    
    # Callback to save the event
    @app.callback(
        [
            Output(f"{id_prefix}-event-alert", "is_open"),
            Output(f"{id_prefix}-event-alert", "children"),
            Output(f"{id_prefix}-events-table", "children"),
        ],
        [
            Input(f"{id_prefix}-save-event", "n_clicks"),
            Input({"type": f"{id_prefix}-delete-event", "index": ALL}, "n_clicks"),
        ],
        [
            State(f"{id_prefix}-event-name", "value"),
            State(f"{id_prefix}-event-date", "value"),
            State(f"{id_prefix}-event-period", "value"),
            State(f"{id_prefix}-event-description", "value"),
            State(f"{id_prefix}-event-impact", "value"),
            State(f"{id_prefix}-event-index", "value"),
            State({"type": f"{id_prefix}-delete-event", "index": ALL}, "id"),
        ],
        prevent_initial_call=True
    )
    def manage_events(save_clicks, delete_clicks, event_name, event_date, event_period, 
                      event_description, event_impact, event_index, delete_button_ids):
        ctx = dash.callback_context
        if not ctx.triggered:
            return False, "", create_events_table(events_manager.events, id_prefix)
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Save event
        if button_id == f"{id_prefix}-save-event" and save_clicks:
            if not event_name or not event_period:
                return True, "Please fill in both the event name and period fields.", dash.no_update
            
            event = {
                "event": event_name,
                "date": event_date or event_period,
                "period": event_period,
                "description": event_description or "",
                "impact": event_impact or 0.5
            }
            
            # Add or update event
            try:
                idx = int(event_index)
                if idx >= 0:
                    # Update existing event
                    success = events_manager.update_event(idx, event)
                    message = "Event updated successfully!" if success else "Error updating event."
                else:
                    # Add new event
                    success = events_manager.add_event(event)
                    message = "Event added successfully!" if success else "Error adding event."
                
                return not success, message, create_events_table(events_manager.events, id_prefix)
            except Exception as e:
                logging.error(f"Error managing event: {e}")
                return True, f"Error: {str(e)}", dash.no_update
        
        # Delete event
        if "index" in button_id:
            try:
                button_data = json.loads(button_id)
                event_index = button_data['index']
                success = events_manager.delete_event(event_index)
                return not success, "Event deleted successfully!" if success else "Error deleting event.", create_events_table(events_manager.events, id_prefix)
            except Exception as e:
                logging.error(f"Error deleting event: {e}")
                return True, f"Error: {str(e)}", dash.no_update
        
        return False, "", create_events_table(events_manager.events, id_prefix)
    
    # Callback to handle import/export modals
    @app.callback(
        [
            Output(f"{id_prefix}-import-modal", "is_open"),
        ],
        [
            Input(f"{id_prefix}-import-events-btn", "n_clicks"),
            Input(f"{id_prefix}-close-import-modal", "n_clicks"),
        ],
        [State(f"{id_prefix}-import-modal", "is_open")],
        prevent_initial_call=True
    )
    def toggle_import_modal(import_clicks, close_clicks, is_open):
        ctx = dash.callback_context
        if not ctx.triggered:
            return is_open
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == f"{id_prefix}-import-events-btn":
            return True
        elif button_id == f"{id_prefix}-close-import-modal":
            return False
        
        return is_open
    
    # Callback to handle file uploads for import
    @app.callback(
        [
            Output(f"{id_prefix}-import-alert", "is_open"),
            Output(f"{id_prefix}-import-alert", "children"),
            Output(f"{id_prefix}-import-alert", "color"),
            Output(f"{id_prefix}-events-table-container", "children"),
        ],
        [Input(f"{id_prefix}-upload-events", "contents")],
        [State(f"{id_prefix}-upload-events", "filename")],
        prevent_initial_call=True
    )
    def import_events(contents, filename):
        if contents is None:
            return False, "", "danger", dash.no_update
        
        try:
            # Parse the uploaded file
            content_type, content_string = contents.split(',')
            import base64
            import io
            decoded = base64.b64decode(content_string)
            
            if filename.endswith('.json'):
                # Load the JSON data
                events = json.loads(decoded.decode('utf-8'))
                
                # Validate events structure
                if not isinstance(events, list):
                    return True, "Invalid events file format. Expected a list of events.", "danger", dash.no_update
                
                # Replace current events with imported ones
                events_manager.events = []
                for event in events:
                    if isinstance(event, dict) and 'event' in event and 'period' in event:
                        events_manager.add_event(event)
                
                # Create updated table
                table = create_events_table(events_manager.events, id_prefix)
                
                return True, f"Successfully imported {len(events_manager.events)} events!", "success", table
            else:
                return True, "Please upload a JSON file.", "danger", dash.no_update
                
        except Exception as e:
            logging.error(f"Error importing events: {e}")
            return True, f"Error importing events: {str(e)}", "danger", dash.no_update
    
    # Callback to export events
    @app.callback(
        Output(f"{id_prefix}-download-events", "data"),
        Input(f"{id_prefix}-export-events-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def export_events(n_clicks):
        if not n_clicks:
            return dash.no_update
            
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"historical_events_{timestamp}.json"
        
        # Convert to JSON string
        json_str = json.dumps(events_manager.events, indent=2)
        
        return dict(content=json_str, filename=filename)


def create_events_table(events: List[Dict[str, Any]], id_prefix: str) -> html.Div:
    """
    Create an HTML table to display the events.
    
    Args:
        events: List of event dictionaries
        id_prefix: Prefix for component IDs
        
    Returns:
        html.Div: Table component
    """
    if not events:
        return html.Div("No events added yet. Click 'Add New Event' to create one.", 
                       className="text-center text-muted my-3")
    
    return html.Table(
        # Header
        [html.Tr([
            html.Th("Event", style={"width": "30%"}),
            html.Th("Period", style={"width": "20%"}),
            html.Th("Impact", style={"width": "20%"}),
            html.Th("Actions", style={"width": "30%"})
        ])] +
        # Rows
        [html.Tr([
            html.Td(event["event"]),
            html.Td(event["period"]),
            html.Td(f"{event['impact']:.1f}"),
            html.Td([
                dbc.Button(
                    "Edit", 
                    id={"type": f"{id_prefix}-edit-event", "index": i},
                    color="primary", 
                    size="sm",
                    className="me-1"
                ),
                dbc.Button(
                    "Delete", 
                    id={"type": f"{id_prefix}-delete-event", "index": i},
                    color="danger", 
                    size="sm",
                    className="me-1"
                )
            ])
        ]) for i, event in enumerate(events)],
        className="table table-striped table-hover table-sm"
    )


def create_events_manager_collapsible(id_prefix: str) -> dbc.Card:
    """
    Create a collapsible card containing the events manager.
    
    Args:
        id_prefix: Prefix for component IDs
        
    Returns:
        dbc.Card: Collapsible card with events manager
    """
    return dbc.Card([
        dbc.CardHeader([
            dbc.Button(
                "Historical Events", 
                id=f"{id_prefix}-events-toggle",
                className="w-100 text-start",
                color="danger"
            )
        ]),
        dbc.Collapse(
            dbc.CardBody([
                create_events_manager_layout(id_prefix)
            ]),
            id=f"{id_prefix}-events-collapse",
            is_open=False,
        )
    ])


def create_timeline_with_events(
    timeline_data: pd.DataFrame, 
    events_manager: EventsManager, 
    title: str = "Timeline with Historical Events"
) -> dict:
    """
    Create a timeline visualization with historical events from the events manager.
    Integrates with visualizations.bursts.create_enhanced_citespace_timeline.
    
    Args:
        timeline_data: DataFrame with element, period, and burst_intensity
        events_manager: EventsManager instance with events to display
        title: Chart title
        
    Returns:
        dict: Parameters to pass to create_enhanced_citespace_timeline
    """
    from visualizations.bursts import create_enhanced_citespace_timeline
    
    # Get events from the manager
    events = events_manager.get_events()
    
    # Create the timeline visualization
    return {
        'summary_df': timeline_data,
        'historical_events': events,
        'title': title,
        'show_annotations': True
    }