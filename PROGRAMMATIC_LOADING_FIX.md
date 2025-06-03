# Programmatic Loading State Monitor

## Problem
The radar pulse loading animation was not properly synchronized with actual loading states. It would disappear before all visualizations were fully rendered because it wasn't tracking:
- Individual graph rendering states
- Plotly-specific loading events
- Nested component loading states
- Callback execution states

## Solution
Created a comprehensive loading state monitor (`dash-loading-monitor.js`) that programmatically tracks ALL loading activity:

### 1. Component Loading States
- Monitors `data-dash-is-loading` attributes
- Tracks Dash loading classes (`_dash-loading`, `_dash-loading-callback`, `dash-spinner`)
- Uses MutationObserver to detect all DOM changes

### 2. Plotly Graph Loading
- Intercepts `Plotly.newPlot()` and `Plotly.react()` calls
- Tracks each graph's rendering lifecycle
- Monitors graph element completion

### 3. Callback Tracking
- Hooks into Dash's callback context
- Tracks active callback executions
- Ensures loading persists during data fetching

### 4. Smart Cleanup
- Only hides radar sweep when ALL loading states are complete
- Uses a 500ms delay to catch any late-loading elements
- Maintains a comprehensive state map of all active loading

## How It Works

1. **Initialization**: The script hooks into Dash and Plotly APIs on page load
2. **State Tracking**: Maintains Maps and Sets of all active loading states
3. **Visual Feedback**: Shows radar sweep when ANY loading is active
4. **Cleanup**: Only hides when ALL tracked states are complete

## Debug Features
In browser console, you can use:
```javascript
// Check current loading states
dashLoadingMonitor.getStates()

// Force hide/show for testing
dashLoadingMonitor.forceHide()
dashLoadingMonitor.forceShow()
```

## Benefits
- No hardcoded timeouts
- Accurate representation of actual loading state
- Works with dynamic content and lazy loading
- Handles nested components and callbacks
- Debug mode for troubleshooting

## Files Created/Modified
1. `static/dash-loading-monitor.js` - Comprehensive loading state monitor
2. `utils/graph_helpers.py` - Helper functions for wrapping graphs
3. `app.py` - Updated to use new loading monitor

The radar pulse now accurately reflects the true loading state of the application, staying visible until ALL components, graphs, and callbacks have completed.