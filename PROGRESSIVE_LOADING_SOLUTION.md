# Progressive Loading Solution

## The Problem
The Explore tab was loading EVERYTHING at once:
- Sunburst chart
- All chunks
- Timeline chart
- All statistics

This caused the radar pulse to stay visible for too long and made the app feel slow.

## The Solution: Progressive Loading

### 1. Priority Loading (Immediate)
Only load what the user needs to see first:
- Sunburst chart
- First 10 chunks when a segment is clicked

Once these are loaded, the radar pulse STOPS and users can interact.

### 2. Background Loading (Deferred)
Continue loading in the background:
- Timeline chart
- Additional chunk pages
- Detailed statistics

A subtle indicator shows background loading is happening.

## Implementation

### JavaScript Side (`progressive-loading.js`)
- Monitors priority components: `sunburst-chart`, `explore-chunks-container`
- Hides radar pulse when priority components are loaded
- Shows subtle "Loading additional data..." indicator for background tasks

### Python Side (Callbacks)
Instead of one massive callback that loads everything:

**Before:**
```python
@app.callback(
    [sunburst, stats, chunks, timeline, pagination, ...],  # 9 outputs!
    Input('sunburst-chart', 'clickData'),
    ...
)
def load_everything_at_once():  # SLOW!
```

**After:**
```python
# Priority callback 1: Just the sunburst
@app.callback(
    [Output('sunburst-chart', 'figure'), Output('sunburst-loading-complete', 'data')],
    Input('explore-filter-button', 'n_clicks')
)

# Priority callback 2: Just first batch of chunks
@app.callback(
    [Output('explore-chunks-container', 'children'), Output('chunks-loading-complete', 'data')],
    Input('sunburst-chart', 'clickData')
)

# Background callback: Timeline (waits for chunks to complete)
@app.callback(
    Output('timeline-chart', 'figure'),
    Input('chunks-loading-complete', 'data')  # Triggered after chunks load
)
```

## Benefits
1. **Faster perceived performance** - Users see content immediately
2. **Better UX** - Radar pulse disappears once priority content is loaded
3. **Progressive enhancement** - Additional features load without blocking
4. **Scalable** - Easy to add more background tasks

## Files Created/Modified
1. `static/progressive-loading.js` - Manages progressive loading states
2. `tabs/explore_progressive.py` - Split callbacks implementation
3. `tabs/explore.py` - Added loading signal stores
4. `app.py` - Included progressive loading script

## Testing
You can monitor loading in the browser console:
```javascript
progressiveLoading.getLoadingStates()  // See what's loaded
progressiveLoading.checkPriority()     // Check if priority items are done
```