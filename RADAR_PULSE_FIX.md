# Radar Pulse Loading Animation Fix

## Summary
Fixed the radar pulse loading animation to ensure it displays properly when:
1. Clicking on a segment in the Explore tab
2. Filtering Named Entities by entity type in the Sources tab
3. Any other loading scenario in the dashboard

## Changes Made

### 1. Enhanced CSS Implementation (`static/custom.css`)
- Added more specific CSS selectors to target Dash loading elements
- Created multiple approaches to ensure the radar pulse displays:
  - Direct targeting of `._dash-loading` and `._dash-loading-callback` classes
  - Targeting `div[data-dash-is-loading="true"]` attributes
  - Specific handling for graph and table pending states
- Added semi-transparent overlay during loading
- Ensured proper z-index layering (9999 for spinner, 9998 for overlay)

### 2. JavaScript Enhancement (`static/loading.js`)
- Added MutationObserver to watch for dynamically added loading elements
- Created `enhanceLoadingSpinner()` function that:
  - Hides default Dash spinner content
  - Creates custom radar pulse structure with 3 expanding rings
  - Adds center dot with glow effect
  - Ensures proper positioning and visibility
- Applied radar pulse to both Dash loading callbacks and error loading scenarios

### 3. App Configuration (`app.py`)
- Added loading.js script inclusion in the HTML template
- Updated CSS version to force cache refresh (v=3)

## Technical Details

### Radar Pulse Structure
```
- 3 expanding circular rings with different delays (0s, 0.5s, 1s)
- Ring colors with decreasing opacity (100%, 70%, 40%)
- Center dot (12px) with box shadow glow
- Animation duration: 2 seconds per cycle
- Expansion from 30px to 100px diameter
```

### CSS Specificity
The implementation uses multiple targeting strategies to ensure compatibility:
1. Class-based: `._dash-loading`, `._dash-loading-callback`
2. Attribute-based: `div[data-dash-is-loading="true"]`
3. State-based: `.dash-graph--pending`, `.dash-table--pending`

### JavaScript Enhancement
- Uses MutationObserver for performance
- Marks enhanced elements to prevent duplicate processing
- Creates DOM elements dynamically for maximum compatibility
- Inline styles ensure visibility regardless of CSS load order

## Testing
To verify the fix:
1. Clear browser cache (Ctrl+F5)
2. Navigate to Explore tab and click on any sunburst segment
3. Navigate to Sources tab > Named Entities subtab and change entity type filter
4. The radar pulse should appear as expanding blue circles during data loading

## Browser Compatibility
- Chrome/Edge: Full support
- Firefox: Full support
- Safari: Full support (with webkit prefixes handled)
- IE11: Graceful degradation to standard spinner