# Horizontal Radar Sweep Loading Animation

## Summary
Implemented a horizontal radar sweep loading animation that displays a translucent blue band sweeping across the screen from left to right when:
1. Clicking on a segment in the Explore tab
2. Filtering Named Entities by entity type in the Sources tab
3. Any other loading scenario in the dashboard

## Changes Made

### 1. CSS Implementation (`static/custom.css`)
- Removed circular pulse animations
- Implemented horizontal radar sweep with gradient opacity
- The sweep band uses the same blue color (#13376f) as the About boxes
- Added gradient effect: transparent → 5% → 10% → 15% (peak) → 10% → 5% → transparent
- Semi-transparent white overlay (30% opacity) for subtle background dimming
- Animation duration: 2 seconds, ease-in-out timing

### 2. JavaScript Enhancement (`static/loading.js`)
- Modified to create horizontal sweep instead of circular pulses
- Creates a full-screen container with the sweeping band
- Automatically cleans up after loading completes or after 10 seconds
- Uses MutationObserver to detect when loading states change
- Hides default Dash spinners completely

### 3. App Configuration (`app.py`)
- Updated CSS version to v=4 for cache refresh
- Updated loading.js version to v=3

## Technical Details

### Horizontal Radar Sweep Structure
```
- Full-width gradient band (100% viewport width)
- Sweeps from left (-100%) to right (100%)
- Gradient opacity: 0% → 5% → 10% → 15% → 10% → 5% → 0%
- Blue color: rgba(19, 55, 111, X) where X is the opacity
- Background overlay: 30% white for subtle dimming
- Animation: 2s ease-in-out infinite
```

### Animation Keyframes
```css
@keyframes radar-sweep {
    0% { left: -100%; }
    100% { left: 100%; }
}
```

### Cleanup Mechanism
- Automatically removes sweep after loading completes
- Fallback removal after 10 seconds
- MutationObserver monitors for loading state changes

## Visual Effect
The loading animation now appears as:
- A translucent blue band that sweeps horizontally across the entire screen
- The band has varying opacity creating a gradient effect
- Matches the blue theme color (#13376f) used in About boxes
- Subtle white overlay dims the background slightly
- Continuous sweeping motion from left to right

## Testing
To verify the fix:
1. Clear browser cache (Ctrl+F5 or Cmd+Shift+R)
2. Navigate to Explore tab and click on any sunburst segment
3. Navigate to Sources tab > Named Entities subtab and change entity type filter
4. You should see a blue horizontal band sweeping across the screen during loading

## Browser Compatibility
- Chrome/Edge: Full support
- Firefox: Full support
- Safari: Full support
- IE11: Graceful degradation (no animation)