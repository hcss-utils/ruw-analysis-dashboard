# Burstiness Tab Horizontal Layout Fix

## Problem
The collapsible sections (Historical Events, Standard Filters, Data Type Filters) in the Burstiness tab were displayed vertically instead of horizontally as requested.

## Root Cause
Each collapsible section was wrapped in its own `dbc.Row` with a single `dbc.Col` spanning the full width (`width=12`), causing them to stack vertically.

## Solution Implemented

### 1. Layout Changes in `tabs/burstiness.py`
- Combined all three collapsible sections into a single `dbc.Row`
- Set each section to `width=4` so they display side-by-side (3 columns in a row)
- Added `w-100` class to buttons to make them full width within their columns
- Added responsive width settings (`width=12, lg=4`) for better mobile behavior
- Kept Visualization Options in a separate row since it needs more space

### 2. CSS Enhancements in `static/custom.css`
Added specific styling for the Burstiness tab:
- Button styling to ensure full width within columns
- Card height limits with scrolling to prevent overflow
- Spacing between collapsible sections
- Responsive behavior for smaller screens (stacks vertically on mobile)
- Hover effects for burst model cards

## Key Changes

### Before:
```python
dbc.Row([
    dbc.Col([
        # Historical Events content
    ], width=12),
]),
dbc.Row([
    dbc.Col([
        # Standard Filters content
    ], width=12),
]),
dbc.Row([
    dbc.Col([
        # Data Type Filters content
    ], width=12),
]),
```

### After:
```python
dbc.Row([
    dbc.Col([
        # Historical Events content
    ], width=4),
    dbc.Col([
        # Standard Filters content
    ], width=4),
    dbc.Col([
        # Data Type Filters content
    ], width=4),
], className="mb-3"),
```

## Result
The three main filter sections now display horizontally in a single row on desktop screens, while still maintaining responsive behavior for smaller screens where they stack vertically.

## Files Modified
1. `/mnt/c/Apps/ruw-analyze - refactor - 250209/tabs/burstiness.py` - Layout structure changes
2. `/mnt/c/Apps/ruw-analyze - refactor - 250209/static/custom.css` - Added Burstiness tab specific styling