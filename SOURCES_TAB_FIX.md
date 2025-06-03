# Sources Tab Display Fix

## Issue
When clicking on the Sources tab, users would see the radar pulse loading animation but no content would appear after loading completed.

## Root Cause
1. The callback was expecting a `sources-result-stats` div that wasn't present in the layout
2. The callback was using `Input("sources-subtabs", "id")` which doesn't trigger reliably on initial load
3. The callback function signature didn't match the inputs after removing the tab_id trigger

## Solution Applied

### 1. Added Missing Element
Added the missing `sources-result-stats` div to the layout:
```python
# Add result stats div that the callback expects
html.Div(id="sources-result-stats", className="mb-3"),
```

### 2. Fixed Callback Trigger
Updated the callback to ensure it runs on initial load:
```python
@app.callback(
    [
        Output("sources-result-stats", "children"),
        Output("sources-subtabs", "children")
    ],
    [
        Input("sources-filter-button", "n_clicks")
    ],
    [
        State("sources-language-dropdown", "value"),
        State("sources-database-dropdown", "value"),
        State("sources-source-type-dropdown", "value"),
        State("sources-date-range-picker", "start_date"),
        State("sources-date-range-picker", "end_date")
    ],
    prevent_initial_call=False  # Ensure it runs on initial load
)
```

### 3. Fixed Function Signature
Updated the function to match the new inputs:
```python
def update_sources_tab(n_clicks, lang_val, db_val, source_type, start_date, end_date):
    # Removed tab_id parameter
```

## Result
The Sources tab now loads content properly on initial page load and displays all the expected data (Documents, Chunks, Taxonomy Combinations, Keywords, and Named Entities).

## Files Modified
- `tabs/sources.py` - Fixed layout and callback issues

## Deployment
✅ Committed to GitHub
✅ Deployed to Heroku (v20)