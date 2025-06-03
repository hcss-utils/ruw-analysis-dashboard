# Explore Tab Performance Fix

## Issue
The Explore tab was taking forever to load on initial page load because it was fetching ALL category data from the database before rendering.

## Root Cause
In `tabs/explore.py`, line 46 was calling `fetch_category_data()` without any filters on page load:
```python
df_init = fetch_category_data()  # This fetches ALL data!
```

This query could be very expensive on large databases, causing the slow initial load.

## Solution Applied

### 1. Removed Initial Data Fetch
Replaced the database query with an empty DataFrame and placeholder chart:
```python
# Create empty initial chart - data will be loaded via callback
df_init = pd.DataFrame(columns=['category', 'subcategory', 'sub_subcategory', 'count'])

# Create placeholder sunburst chart
fig_init = go.Figure()
fig_init.update_layout(
    title={'text': "Click 'Apply Filters' to load data", ...},
    annotations=[{'text': 'No filters applied yet', ...}]
)
```

### 2. Added prevent_initial_call
Added `prevent_initial_call=True` to the sunburst update callback to ensure it doesn't run on page load.

## Result
- The Explore tab now loads instantly
- Data is only fetched when the user clicks "Apply Filters"
- No unnecessary database queries on page load

## Additional Optimization Opportunities
The app.py file also fetches database list and date range on startup:
- `fetch_all_databases()` - Could be cached or lazy-loaded
- `fetch_date_range()` - Could be cached or lazy-loaded

These could be optimized in a future update if they're also causing slowness.