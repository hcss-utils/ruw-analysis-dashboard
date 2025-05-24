# Search Functionality Enhancement

This enhancement addresses two issues with the search functionality in the dashboard:

## Issue 1: Mismatch in Search Logic
The first issue was a mismatch in search logic between:
1. fetch_search_category_data (used to populate the sunburst chart)
2. fetch_all_text_chunks_for_search (used to fetch the actual text chunks)

The first function was using PostgreSQL's text search vector functionality (to_tsvector and plainto_tsquery), while the second was using regex pattern matching with word boundaries. This inconsistency caused situations where chunks would appear in the sunburst chart categories but no actual text chunks would be retrieved.

### Fix:
- Changed fetch_all_text_chunks_for_search to use the same PostgreSQL text search functionality
- Made the same change for the semantic search fallback to ensure consistency

## Issue 2: Artificial Result Limitation
The second issue was an artificial limitation of search results:
- The code was hardcoding a limit of 500 chunks in the fetch_all_text_chunks_for_search function
- This prevented users from seeing all relevant text chunks for their search query

### Fix:
- Removed the hardcoded limit of 500 chunks
- Modified the function to accept an optional limit parameter (default: None)
- Updated the SQL query to only add a LIMIT clause when a limit is explicitly specified
- Changed the search tab to call the function without specifying a limit

## Benefits

- The sunburst chart and text chunks now use the same underlying search logic
- Clicking on segments in the search tab's sunburst chart now works correctly
- Users can now see ALL text chunks that match their search criteria, not just the first 500
- The search functionality in the Search tab now behaves identically to the Explore tab

This ensures a comprehensive and consistent user experience across the dashboard.

