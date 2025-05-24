# Search Tab Sunburst Chart Fix

The search tab sunburst chart functionality has been fixed to match the explore tab's behavior. The following changes were made:

1. Added a fixed-position "Back to Text Chunks" button in the search tab with the same styling as the explore tab.
2. Enhanced the sunburst chart to be properly clickable by setting the clickmode and adding marker properties.
3. Added a proper anchor div for the "Back to Text Chunks" button to scroll to.
4. Updated the search results pagination callback to:
   - Filter results based on the selected taxonomic element when a sunburst segment is clicked
   - Add a header showing the selected element and count information (similar to the explore tab)
   - Reset pagination to the first page when a new segment is clicked
   - Display detailed pagination information
5. Made the date range picker consistent between tabs by ensuring initial_visible_month is set.

## Implementation Notes

The implementation keeps the same behavior between the explore and search tabs:
- When a segment in the sunburst chart is clicked, the results are filtered to show only items from that taxonomic element
- A header is displayed showing the selected element name and count information
- The "Back to Text Chunks" button allows easy navigation back to the text chunks section

These changes provide a consistent user experience across the dashboard and make the search tab's sunburst chart as interactive and useful as the explore tab's chart.
