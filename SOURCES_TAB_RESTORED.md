# Sources Tab Restored - June 10, 2025

## What Was Fixed

The Sources tab now has ALL original visualizations restored:

### Visualizations per Subtab:

1. **Documents Tab**
   - Documents Overview stats card
   - Relevance donut chart (relevant vs irrelevant)
   - Documents Over Time line chart
   - Documents by Language bar chart
   - Documents by Database bar chart  
   - Language Time Series multi-line chart
   - Database Time Series multi-line chart
   - **Database Coverage Donut Charts** (multiple donuts showing coverage per database)

2. **Chunks Tab**
   - Chunks Overview stats card
   - Relevance donut chart
   - Chunks Over Time line chart
   - Chunks by Language bar chart
   - Chunks by Database bar chart
   - Language Time Series multi-line chart
   - Database Time Series multi-line chart
   - **Database Coverage Donut Charts**

3. **Taxonomy Tab**
   - Taxonomy stats card with distribution table
   - Taxonomy Combinations donut chart
   - Taxonomy Over Time line chart
   - Taxonomy by Language pie chart
   - Taxonomy by Database pie chart
   - Language Time Series multi-line chart
   - Database Time Series multi-line chart

4. **Keywords Tab**
   - Keywords Overview stats card
   - Top 15 Keywords horizontal bar chart
   - Keywords per Chunk Distribution bar chart
   - Keywords by Language bar chart
   - Keywords by Database bar chart
   - Keywords Over Time line chart
   - Language Time Series multi-line chart
   - Database Time Series multi-line chart
   - **Database Coverage Donut Charts**

5. **Named Entities Tab**
   - Named Entities Overview stats card
   - Top 15 Named Entities horizontal bar chart
   - Distribution by Entity Type bar chart
   - Entities per Chunk Distribution bar chart
   - Entities by Language bar chart
   - Entities by Database bar chart
   - Entities Over Time line chart
   - Language Time Series multi-line chart
   - Database Time Series multi-line chart
   - **Database Coverage Donut Charts**

## How to Run

```bash
cd "/mnt/c/Apps/ruw-analyze - refactor - 250209"
python app.py
```

## Performance Notes

- Restored from sources_backup.py which has the complete implementation
- Named entities queries still use sampling (50000 limit) for performance
- Consider running database_optimizations.sql for best performance
- All visualizations including database donut charts are now present

## Files Modified

- `tabs/sources.py` - Replaced with complete version from sources_backup.py
- `database/data_fetchers_sources.py` - Increased sampling limit to 50000 for better accuracy