# Keywords and Named Entities - Relevant Chunks Only Update

## Summary of Changes - January 6, 2025

### 1. Updated About Sections
Modified the About modal text in `tabs/sources.py` to clarify that keywords and named entities are extracted from RELEVANT chunks only (those with taxonomic classifications):

- **Keywords**: "NOTE: Keywords are extracted from RELEVANT chunks only (those with taxonomic classifications). This ensures focus on war-related content."
- **Named Entities**: "NOTE: Like keywords, named entities are extracted from RELEVANT chunks only (those with taxonomic classifications)."

### 2. Database Query Updates
Modified all SQL queries in `database/data_fetchers_sources.py` to include `INNER JOIN taxonomy t ON dsc.id = t.chunk_id` to filter for only relevant chunks:

#### Keywords Function Updates:
- `fetch_keywords_data()`:
  - Stats query - added taxonomy join
  - Top keywords query - added taxonomy join
  - Keywords per chunk distribution - added taxonomy join
  - Language distribution - added taxonomy join
  - Database distribution - added taxonomy join
  - Total chunks count - added taxonomy join

#### Named Entities Function Updates:
- `fetch_named_entities_data()`:
  - Stats query - added taxonomy join
  - Top entities query - added taxonomy join
  - Entity types distribution - added taxonomy join
  - Entities per chunk distribution - added taxonomy join
  - Language distribution - added taxonomy join
  - Database distribution - added taxonomy join
  - Total chunks count - added taxonomy join

#### Time Series Updates:
- `fetch_sources_time_series()`:
  - Keyword time series - added taxonomy join
  - Entity time series - added taxonomy join

- `fetch_language_time_series()`:
  - Top languages query for keywords - added taxonomy join
  - Top languages query for entities - added taxonomy join
  - Time series query for keywords - added taxonomy join
  - Time series query for entities - added taxonomy join

- `fetch_database_time_series()`:
  - Top databases query for keywords - added taxonomy join
  - Top databases query for entities - added taxonomy join
  - Time series query for keywords - added taxonomy join
  - Time series query for entities - added taxonomy join

### 3. Entity Type Filter Implementation
Added a new callback in `tabs/sources.py` to handle entity type filtering:
- `filter_by_entity_type()` callback that re-fetches all data when entity type is changed
- `filter_entities_by_type()` helper function to filter the named entities data

### Impact
- Keywords and named entities statistics now only reflect content from chunks that have taxonomic classifications
- This provides a more focused view on war-related content
- Coverage percentages are calculated against relevant chunks only
- All visualizations (bar charts, time series, distributions) now show data from relevant chunks only

### Technical Note
The `INNER JOIN taxonomy t ON dsc.id = t.chunk_id` ensures that only chunks with at least one taxonomic classification are included in the analysis. This filters out chunks that were deemed irrelevant to the war context.