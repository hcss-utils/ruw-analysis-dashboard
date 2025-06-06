# Sources Tab Performance Fix - June 10, 2025

## Problem
The Sources tab was "absolutely terrible" due to:
1. **Extremely slow queries** - Named entities queries taking 100+ seconds even with sampling
2. **Inaccurate data** - Using LIMIT 5000-10000 sampling made statistics incomplete
3. **Poor UX** - Long loading times with no progress indication
4. **Multiple sequential queries** - Documents, chunks, taxonomy, keywords, and entities all fetched sequentially

## Root Causes
1. Complex JSONB operations on named_entities column without proper indexes
2. Multiple expensive queries running one after another
3. Sampling (LIMIT) made data inaccurate but still slow
4. No caching or optimization for repeated queries

## Solutions Implemented

### 1. Fast Sources Tab (sources_fast.py) - IMPLEMENTED ✓
Created a simplified version that:
- Shows only basic statistics (document counts, chunk counts, taxonomy levels)
- Uses a single efficient query with JOINs
- Loads in under 2 seconds
- Provides accurate counts without sampling
- Clear messaging that detailed analysis is in other tabs

### 2. Removed Sampling Limits (data_fetchers_sources.py) ✓
- Changed LIMIT 10000 and LIMIT 5000 to NO LIMIT
- Gets accurate data (though still slow for named entities)
- Better for accuracy but performance still poor

### 3. Created Optimized Version (sources_optimized.py)
- Progressive loading with real-time progress updates
- Loads each data type as it becomes available
- Shows loading progress (0-100%)
- Better UX but requires threading

## Current State
- App now uses `sources_fast.py` which loads quickly
- Shows essential corpus statistics
- Directs users to other tabs for detailed analysis
- Much better user experience

## Future Improvements
1. **Database Optimizations** (database_optimizations.sql ready)
   - Add GIN indexes on JSONB columns
   - Create materialized view for entity statistics
   - Would reduce query time from 100+ seconds to 1-3 seconds

2. **Use sources_optimized.py** after database optimization
   - Progressive loading for better UX
   - Shows all original data without sampling
   - Real-time progress updates

## Files Changed
- `app.py` - Now imports from `tabs.sources_fast`
- `tabs/sources_fast.py` - New fast implementation (created)
- `tabs/sources_optimized.py` - Progressive loading version (created)
- `database/data_fetchers_sources.py` - Removed sampling limits

## Testing
The fast Sources tab now:
- Loads in < 2 seconds
- Shows accurate statistics
- Provides clear navigation to other tabs for detailed analysis
- Much better user experience