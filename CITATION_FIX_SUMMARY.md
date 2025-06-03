# Citation Formatting Fix Summary

## Issues Fixed

### 1. Duplicate Language Display
- **Problem**: Language was appearing twice - once in the old metadata lines and once in the citation
- **Solution**: Removed the old metadata lines (metadata_line1, metadata_line2, metadata_line3) that were no longer being used in the layout

### 2. Font Size Mismatch
- **Problem**: Citations were using 13px font while content was larger
- **Solution**: Changed citation font size from 13px to 16px to match content
- **Additional**: Also updated all reasoning text to use 16px font size with proper margins

### 3. Citation Element Reordering
- **Problem**: Citation elements were not in the requested order
- **Solution**: Reordered to: Author, Source, Section (with position info), Date, Database, Language, Document, Text status, Keywords, Entities

## Changes Made in utils/helpers.py

1. **Removed duplicate metadata display** (lines 116-134 and 167-170):
   - Deleted metadata_line1 showing Document, Database, Language, Date
   - Deleted metadata_line2 showing Section, Author, Full text status
   - Deleted metadata_line3 showing Keywords and Entities
   - These were redundant as all info is now in the citation

2. **Updated citation ordering** (line 183-195):
   - Reordered citation_parts array to match requested sequence
   - Added position info directly to the Section field

3. **Fixed font sizes**:
   - Citation: Changed from 13px to 16px (line 199)
   - Chunk text: Added explicit 16px font size (line 216)
   - Reasoning headers: Added 16px font size (line 207)
   - Reasoning bullet points: Added 16px font size with 20px left margin (line 211)
   - Other reasoning text: Added 16px font size (line 217)
   - "No reasoning available" text: Added 16px font size (line 220)

## Result
Citations now display in a single, clean format with:
- No duplicate information
- Consistent 16px font size throughout
- Proper element ordering as requested
- Position info integrated into the Section field