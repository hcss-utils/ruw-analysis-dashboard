#!/usr/bin/env python3
"""
Script to update all keyword references to use keywords_llm
This creates backup files and updates all necessary code
"""

import os
import re
import shutil
from datetime import datetime

# Files that need updating based on the search
FILES_TO_UPDATE = [
    'database/data_fetchers_sources.py',
    'database/data_fetchers_freshness.py',
    'database/data_fetchers_burst.py',
    'database/data_fetchers.py',
    'tabs/sources.py',
    'tabs/compare.py',
    'tabs/burstiness.py',
    'utils/helpers_concordance.py',
    'visualizations/bursts.py',
    'visualizations/co_occurrence.py'
]

# SQL patterns to replace
REPLACEMENTS = [
    # Array operations
    (r'unnest\(dsc\.keywords\)\s+as\s+keyword', 
     "(elem->>'lemma')::text as keyword FROM jsonb_array_elements(dsc.keywords_llm) as elem WHERE elem->>'lemma' IS NOT NULL) as keyword_extract(keyword"),
    
    # WHERE conditions
    (r'dsc\.keywords\s+IS\s+NOT\s+NULL\s+AND\s+array_length\(dsc\.keywords,\s*1\)\s*>\s*0',
     "dsc.keywords_llm IS NOT NULL AND jsonb_typeof(dsc.keywords_llm) = 'array' AND jsonb_array_length(dsc.keywords_llm) > 0"),
     
    # Simple keywords IS NOT NULL
    (r'dsc\.keywords\s+IS\s+NOT\s+NULL(?!\s+AND)',
     "dsc.keywords_llm IS NOT NULL"),
     
    # Direct column selection
    (r'dsc\.keywords(?!\s*\))',
     "dsc.keywords_llm"),
     
    # In SELECT statements
    (r'SELECT\s+keywords\s+FROM', 
     "SELECT keywords_llm FROM"),
     
    # Array length operations
    (r'array_length\(keywords,\s*1\)',
     "jsonb_array_length(keywords_llm)"),
]

def backup_file(filepath):
    """Create a backup of the file with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{filepath}.backup_{timestamp}"
    shutil.copy2(filepath, backup_path)
    print(f"Backed up {filepath} to {backup_path}")
    return backup_path

def update_file(filepath):
    """Update keyword references in a file"""
    print(f"\nProcessing {filepath}...")
    
    # Read the file
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Track changes
    changes_made = False
    original_content = content
    
    # Apply replacements
    for pattern, replacement in REPLACEMENTS:
        new_content = re.sub(pattern, replacement, content, flags=re.IGNORECASE | re.MULTILINE)
        if new_content != content:
            changes_made = True
            content = new_content
            print(f"  Applied: {pattern[:50]}...")
    
    # Special handling for complex patterns
    # Fix the jsonb_array_elements subquery issue
    content = re.sub(
        r'FROM\s+\(elem->',
        'FROM (SELECT elem->',
        content
    )
    
    # Handle the column type in fetch_text_chunks
    content = re.sub(
        r"'keywords':\s*chunk\['keywords'\]",
        "'keywords_llm': chunk['keywords_llm']",
        content
    )
    
    if changes_made:
        # Backup original
        backup_file(filepath)
        
        # Write updated content
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"  ✓ Updated {filepath}")
    else:
        print(f"  - No changes needed in {filepath}")
    
    return changes_made

def main():
    """Main update function"""
    print("Starting keywords to keywords_llm migration...")
    print("=" * 60)
    
    updated_files = []
    
    for file_path in FILES_TO_UPDATE:
        full_path = f"/mnt/c/Apps/ruw-analyze - refactor - 250209/{file_path}"
        if os.path.exists(full_path):
            if update_file(full_path):
                updated_files.append(file_path)
        else:
            print(f"Warning: {full_path} not found")
    
    print("\n" + "=" * 60)
    print(f"Migration complete! Updated {len(updated_files)} files:")
    for f in updated_files:
        print(f"  - {f}")
    
    print("\nNext steps:")
    print("1. Test the application to ensure keywords_llm is working")
    print("2. Update any visualization functions that process keyword data")
    print("3. Consider adding indexes: CREATE INDEX idx_keywords_llm_gin ON document_section_chunk USING GIN (keywords_llm);")

if __name__ == "__main__":
    main()