"""
Helper functions for working with keywords_llm JSONB column
"""

def get_keywords_llm_extraction_sql(language_preference='en'):
    """
    Returns SQL fragment to extract keywords from keywords_llm JSONB array.
    
    Args:
        language_preference: 'en', 'ru', or 'lemma' (default)
    
    Returns:
        SQL string to extract keywords from JSONB
    """
    if language_preference == 'en':
        return """
        SELECT DISTINCT 
            COALESCE(
                elem->>'lemma',
                elem->'translations'->>'en',
                elem->'translations'->>'ru'
            ) as keyword
        FROM jsonb_array_elements(dsc.keywords_llm) as elem
        WHERE elem->>'lemma' IS NOT NULL OR elem->'translations' IS NOT NULL
        """
    elif language_preference == 'ru':
        return """
        SELECT DISTINCT 
            COALESCE(
                elem->'translations'->>'ru',
                elem->>'lemma',
                elem->'translations'->>'en'
            ) as keyword
        FROM jsonb_array_elements(dsc.keywords_llm) as elem
        WHERE elem->>'lemma' IS NOT NULL OR elem->'translations' IS NOT NULL
        """
    else:  # default to lemma
        return """
        SELECT DISTINCT 
            elem->>'lemma' as keyword
        FROM jsonb_array_elements(dsc.keywords_llm) as elem
        WHERE elem->>'lemma' IS NOT NULL
        """

def get_keywords_llm_with_scores_sql():
    """
    Returns SQL to extract keywords with their scores and ranks.
    """
    return """
    SELECT 
        elem->>'lemma' as keyword,
        (elem->>'score')::float as score,
        (elem->>'rank')::int as rank
    FROM jsonb_array_elements(dsc.keywords_llm) as elem
    WHERE elem->>'lemma' IS NOT NULL
    ORDER BY (elem->>'rank')::int
    """

def get_keywords_llm_filter_sql():
    """
    Returns SQL WHERE clause to filter chunks with keywords_llm.
    """
    return """
    dsc.keywords_llm IS NOT NULL 
    AND jsonb_typeof(dsc.keywords_llm) = 'array' 
    AND jsonb_array_length(dsc.keywords_llm) > 0
    """

def convert_keywords_array_to_llm_format(keywords_array):
    """
    Convert old keywords array format to keywords_llm format for compatibility.
    
    Args:
        keywords_array: List of keyword strings
        
    Returns:
        List of dicts in keywords_llm format
    """
    if not keywords_array:
        return []
    
    return [
        {
            "lemma": keyword,
            "score": 0.8,  # Default score
            "rank": idx + 1,
            "translations": {
                "en": keyword if not any(ord(c) > 127 for c in keyword) else "",
                "ru": keyword if any(ord(c) > 127 for c in keyword) else ""
            }
        }
        for idx, keyword in enumerate(keywords_array)
    ]