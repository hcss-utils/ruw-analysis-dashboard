"""
Boolean search query parser for PostgreSQL full-text search.
"""

import re
import logging
from typing import Tuple


def parse_boolean_query(query: str) -> Tuple[str, bool]:
    """
    Parse a boolean search query and convert it to PostgreSQL tsquery format.
    
    Args:
        query: Boolean search query string
        
    Returns:
        Tuple of (parsed_query, is_valid)
    """
    if not query or not query.strip():
        return "", False
    
    try:
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # If query has no boolean operators, add AND between words
        if not re.search(r'\b(AND|OR|NOT)\b', query, re.IGNORECASE):
            # Split on spaces and join with AND
            words = query.split()
            if len(words) > 1:
                query = ' AND '.join(words)
                logging.info(f"Auto-converted '{' '.join(words)}' to '{query}'")
        
        # Handle case-insensitive boolean operators
        # Use word boundaries to avoid replacing parts of words
        query = re.sub(r'\bAND\b', '&', query, flags=re.IGNORECASE)
        query = re.sub(r'\bOR\b', '|', query, flags=re.IGNORECASE)
        query = re.sub(r'\bNOT\b', '!', query, flags=re.IGNORECASE)
        
        # Validate parentheses balance
        open_count = query.count('(')
        close_count = query.count(')')
        if open_count != close_count:
            logging.warning(f"Mismatched parentheses in query: {query}")
            return "", False
        
        # Basic validation - query shouldn't start or end with operators
        if re.match(r'^\s*[&|!]', query) or re.search(r'[&|!]\s*$', query):
            logging.warning(f"Query starts or ends with operator: {query}")
            return "", False
        
        # Check for consecutive operators
        if re.search(r'[&|!]\s*[&|!]', query):
            logging.warning(f"Consecutive operators in query: {query}")
            return "", False
        
        # Remove any remaining extra spaces around operators
        query = re.sub(r'\s*([&|!])\s*', r' \1 ', query)
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query, True
        
    except Exception as e:
        logging.error(f"Error parsing boolean query '{query}': {e}")
        return "", False


def validate_boolean_syntax(query: str) -> Tuple[bool, str]:
    """
    Validate boolean search syntax and provide user-friendly error messages.
    
    Args:
        query: Boolean search query string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query or not query.strip():
        return False, "Query cannot be empty"
    
    # Check for balanced parentheses
    if query.count('(') != query.count(')'):
        return False, "Mismatched parentheses - make sure each '(' has a matching ')'"
    
    # Check for operators at start/end
    if re.match(r'^\s*(AND|OR|NOT)\b', query, re.IGNORECASE):
        return False, "Query cannot start with AND, OR, or NOT"
    
    if re.search(r'\b(AND|OR|NOT)\s*$', query, re.IGNORECASE):
        return False, "Query cannot end with AND, OR, or NOT"
    
    # Check for consecutive operators
    if re.search(r'\b(AND|OR|NOT)\s+(AND|OR|NOT)\b', query, re.IGNORECASE):
        return False, "Cannot have consecutive operators (AND, OR, NOT)"
    
    # Check for empty parentheses
    if re.search(r'\(\s*\)', query):
        return False, "Empty parentheses are not allowed"
    
    # Check for operators immediately after opening parenthesis
    if re.search(r'\(\s*(AND|OR|NOT)\b', query, re.IGNORECASE):
        return False, "Cannot start a parenthetical group with AND, OR, or NOT"
    
    return True, ""


def get_boolean_search_help() -> str:
    """
    Get help text for boolean search syntax.
    
    Returns:
        Help text string
    """
    return """Boolean Search Examples:
• war AND Ukraine (both terms must appear)
• Russia OR "Russian Federation" (either term can appear)
• military NOT peace (first term but not second)
• (Ukraine OR Russia) AND sanctions (grouping with parentheses)
• "exact phrase" AND keyword (combine exact phrases with keywords)

Tips:
• Use quotes for exact phrases
• Use parentheses to group terms
• Operators must be uppercase: AND, OR, NOT
• Cannot start or end with operators"""