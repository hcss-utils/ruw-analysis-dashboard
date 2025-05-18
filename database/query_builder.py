#!/usr/bin/env python
# coding: utf-8

"""
SQL Query builder for the dashboard.
Provides functions to build complex queries with proper escaping and parameter handling.
"""

import logging
import re
from typing import Dict, List, Any, Tuple, Optional, Union

from sqlalchemy import text


class QueryBuilder:
    """
    Builder class for SQL queries with parameter binding.
    """
    
    def __init__(self):
        """Initialize an empty query."""
        self.query_parts = []
        self.params = {}
        self.order_by = None
        self.limit = None
        self.offset = None
    
    def add_select(self, select_clause: str) -> 'QueryBuilder':
        """
        Add SELECT clause to the query.
        
        Args:
            select_clause: SELECT clause text
            
        Returns:
            QueryBuilder: Self for chaining
        """
        self.query_parts.append(f"SELECT {select_clause}")
        return self
    
    def add_from(self, from_clause: str) -> 'QueryBuilder':
        """
        Add FROM clause to the query.
        
        Args:
            from_clause: FROM clause text
            
        Returns:
            QueryBuilder: Self for chaining
        """
        self.query_parts.append(f"FROM {from_clause}")
        return self
    
    def add_join(self, join_clause: str) -> 'QueryBuilder':
        """
        Add JOIN clause to the query.
        
        Args:
            join_clause: JOIN clause text
            
        Returns:
            QueryBuilder: Self for chaining
        """
        self.query_parts.append(join_clause)
        return self
    
    def add_where(self, where_clause: str) -> 'QueryBuilder':
        """
        Add or append to WHERE clause.
        
        Args:
            where_clause: WHERE condition text
            
        Returns:
            QueryBuilder: Self for chaining
        """
        where_index = next((i for i, part in enumerate(self.query_parts) 
                           if part.startswith("WHERE")), None)
        
        if where_index is None:
            self.query_parts.append(f"WHERE {where_clause}")
        else:
            self.query_parts[where_index] += f" AND {where_clause}"
        
        return self
    
    def add_where_if(self, condition: bool, where_clause: str) -> 'QueryBuilder':
        """
        Add WHERE clause conditionally.
        
        Args:
            condition: Boolean condition
            where_clause: WHERE condition text to add if condition is True
            
        Returns:
            QueryBuilder: Self for chaining
        """
        if condition:
            self.add_where(where_clause)
        return self
    
    def add_parameter(self, name: str, value: Any) -> 'QueryBuilder':
        """
        Add a parameter to be bound to the query.
        
        Args:
            name: Parameter name
            value: Parameter value
            
        Returns:
            QueryBuilder: Self for chaining
        """
        self.params[name] = value
        return self
    
    def set_order_by(self, order_clause: str) -> 'QueryBuilder':
        """
        Set ORDER BY clause.
        
        Args:
            order_clause: ORDER BY clause text
            
        Returns:
            QueryBuilder: Self for chaining
        """
        self.order_by = f"ORDER BY {order_clause}"
        return self
    
    def set_limit(self, limit: int) -> 'QueryBuilder':
        """
        Set LIMIT clause.
        
        Args:
            limit: Maximum number of rows to return
            
        Returns:
            QueryBuilder: Self for chaining
        """
        self.limit = f"LIMIT {limit}"
        return self
    
    def set_offset(self, offset: int) -> 'QueryBuilder':
        """
        Set OFFSET clause.
        
        Args:
            offset: Number of rows to skip
            
        Returns:
            QueryBuilder: Self for chaining
        """
        self.offset = f"OFFSET {offset}"
        return self
    
    def build(self) -> Tuple[str, Dict[str, Any]]:
        """
        Build the final query and parameters.
        
        Returns:
            Tuple[str, Dict[str, Any]]: Query string and parameters dict
        """
        query_str = " ".join(self.query_parts)
        
        # Add optional clauses
        if self.order_by:
            query_str += f" {self.order_by}"
        
        if self.limit:
            query_str += f" {self.limit}"
        
        if self.offset:
            query_str += f" {self.offset}"
        
        return query_str, self.params
    
    def build_text_query(self) -> Tuple[text, Dict[str, Any]]:
        """
        Build the final query as SQLAlchemy text object and parameters.
        
        Returns:
            Tuple[text, Dict[str, Any]]: SQLAlchemy text query and parameters dict
        """
        query_str, params = self.build()
        return text(query_str), params


def build_date_range_query(
    table_alias: str, 
    date_field: str, 
    date_range: Optional[Tuple[str, str]] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Build a date range condition for queries.
    
    Args:
        table_alias: Table alias
        date_field: Date field name
        date_range: Optional (start_date, end_date) tuple
        
    Returns:
        Tuple[str, Dict[str, Any]]: Condition string and parameters
    """
    if not date_range or not all(date_range):
        return "", {}
    
    start_date, end_date = date_range
    return (
        f"{table_alias}.{date_field} BETWEEN :start_date AND :end_date",
        {'start_date': start_date, 'end_date': end_date}
    )


def build_source_type_condition(source_type: Optional[str], source_types_map: Dict[str, str]) -> Tuple[str, Dict[str, Any]]:
    """
    Build a source type condition for queries.
    
    Args:
        source_type: Source type name
        source_types_map: Mapping of source types to SQL conditions
        
    Returns:
        Tuple[str, Dict[str, Any]]: Condition string and parameters
    """
    if not source_type or source_type not in source_types_map:
        return "", {}
    
    return source_types_map[source_type], {}