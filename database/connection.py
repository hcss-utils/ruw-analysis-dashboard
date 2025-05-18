#!/usr/bin/env python
# coding: utf-8

"""
Database connection module for the Russian-Ukrainian War Data Analysis Dashboard.
Provides functions to create and manage database connections.
"""

import logging
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.pool import QueuePool
from urllib.parse import quote_plus
import sys
import os

# Add the project root to the path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DB_CONFIG


def create_connection_string() -> str:
    """
    Create a connection string for the database from configuration.
    
    Returns:
        str: Database connection string
    """
    user = DB_CONFIG['user']
    password = quote_plus(DB_CONFIG['password'])
    host = DB_CONFIG['host']
    port = DB_CONFIG['port']
    database = DB_CONFIG['database']
    
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"


def create_db_engine() -> Engine:
    """
    Create database engine with connection pooling.
    
    Returns:
        Engine: SQLAlchemy engine object
    
    Raises:
        Exception: If database connection fails
    """
    try:
        connection_string = create_connection_string()
        
        engine = create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=DB_CONFIG['pool_size'],
            max_overflow=DB_CONFIG['max_overflow'],
            pool_timeout=DB_CONFIG['pool_timeout'],
            pool_recycle=DB_CONFIG['pool_recycle']
        )
        
        # Test the connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            
        logging.info("Database connection established successfully")
        return engine
    
    except Exception as e:
        logging.error(f"Error connecting to database: {e}")
        raise Exception(f"Failed to connect to database: {e}")


# Create a global engine instance
engine = create_db_engine()


def get_engine() -> Engine:
    """
    Get the global engine instance.
    
    Returns:
        Engine: SQLAlchemy engine object
    """
    return engine


def dispose_engine() -> None:
    """
    Dispose of the current engine instance.
    Should be called during application shutdown.
    """
    global engine
    if engine:
        engine.dispose()
        logging.info("Database engine disposed")