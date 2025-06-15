#!/usr/bin/env python
# coding: utf-8

"""
Database connection module for the Russian-Ukrainian War Data Analysis Dashboard.
Provides functions to create and manage database connections.
"""

import logging
from typing import Optional, Dict, Any, Union
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.pool import QueuePool
from urllib.parse import quote_plus
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import re

# Add the project root to the path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DB_CONFIG

# Flag to track if we're in demo mode
DEMO_MODE = False


def create_search_indexes():
    """
    Create GIN indexes for full-text search to improve boolean search performance.
    This is an optional optimization that dramatically speeds up boolean searches.
    Run this once to improve search performance across all future searches.
    """
    logging.info("Creating full-text search indexes for optimal boolean search performance...")
    
    try:
        engine = get_engine()
        
        # Use regular connection for index creation
        conn = engine.connect()
        
        try:
            # Create GIN index on content column (most important)
            logging.info("Creating GIN index on document_section_chunk.content (this may take 10-20 minutes)...")
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_gin_content_fts 
                ON document_section_chunk 
                USING gin(to_tsvector('english', content))
            """))
            
            # Create GIN index on reasoning column  
            logging.info("Creating GIN index on taxonomy.chunk_level_reasoning...")
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_gin_reasoning_fts 
                ON taxonomy 
                USING gin(to_tsvector('english', chunk_level_reasoning))
            """))
            
            conn.commit()
            logging.info("✅ Full-text search indexes created successfully!")
            logging.info("Boolean searches will now be much faster (5-10x speed improvement)")
            
        finally:
            conn.close()
        
    except Exception as e:
        logging.error(f"Failed to create search indexes: {e}")
        logging.info("Boolean search will still work but may be slower without indexes")
        logging.info("Consider running this optimization during off-peak hours")


def create_connection_string() -> Optional[str]:
    """
    Create a connection string for the database from configuration.
    Supports both local configuration and Heroku DATABASE_URL.
    
    Returns:
        Optional[str]: Database connection string or None if using demo mode
    """
    global DEMO_MODE  # Declare at the beginning of the function
    
    # Check if running on Heroku (with DATABASE_URL environment variable)
    database_url = os.environ.get('DATABASE_URL')
    
    # Check if database_url is a placeholder or invalid value
    if database_url:
        # Check for placeholder values that indicate demo mode should be used
        if database_url in ['your_existing_database_url', 'placeholder', 'demo']:
            logging.info("Using demo mode with sample data (placeholder DATABASE_URL detected)")
            DEMO_MODE = True
            return None
            
        # Heroku PostgreSQL connection strings start with postgres://, but SQLAlchemy needs postgresql://
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        
        # Check if it's a valid connection string format
        if not (database_url.startswith('postgresql://') or 
                database_url.startswith('sqlite://') or
                re.match(r'postgresql(\+[a-zA-Z0-9]+)?://', database_url)):
            logging.warning(f"Invalid DATABASE_URL format, switching to demo mode: {database_url}")
            DEMO_MODE = True
            return None
            
        logging.info("Using DATABASE_URL from environment variables")
        return database_url
    
    # Otherwise use local configuration
    try:
        user = DB_CONFIG['user']
        password = DB_CONFIG['password']
        host = DB_CONFIG['host']
        port = DB_CONFIG['port']
        database = DB_CONFIG['database']
        
        # Check if any required field is None
        if not all([user, password, host, port, database]):
            logging.error("Missing required database configuration")
            logging.info("Falling back to demo mode with sample data")
            DEMO_MODE = True
            return None
        
        # Quote the password to handle special characters
        password_quoted = quote_plus(password)
        
        logging.info(f"Using local database configuration for {host}:{port}/{database}")
        return f"postgresql+psycopg2://{user}:{password_quoted}@{host}:{port}/{database}"
    except Exception as e:
        logging.error(f"Error creating connection string from DB_CONFIG: {e}")
        logging.info("Falling back to demo mode with sample data")
        DEMO_MODE = True
        return None


def create_db_engine() -> Optional[Engine]:
    """
    Create database engine with connection pooling.
    If database connection fails, switches to demo mode.
    
    Returns:
        Optional[Engine]: SQLAlchemy engine object or None if using demo mode
    """
    global DEMO_MODE  # Declare at the beginning of the function
    
    connection_string = create_connection_string()
    
    # If already in demo mode or connection string is None, return None
    if DEMO_MODE or connection_string is None:
        DEMO_MODE = True
        logging.info("Using demo mode with sample data")
        return None
        
    try:
        # Add timeout for Heroku - increase to 120 seconds for complex queries
        connect_args = {}
        if 'DYNO' in os.environ:
            connect_args = {'connect_timeout': 120, 'options': '-c statement_timeout=120000'}
        
        engine = create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=DB_CONFIG['pool_size'],
            max_overflow=DB_CONFIG['max_overflow'],
            pool_timeout=DB_CONFIG['pool_timeout'],
            pool_recycle=DB_CONFIG['pool_recycle'],
            connect_args=connect_args
        )
        
        # Test the connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            
        logging.info("Database connection established successfully")
        return engine
    
    except Exception as e:
        logging.error(f"Error connecting to database: {e}")
        logging.info("Falling back to demo mode with sample data")
        DEMO_MODE = True
        return None


# Create a global engine instance
engine = create_db_engine()


def get_engine() -> Union[Engine, None]:
    """
    Get the global engine instance.
    
    Returns:
        Union[Engine, None]: SQLAlchemy engine object or None if in demo mode
    """
    return engine


def is_demo_mode() -> bool:
    """
    Check if the application is running in demo mode.
    
    Returns:
        bool: True if in demo mode, False otherwise
    """
    return DEMO_MODE


def dispose_engine() -> None:
    """
    Dispose of the current engine instance.
    Should be called during application shutdown.
    """
    global engine
    if engine:
        engine.dispose()
        logging.info("Database engine disposed")