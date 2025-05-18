#!/usr/bin/env python
# coding: utf-8

"""
Named Entity Inspection Script for the Russian-Ukrainian War database.
This script examines the structure of named entities in the database.
"""

import os
import sys
import logging
import json
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration - copied from config.py
DB_CONFIG = {
    'user': 'postgres',
    'password': os.environ.get('DB_PASSWORD', 'GoNKJWp64NkMr9UdgCnT'),
    'host': os.environ.get('DB_HOST', '138.201.62.161'),
    'port': os.environ.get('DB_PORT', '5434'),
    'database': os.environ.get('DB_NAME', 'russian_ukrainian_war'),
}

def create_connection_string():
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

def inspect_named_entities():
    """
    Inspect the structure of named_entities data in the document_section_chunk table.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create connection string and engine
        connection_string = create_connection_string()
        engine = create_engine(connection_string)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("Database connection established successfully")
        
        # Query to get some sample named_entities data
        query = """
        SELECT 
            id, 
            named_entities
        FROM 
            document_section_chunk
        WHERE 
            named_entities IS NOT NULL 
            AND jsonb_typeof(named_entities) = 'object'
            AND jsonb_array_length(CASE 
                WHEN jsonb_typeof(named_entities) = 'object' 
                THEN COALESCE(
                    (SELECT jsonb_agg(value) FROM jsonb_each(named_entities)),
                    '[]'::jsonb
                )
                ELSE '[]'::jsonb
            END) > 0
        LIMIT 10;
        """
        
        # Execute query and fetch results
        logger.info("Fetching sample named_entities data...")
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
        
        # Log the structure of each named_entities
        logger.info(f"Found {len(rows)} sample rows with named_entities.")
        
        if len(rows) == 0:
            # If the query returns no rows, try a simpler query
            logger.info("No matching rows found with the complex query. Trying a simpler query...")
            
            simpler_query = """
            SELECT 
                id, 
                named_entities
            FROM 
                document_section_chunk
            WHERE 
                named_entities IS NOT NULL
            LIMIT 10;
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(simpler_query))
                rows = result.fetchall()
                
            logger.info(f"Found {len(rows)} sample rows with named_entities using the simpler query.")
            
        for i, row in enumerate(rows, 1):
            row_id = row[0]
            ne_data = row[1]
            
            logger.info(f"Row {i} (ID: {row_id}):")
            logger.info(f"JSONB type: {type(ne_data)}")
            logger.info(f"Named Entities: {json.dumps(ne_data, indent=2)}")
            
            # Try to get the keys (entity types)
            if isinstance(ne_data, dict):
                logger.info(f"Entity Types: {list(ne_data.keys())}")
                
                # Show structure of each entity type
                for entity_type, entities in ne_data.items():
                    logger.info(f"  {entity_type} type: {type(entities)}")
                    
                    if isinstance(entities, list):
                        logger.info(f"  {entity_type} count: {len(entities)}")
                        if entities:
                            logger.info(f"  {entity_type} sample: {entities[:3]}")
                    else:
                        logger.info(f"  {entity_type} value: {entities}")
            else:
                logger.info(f"Not a dictionary: {ne_data}")
            
            logger.info("---")
        
        # Get count of rows with named_entities
        count_query = """
        SELECT COUNT(*) 
        FROM document_section_chunk
        WHERE named_entities IS NOT NULL;
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(count_query))
            total_count = result.scalar()
            
        logger.info(f"Total number of rows with named_entities: {total_count}")
        
        # Count entity types
        type_query = """
        SELECT 
            jsonb_object_keys(named_entities) as entity_type,
            COUNT(*) as count
        FROM 
            document_section_chunk
        WHERE 
            named_entities IS NOT NULL AND
            jsonb_typeof(named_entities) = 'object'
        GROUP BY 
            entity_type
        ORDER BY 
            count DESC;
        """
        
        logger.info("Counting entity types...")
        try:
            with engine.connect() as conn:
                result = conn.execute(text(type_query))
                type_counts = result.fetchall()
                
            logger.info(f"Found {len(type_counts)} different entity types.")
            for entity_type, count in type_counts:
                logger.info(f"  {entity_type}: {count}")
                
        except Exception as e:
            logger.warning(f"Could not count entity types: {e}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error inspecting named entities: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """
    Main function to run the named entity inspection.
    """
    # Check if required packages are installed
    try:
        import sqlalchemy
        logger.info("SQLAlchemy is installed.")
    except ImportError:
        logger.error("SQLAlchemy is not installed. Please install it with: pip install sqlalchemy")
        sys.exit(1)
    
    try:
        import psycopg2
        logger.info("psycopg2 is installed.")
    except ImportError:
        logger.error("psycopg2 is not installed. Please install it with: pip install psycopg2-binary")
        sys.exit(1)
    
    # Run the inspection
    if inspect_named_entities():
        logger.info("Named entity inspection completed successfully.")
    else:
        logger.error("Named entity inspection failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()