#!/usr/bin/env python
# coding: utf-8

"""
Named Entity Frequency Analysis Generator for the Russian-Ukrainian War database.
This script extracts all named entities from document chunks, calculates their frequencies,
and exports the results to a CSV file.
"""

import os
import sys
import logging
import csv
import json
from datetime import datetime
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

def extract_named_entity_frequencies():
    """
    Extract named entities from document_section_chunk table, calculate frequencies, 
    and save to CSV.
    
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

        # Since the named_entities column contains JSON strings, we need to process it with Python
        # First, get all named entities
        query = """
        SELECT named_entities
        FROM document_section_chunk
        WHERE named_entities IS NOT NULL
        """
        
        logger.info("Fetching named entities data...")
        rows = []
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
        
        logger.info(f"Processing {len(rows)} rows with named entities...")
        
        # Dictionary to store entity frequencies
        entity_counts = {}
        total_count = 0
        
        # Process each row
        for i, row in enumerate(rows):
            if i % 10000 == 0 and i > 0:
                logger.info(f"Processed {i} rows so far...")
            
            try:
                # Parse the JSON string
                ne_string = row[0]
                entities = json.loads(ne_string)
                
                # Process each entity in the array
                for entity in entities:
                    entity_text = entity.get("text", "").strip()
                    entity_type = entity.get("label", "UNKNOWN")
                    
                    if entity_text and entity_type:
                        # Create a composite key for the entity
                        key = (entity_type, entity_text)
                        
                        # Increment the count
                        if key in entity_counts:
                            entity_counts[key] += 1
                        else:
                            entity_counts[key] = 1
                        
                        total_count += 1
            
            except (json.JSONDecodeError, AttributeError, TypeError) as e:
                if i < 10:  # Only log the first few errors to avoid flooding
                    logger.warning(f"Error processing row: {e} - Data: {row[0][:100]}")
                continue
        
        logger.info(f"Extracted {len(entity_counts)} unique entity-type combinations from {total_count} total entities")
        
        # Prepare output file
        output_filename = f"named_entity_frequencies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Convert to a list and sort by frequency
        entity_items = [(entity_type, entity_text, count) 
                         for (entity_type, entity_text), count in entity_counts.items()]
        entity_items.sort(key=lambda x: (x[0], -x[2]))  # Sort by entity type, then count desc
        
        # Calculate relative frequencies
        entity_items_with_freq = []
        for entity_type, entity_text, count in entity_items:
            relative_freq = (count / total_count) * 100
            entity_items_with_freq.append((entity_type, entity_text, count, relative_freq))
        
        # Save results to CSV
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['Entity Type', 'Entity Value', 'Count', 'Relative Frequency (%)'])
            # Write data rows
            for entity_type, entity_text, count, relative_freq in entity_items_with_freq:
                writer.writerow([entity_type, entity_text, count, round(relative_freq, 2)])
                
        logger.info(f"Successfully saved named entity frequencies to {output_filename}")
        
        # Print statistics by entity type
        entity_types = {}
        for entity_type, entity_text, count, relative_freq in entity_items_with_freq:
            if entity_type not in entity_types:
                entity_types[entity_type] = []
            entity_types[entity_type].append((entity_text, count, relative_freq))
        
        logger.info("\nEntity type statistics:")
        for entity_type, entities in entity_types.items():
            type_count = sum(count for _, count, _ in entities)
            type_percentage = (type_count / total_count) * 100
            logger.info(f"{entity_type}: {type_count} occurrences ({round(type_percentage, 2)}%)")
            
            # Print top entities for each type
            top_n = min(10, len(entities))
            logger.info(f"\nTop {top_n} {entity_type} entities:")
            for i, (entity_text, count, relative_freq) in enumerate(entities[:top_n], 1):
                logger.info(f"{i}. {entity_text}: {count} occurrences ({round(relative_freq, 2)}%)")
            
            logger.info("")  # Empty line for readability
                
        return True
        
    except Exception as e:
        logger.error(f"Error extracting named entity frequencies: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """
    Main function to run the named entity frequency extraction.
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
    
    # Run the extraction
    if extract_named_entity_frequencies():
        logger.info("Named entity frequency extraction completed successfully.")
    else:
        logger.error("Named entity frequency extraction failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()