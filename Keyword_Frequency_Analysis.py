#!/usr/bin/env python
# coding: utf-8

"""
Keyword Frequency Analysis Generator for the Russian-Ukrainian War database.
This script extracts all keywords from document chunks, calculates their frequencies,
and exports the results to a CSV file.
"""

import os
import sys
import logging
import csv
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

def extract_keyword_frequencies():
    """
    Extract keywords from document_section_chunk table, calculate frequencies, 
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
        
        # SQL query to extract and count all keywords
        # Modified to avoid using ROUND function which is causing errors
        query = """
        WITH keyword_counts AS (
          SELECT 
            keyword,
            COUNT(*) as count
          FROM 
            document_section_chunk,
            UNNEST(keywords) as keyword
          WHERE 
            keyword IS NOT NULL
          GROUP BY 
            keyword
        ),
        total AS (
          SELECT SUM(count) as total_count FROM keyword_counts
        )
        SELECT 
          keyword_counts.keyword as "Keyword",
          keyword_counts.count as "Count",
          (keyword_counts.count::float / total.total_count) * 100 as "RelativeFrequency"
        FROM 
          keyword_counts, total
        ORDER BY 
          keyword_counts.count DESC;
        """
        
        # Execute query and fetch results
        logger.info("Executing keyword frequency query...")
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
            
        # Prepare output file
        output_filename = f"keyword_frequencies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        logger.info(f"Found {len(rows)} unique keywords. Saving to {output_filename}")
        
        # Save results to CSV with rounding done in Python instead of SQL
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['Keyword', 'Count', 'Relative Frequency (%)'])
            # Write data rows with rounding in Python
            for row in rows:
                keyword = row[0]
                count = row[1]
                relative_freq = round(row[2], 2)  # Round to 2 decimal places in Python
                writer.writerow([keyword, count, relative_freq])
                
        logger.info(f"Successfully saved keyword frequencies to {output_filename}")
        
        # Print some stats
        if rows:
            top_keywords = rows[:10]
            logger.info("Top 10 keywords by frequency:")
            for i, row in enumerate(top_keywords, 1):
                keyword = row[0]
                count = row[1]
                relative_freq = round(row[2], 2)  # Round to 2 decimal places in Python
                logger.info(f"{i}. {keyword}: {count} occurrences ({relative_freq}%)")
                
        return True
        
    except Exception as e:
        logger.error(f"Error extracting keyword frequencies: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """
    Main function to run the keyword frequency extraction.
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
    if extract_keyword_frequencies():
        logger.info("Keyword frequency extraction completed successfully.")
    else:
        logger.error("Keyword frequency extraction failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()