#!/usr/bin/env python
# coding: utf-8

"""
Final Entity-Relationship Diagram (ERD) Generator for the Russian-Ukrainian War database.
This script connects to the database and generates a comprehensive schema documentation.
"""

import os
import sys
import re  # Added missing re import
import logging
from datetime import datetime
from urllib.parse import quote_plus
from sqlalchemy import create_engine, inspect, MetaData, text, Table, Column

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

def get_table_names(engine):
    """
    Get list of all tables in the database.
    
    Args:
        engine: SQLAlchemy engine
        
    Returns:
        list: List of table names
    """
    try:
        with engine.connect() as conn:
            query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE' 
            ORDER BY table_name;
            """
            result = conn.execute(text(query))
            tables = [row[0] for row in result]
        return tables
    except Exception as e:
        logger.error(f"Error getting table names: {e}")
        return []

def get_table_row_counts(engine):
    """
    Get the row counts for all tables.
    
    Args:
        engine: SQLAlchemy engine
        
    Returns:
        dict: Dictionary mapping table names to row counts
    """
    try:
        tables = get_table_names(engine)
        row_counts = {}
        
        with engine.connect() as conn:
            for table_name in tables:
                try:
                    query = text(f"SELECT COUNT(*) FROM \"{table_name}\"")
                    result = conn.execute(query).scalar()
                    row_counts[table_name] = result
                except Exception as e:
                    logger.warning(f"Could not get row count for {table_name}: {e}")
                    row_counts[table_name] = 0
        
        return row_counts
    except Exception as e:
        logger.error(f"Error getting table row counts: {e}")
        return {}

def get_column_info(engine, table_name):
    """
    Get information about columns for a specific table using raw SQL to avoid type issues.
    
    Args:
        engine: SQLAlchemy engine
        table_name: Name of the table
        
    Returns:
        list: List of column information dictionaries
    """
    try:
        query = """
        SELECT 
            column_name, 
            data_type, 
            is_nullable,
            column_default
        FROM 
            information_schema.columns 
        WHERE 
            table_schema = 'public' 
            AND table_name = :table_name
        ORDER BY 
            ordinal_position;
        """
        
        columns = []
        with engine.connect() as conn:
            result = conn.execute(text(query), {"table_name": table_name})
            
            for row in result:
                column_name, data_type, is_nullable, default = row
                
                column_info = {
                    'name': column_name,
                    'type': data_type,
                    'nullable': is_nullable == 'YES',
                    'default': default
                }
                
                columns.append(column_info)
        
        return columns
    except Exception as e:
        logger.error(f"Error getting column info for {table_name}: {e}")
        return []

def get_primary_keys(engine, table_name):
    """
    Get primary key columns for a table.
    
    Args:
        engine: SQLAlchemy engine
        table_name: Name of the table
        
    Returns:
        list: List of primary key column names
    """
    try:
        query = """
        SELECT 
            kcu.column_name
        FROM 
            information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu 
              ON tc.constraint_name = kcu.constraint_name
              AND tc.table_schema = kcu.table_schema
        WHERE 
            tc.constraint_type = 'PRIMARY KEY'
            AND tc.table_schema = 'public'
            AND tc.table_name = :table_name
        ORDER BY 
            kcu.ordinal_position;
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(query), {"table_name": table_name})
            primary_keys = [row[0] for row in result]
        
        return primary_keys
    except Exception as e:
        logger.error(f"Error getting primary keys for {table_name}: {e}")
        return []

def get_foreign_keys(engine, table_name):
    """
    Get foreign key constraints for a table.
    
    Args:
        engine: SQLAlchemy engine
        table_name: Name of the table
        
    Returns:
        list: List of foreign key information dictionaries
    """
    try:
        query = """
        SELECT 
            tc.constraint_name,
            kcu.column_name,
            ccu.table_name AS referenced_table,
            ccu.column_name AS referenced_column
        FROM 
            information_schema.table_constraints AS tc 
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name
              AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
              ON ccu.constraint_name = tc.constraint_name
              AND ccu.table_schema = tc.table_schema
        WHERE 
            tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_schema = 'public'
            AND tc.table_name = :table_name
        ORDER BY 
            kcu.ordinal_position;
        """
        
        # Group by constraint name to handle multi-column foreign keys
        fk_constraints = {}
        
        with engine.connect() as conn:
            result = conn.execute(text(query), {"table_name": table_name})
            
            for row in result:
                constraint_name, column_name, referenced_table, referenced_column = row
                
                if constraint_name not in fk_constraints:
                    fk_constraints[constraint_name] = {
                        'name': constraint_name,
                        'columns': [],
                        'referred_table': referenced_table,
                        'referred_columns': []
                    }
                
                fk_constraints[constraint_name]['columns'].append(column_name)
                fk_constraints[constraint_name]['referred_columns'].append(referenced_column)
        
        return list(fk_constraints.values())
    except Exception as e:
        logger.error(f"Error getting foreign keys for {table_name}: {e}")
        return []

def get_indexes(engine, table_name):
    """
    Get indexes for a table.
    
    Args:
        engine: SQLAlchemy engine
        table_name: Name of the table
        
    Returns:
        list: List of index information dictionaries
    """
    try:
        # First get all indexes from pg_indexes
        indexes_query = """
        SELECT
            indexname,
            indexdef
        FROM
            pg_indexes
        WHERE
            schemaname = 'public'
            AND tablename = :table_name;
        """
        
        indexes = []
        
        with engine.connect() as conn:
            result = conn.execute(text(indexes_query), {"table_name": table_name})
            
            for row in result:
                index_name, index_def = row
                
                # Skip primary key indexes which are already covered in primary keys
                if "pkey" in index_name:
                    continue
                
                # Extract column names from index definition
                # This is a simplified approach
                columns = []
                unique = False
                
                if "UNIQUE INDEX" in index_def:
                    unique = True
                
                # Very simple parsing, may need improvement for complex indexes
                col_match = re.search(r'\((.*?)\)', index_def)
                if col_match:
                    col_str = col_match.group(1)
                    # Split by comma and strip quotes and whitespace
                    columns = [c.strip(' "\'') for c in col_str.split(',')]
                
                indexes.append({
                    'name': index_name,
                    'columns': columns,
                    'unique': unique
                })
        
        return indexes
    except Exception as e:
        logger.error(f"Error getting indexes for {table_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def get_table_details(engine):
    """
    Get detailed information about each table in the database.
    
    Args:
        engine: SQLAlchemy engine
        
    Returns:
        dict: Dictionary with table details
    """
    try:
        tables = get_table_names(engine)
        row_counts = get_table_row_counts(engine)
        
        tables_dict = {}
        
        for table_name in tables:
            tables_dict[table_name] = {
                'columns': get_column_info(engine, table_name),
                'primary_keys': get_primary_keys(engine, table_name),
                'foreign_keys': get_foreign_keys(engine, table_name),
                'indexes': get_indexes(engine, table_name),
                'row_count': row_counts.get(table_name, 0)
            }
        
        return tables_dict
    
    except Exception as e:
        logger.error(f"Error getting table details: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}

def generate_visual_erd(engine, connection_string):
    """
    Generate a visual ERD by creating intermediary SQL file then using eralchemy2.
    
    Args:
        engine: SQLAlchemy engine
        connection_string: Database connection string
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get table details
        tables_dict = get_table_details(engine)
        
        # Create a .er file for eralchemy2
        with open("db_schema.er", "w") as f:
            # Write table definitions
            for table_name, table_info in tables_dict.items():
                f.write(f"[{table_name}]\n")
                
                for col in table_info['columns']:
                    col_str = f"{col['name']} {col['type']}"
                    
                    if col['name'] in table_info['primary_keys']:
                        col_str += " PK"
                    
                    f.write(f"    {col_str}\n")
                
                f.write("\n")
            
            # Write relationships
            for table_name, table_info in tables_dict.items():
                for fk in table_info['foreign_keys']:
                    ref_table = fk['referred_table']
                    
                    # Using * for many side and 1 for one side (many-to-one)
                    f.write(f"{table_name} *--1 {ref_table}\n")
        
        logger.info("Created ERD schema file at db_schema.er")
        
        # Now use eralchemy2 to render the diagram
        try:
            from eralchemy2 import render_er
            render_er("db_schema.er", "db_diagram.png")
            logger.info("Visual ERD generated successfully at db_diagram.png")
            return True
        except ImportError:
            logger.error("eralchemy2 package not installed. Install with: pip install eralchemy2")
            return False
        
    except Exception as e:
        logger.error(f"Error generating visual ERD: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def generate_textual_erd(tables_dict):
    """
    Generate a textual representation of the database schema.
    
    Args:
        tables_dict: Dictionary with table details
        
    Returns:
        str: Textual representation of the database schema
    """
    output = []
    
    # Generate header
    output.append("# Database Schema: russian_ukrainian_war")
    output.append("# Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    output.append("\n")
    
    # Generate table of contents
    output.append("## Table of Contents")
    for table_name in sorted(tables_dict.keys()):
        output.append(f"- [{table_name}](#{table_name.lower()})")
    
    output.append("\n" + "---" + "\n")
    
    # Calculate total statistics
    total_columns = sum(len(table_info['columns']) for table_info in tables_dict.values())
    total_fks = sum(len(table_info['foreign_keys']) for table_info in tables_dict.values())
    total_rows = sum(table_info['row_count'] for table_info in tables_dict.values())
    total_indexes = sum(len(table_info['indexes']) for table_info in tables_dict.values())
    
    # Database summary
    output.append("## Database Summary")
    output.append(f"- Tables: {len(tables_dict)}")
    output.append(f"- Total Columns: {total_columns}")
    output.append(f"- Total Foreign Keys: {total_fks}")
    output.append(f"- Total Indexes: {total_indexes}")
    output.append(f"- Total Rows: {total_rows:,}")
    output.append("\n" + "---" + "\n")
    
    # Generate table definitions
    for table_name, table_info in sorted(tables_dict.items()):
        output.append(f"## {table_name}")
        output.append(f"<a id=\"{table_name.lower()}\"></a>\n")
        
        output.append(f"**Row count**: {table_info['row_count']:,}\n")
        
        # Columns
        output.append("### Columns")
        
        # Create a table using markdown
        output.append("| Column | Type | Nullable | Default |")
        output.append("|--------|------|----------|---------|")
        
        for column in table_info['columns']:
            name = column['name']
            type_str = column['type']
            nullable = "YES" if column['nullable'] else "NO"
            default = column.get('default', '')
            
            # Mark primary keys
            if name in table_info['primary_keys']:
                name = f"**{name}** (PK)"
            
            # Mark foreign keys
            for fk in table_info['foreign_keys']:
                if name in fk['columns']:
                    name = f"{name} (FK)"
                    break
            
            output.append(f"| {name} | {type_str} | {nullable} | {default} |")
        
        # Primary Keys
        if table_info['primary_keys']:
            output.append("\n### Primary Key")
            pk_columns = ", ".join(table_info['primary_keys'])
            output.append(f"- **Columns**: {pk_columns}")
        
        # Foreign Keys
        if table_info['foreign_keys']:
            output.append("\n### Foreign Keys")
            for i, fk in enumerate(table_info['foreign_keys'], 1):
                fk_cols = ", ".join(fk['columns'])
                ref_cols = ", ".join(fk['referred_columns'])
                fk_name = fk.get('name', f"FK_{i}")
                
                output.append(f"- **{fk_name}**")
                output.append(f"  - **Columns**: {fk_cols}")
                output.append(f"  - **References**: {fk['referred_table']}({ref_cols})")
        
        # Indexes
        if table_info['indexes']:
            output.append("\n### Indexes")
            for idx in table_info['indexes']:
                unique = "UNIQUE " if idx['unique'] else ""
                cols = ", ".join(idx['columns'])
                output.append(f"- **{idx['name']}**: {unique}({cols})")
        
        # Add a separator between tables
        output.append("\n" + "---" + "\n")
    
    return "\n".join(output)

def generate_schema_stats(tables_dict):
    """
    Generate statistics about the database schema.
    
    Args:
        tables_dict: Dictionary with table details
        
    Returns:
        dict: Dictionary with schema statistics
    """
    # Initialize statistics
    stats = {
        'tables': len(tables_dict),
        'columns': 0,
        'primary_keys': 0,
        'foreign_keys': 0,
        'indexes': 0,
        'nullable_columns': 0,
        'total_rows': 0
    }
    
    # Calculate detailed statistics
    for table_name, table_info in tables_dict.items():
        stats['columns'] += len(table_info['columns'])
        stats['primary_keys'] += len(table_info['primary_keys'])
        stats['foreign_keys'] += len(table_info['foreign_keys'])
        stats['indexes'] += len(table_info['indexes'])
        stats['total_rows'] += table_info['row_count']
        
        # Count nullable columns
        for col in table_info['columns']:
            if col['nullable']:
                stats['nullable_columns'] += 1
    
    return stats

def generate_detailed_erd():
    """
    Generate a comprehensive ERD with detailed schema information.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create connection 
        connection_string = create_connection_string()
        engine = create_engine(connection_string)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("Database connection established successfully")
        
        # Get schema information directly from information_schema
        tables_dict = get_table_details(engine)
        
        # Generate textual ERD
        textual_erd = generate_textual_erd(tables_dict)
        with open("db_schema.md", "w") as f:
            f.write(textual_erd)
        logger.info("Textual ERD generated successfully at db_schema.md")
        
        # Save detailed schema information as JSON
        import json
        schema_data = {
            'tables': tables_dict,
            'statistics': generate_schema_stats(tables_dict)
        }
        
        with open("db_schema.json", "w") as f:
            json.dump(schema_data, f, indent=2, default=str)
        logger.info("Detailed schema information saved to db_schema.json")
        
        # Generate visual ERD
        visual_success = generate_visual_erd(engine, connection_string)
        
        # Print summary statistics
        stats = generate_schema_stats(tables_dict)
        logger.info(f"Schema Statistics:")
        logger.info(f"  Tables: {stats['tables']}")
        logger.info(f"  Total Columns: {stats['columns']}")
        logger.info(f"  Primary Keys: {stats['primary_keys']}")
        logger.info(f"  Foreign Keys: {stats['foreign_keys']}")
        logger.info(f"  Indexes: {stats['indexes']}")
        logger.info(f"  Total Rows: {stats['total_rows']:,}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error generating detailed ERD: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
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
    
    # Check for optional packages
    try:
        import eralchemy2
        logger.info("eralchemy2 is installed.")
    except ImportError:
        logger.warning("eralchemy2 is not installed. Visual ERD will not be generated.")
        logger.warning("Install eralchemy2 with: pip install eralchemy2")
    
    # Generate ERD
    if generate_detailed_erd():
        logger.info("ERD generation script completed successfully.")
    else:
        logger.error("ERD generation script failed.")