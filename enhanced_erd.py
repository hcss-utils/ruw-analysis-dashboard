#!/usr/bin/env python
# coding: utf-8

"""
Enhanced Entity-Relationship Diagram (ERD) Generator for the Russian-Ukrainian War database.
This script connects to the database and generates a comprehensive schema documentation
including table relationships, constraints, and detailed column information.
"""

import os
import sys
import logging
from datetime import datetime
from urllib.parse import quote_plus
from sqlalchemy import create_engine, inspect, MetaData, text, Table, Column
import pandas as pd
from collections import defaultdict
import json
import re

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

def get_table_comments(engine):
    """
    Get comments for all tables.
    
    Args:
        engine: SQLAlchemy engine
        
    Returns:
        dict: Dictionary mapping table names to comments
    """
    try:
        query = """
        SELECT 
            c.relname AS table_name,
            COALESCE(pg_catalog.obj_description(c.oid, 'pg_class'), '') AS comment
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE 
            c.relkind = 'r'
            AND n.nspname = 'public'
            AND pg_catalog.obj_description(c.oid, 'pg_class') IS NOT NULL;
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(query))
            comments = {row[0]: row[1] for row in result}
        
        return comments
    except Exception as e:
        logger.error(f"Error getting table comments: {e}")
        return {}

def get_column_comments(engine):
    """
    Get comments for all columns.
    
    Args:
        engine: SQLAlchemy engine
        
    Returns:
        dict: Nested dictionary mapping table names to column names to comments
    """
    try:
        query = """
        SELECT 
            c.relname AS table_name,
            a.attname AS column_name,
            COALESCE(pg_catalog.col_description(c.oid, a.attnum), '') AS comment
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid
        LEFT JOIN pg_catalog.pg_description d ON d.objoid = c.oid AND d.objsubid = a.attnum
        WHERE 
            c.relkind = 'r'
            AND n.nspname = 'public'
            AND a.attnum > 0
            AND NOT a.attisdropped
            AND pg_catalog.col_description(c.oid, a.attnum) IS NOT NULL;
        """
        
        comments = defaultdict(dict)
        
        with engine.connect() as conn:
            result = conn.execute(text(query))
            for row in result:
                table_name, column_name, comment = row
                comments[table_name][column_name] = comment
        
        return comments
    except Exception as e:
        logger.error(f"Error getting column comments: {e}")
        return {}

def get_table_row_counts(engine):
    """
    Get the row counts for all tables.
    
    Args:
        engine: SQLAlchemy engine
        
    Returns:
        dict: Dictionary mapping table names to row counts
    """
    try:
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        
        row_counts = {}
        
        with engine.connect() as conn:
            for table_name in table_names:
                query = text(f"SELECT COUNT(*) FROM {table_name}")
                result = conn.execute(query).scalar()
                row_counts[table_name] = result
        
        return row_counts
    except Exception as e:
        logger.error(f"Error getting table row counts: {e}")
        return {}

def get_sequences(engine):
    """
    Get all sequences in the database.
    
    Args:
        engine: SQLAlchemy engine
        
    Returns:
        list: List of sequence information dictionaries
    """
    try:
        query = """
        SELECT 
            sequence_name,
            start_value,
            increment_by,
            max_value,
            min_value,
            cycle_option
        FROM information_schema.sequences
        WHERE sequence_schema = 'public';
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(query))
            sequences = [dict(row._mapping) for row in result]
        
        return sequences
    except Exception as e:
        logger.error(f"Error getting sequences: {e}")
        return []

def get_views(engine):
    """
    Get all views in the database.
    
    Args:
        engine: SQLAlchemy engine
        
    Returns:
        dict: Dictionary mapping view names to view definitions
    """
    try:
        query = """
        SELECT 
            table_name AS view_name,
            view_definition
        FROM information_schema.views
        WHERE table_schema = 'public';
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(query))
            views = {row[0]: row[1] for row in result}
        
        return views
    except Exception as e:
        logger.error(f"Error getting views: {e}")
        return {}

def get_constraints(engine):
    """
    Get all constraints in the database.
    
    Args:
        engine: SQLAlchemy engine
        
    Returns:
        dict: Nested dictionary mapping table names to constraint names to constraint details
    """
    try:
        query = """
        SELECT 
            tc.table_name, 
            tc.constraint_name, 
            tc.constraint_type,
            kcu.column_name,
            CASE 
                WHEN tc.constraint_type = 'FOREIGN KEY' THEN ccu.table_name
                ELSE NULL
            END AS referenced_table,
            CASE 
                WHEN tc.constraint_type = 'FOREIGN KEY' THEN ccu.column_name
                ELSE NULL
            END AS referenced_column
        FROM 
            information_schema.table_constraints AS tc 
        JOIN 
            information_schema.key_column_usage AS kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        LEFT JOIN 
            information_schema.constraint_column_usage AS ccu
            ON ccu.constraint_name = tc.constraint_name
            AND ccu.table_schema = tc.table_schema
        WHERE 
            tc.table_schema = 'public';
        """
        
        constraints = defaultdict(lambda: defaultdict(list))
        
        with engine.connect() as conn:
            result = conn.execute(text(query))
            
            for row in result:
                table, constraint, type_, column, ref_table, ref_column = row
                
                # Format for easy lookup
                constraint_info = {
                    'column': column,
                    'type': type_
                }
                
                if ref_table and ref_column:
                    constraint_info['referenced_table'] = ref_table
                    constraint_info['referenced_column'] = ref_column
                
                constraints[table][constraint].append(constraint_info)
        
        return constraints
    except Exception as e:
        logger.error(f"Error getting constraints: {e}")
        return {}

def get_triggers(engine):
    """
    Get all triggers in the database.
    
    Args:
        engine: SQLAlchemy engine
        
    Returns:
        list: List of trigger information dictionaries
    """
    try:
        query = """
        SELECT 
            trigger_name,
            event_manipulation,
            event_object_table,
            action_statement,
            action_timing
        FROM information_schema.triggers
        WHERE trigger_schema = 'public';
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(query))
            triggers = [dict(row._mapping) for row in result]
        
        return triggers
    except Exception as e:
        logger.error(f"Error getting triggers: {e}")
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
        inspector = inspect(engine)
        metadata = MetaData()
        metadata.reflect(bind=engine)
        
        # Get additional schema info
        table_comments = get_table_comments(engine)
        column_comments = get_column_comments(engine)
        row_counts = get_table_row_counts(engine)
        constraints = get_constraints(engine)
        
        tables_dict = {}
        
        # Get all table names
        for table_name in inspector.get_table_names():
            tables_dict[table_name] = {
                'columns': [],
                'primary_keys': [],
                'foreign_keys': [],
                'indexes': [],
                'constraints': dict(constraints.get(table_name, {})),
                'comment': table_comments.get(table_name, ''),
                'row_count': row_counts.get(table_name, 0)
            }
            
            # Get column information
            for column in inspector.get_columns(table_name):
                col_info = {
                    'name': column['name'],
                    'type': str(column['type']),
                    'nullable': column['nullable'],
                    'comment': column_comments.get(table_name, {}).get(column['name'], '')
                }
                
                if column.get('default') is not None:
                    col_info['default'] = str(column['default'])
                
                tables_dict[table_name]['columns'].append(col_info)
            
            # Get primary key information
            pk_constraint = inspector.get_pk_constraint(table_name)
            if pk_constraint and 'constrained_columns' in pk_constraint:
                tables_dict[table_name]['primary_keys'] = pk_constraint['constrained_columns']
            
            # Get foreign key information
            for fk in inspector.get_foreign_keys(table_name):
                tables_dict[table_name]['foreign_keys'].append({
                    'name': fk.get('name', ''),
                    'columns': fk['constrained_columns'],
                    'referred_schema': fk.get('referred_schema'),
                    'referred_table': fk['referred_table'],
                    'referred_columns': fk['referred_columns'],
                    'options': fk.get('options', {})
                })
            
            # Get index information
            for index in inspector.get_indexes(table_name):
                tables_dict[table_name]['indexes'].append({
                    'name': index['name'],
                    'columns': index['column_names'],
                    'unique': index['unique']
                })
        
        return tables_dict
    
    except Exception as e:
        logger.error(f"Error getting table details: {e}")
        return {}

def visualize_relationships(tables_dict):
    """
    Generate a PlantUML entity-relationship diagram.
    
    Args:
        tables_dict: Dictionary with table details
        
    Returns:
        str: PlantUML code for ER diagram
    """
    plantuml = ["@startuml", "!theme plain", "skinparam linetype ortho", ""]
    
    # Add tables to diagram
    for table_name, table_info in tables_dict.items():
        plantuml.append(f"entity {table_name} {{")
        
        # Add primary key
        for pk in table_info['primary_keys']:
            for col in table_info['columns']:
                if col['name'] == pk:
                    nullable = "nullable" if col['nullable'] else "not null"
                    type_str = col['type']
                    plantuml.append(f"  * {pk} : {type_str} [{nullable}]")
        
        # Add other columns
        for col in table_info['columns']:
            if col['name'] not in table_info['primary_keys']:
                nullable = "nullable" if col['nullable'] else "not null"
                type_str = col['type']
                plantuml.append(f"  {col['name']} : {type_str} [{nullable}]")
        
        plantuml.append("}")
        plantuml.append("")
    
    # Add relationships
    for table_name, table_info in tables_dict.items():
        for fk in table_info['foreign_keys']:
            ref_table = fk['referred_table']
            # One-to-many relationship
            plantuml.append(f"{ref_table} ||--o{{ {table_name} : FK {'/'.join(fk['columns'])}")
    
    plantuml.append("@enduml")
    
    return "\n".join(plantuml)

def generate_textual_erd(tables_dict, sequences, views, triggers):
    """
    Generate a textual representation of the database schema in Markdown format.
    
    Args:
        tables_dict: Dictionary with table details
        sequences: List of sequence information
        views: Dictionary of views
        triggers: List of trigger information
        
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
    output.append("\n### Tables")
    for table_name in sorted(tables_dict.keys()):
        output.append(f"- [{table_name}](#{table_name.lower()})")
    
    if views:
        output.append("\n### Views")
        for view_name in sorted(views.keys()):
            output.append(f"- [{view_name}](#{view_name.lower()}-view)")
    
    if sequences:
        output.append("\n### Sequences")
        for seq in sequences:
            output.append(f"- [{seq['sequence_name']}](#{seq['sequence_name'].lower()}-sequence)")
    
    if triggers:
        output.append("\n### Triggers")
        for trigger in triggers:
            trigger_id = f"{trigger['trigger_name'].lower()}-trigger-on-{trigger['event_object_table'].lower()}"
            output.append(f"- [{trigger['trigger_name']} on {trigger['event_object_table']}](#{trigger_id})")
    
    output.append("\n" + "---" + "\n")
    
    # Generate table definitions
    output.append("# Tables")
    output.append("\n")
    
    for table_name, table_info in sorted(tables_dict.items()):
        output.append(f"## {table_name}")
        output.append(f"<a id=\"{table_name.lower()}\"></a>\n")
        
        # Add table comment if available
        if table_info['comment']:
            output.append(f"**Description**: {table_info['comment']}\n")
        
        output.append(f"**Row count**: {table_info['row_count']:,}\n")
        
        # Columns
        output.append("### Columns")
        
        # Create a table using markdown
        output.append("| Column | Type | Nullable | Default | Description |")
        output.append("|--------|------|----------|---------|-------------|")
        
        for column in table_info['columns']:
            name = column['name']
            type_str = column['type']
            nullable = "YES" if column['nullable'] else "NO"
            default = column.get('default', '')
            comment = column.get('comment', '')
            
            # Mark primary keys
            if name in table_info['primary_keys']:
                name = f"**{name}** (PK)"
            
            # Mark foreign keys
            for fk in table_info['foreign_keys']:
                if name in fk['columns']:
                    name = f"{name} (FK)"
                    break
            
            output.append(f"| {name} | {type_str} | {nullable} | {default} | {comment} |")
        
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
                
                if fk.get('options'):
                    output.append(f"  - **Options**: {fk['options']}")
        
        # Indexes
        if table_info['indexes']:
            output.append("\n### Indexes")
            for idx in table_info['indexes']:
                unique = "UNIQUE " if idx['unique'] else ""
                cols = ", ".join(idx['columns'])
                output.append(f"- **{idx['name']}**: {unique}({cols})")
        
        # Add a separator between tables
        output.append("\n" + "---" + "\n")
    
    # Add views information
    if views:
        output.append("# Views")
        output.append("\n")
        
        for view_name, view_def in sorted(views.items()):
            output.append(f"## {view_name}")
            output.append(f"<a id=\"{view_name.lower()}-view\"></a>\n")
            
            output.append("### Definition")
            # Format the SQL definition with proper indentation
            formatted_def = view_def.replace("\n", "\n    ")
            output.append(f"```sql\n{formatted_def}\n```")
            
            output.append("\n" + "---" + "\n")
    
    # Add sequences information
    if sequences:
        output.append("# Sequences")
        output.append("\n")
        
        for seq in sorted(sequences, key=lambda x: x['sequence_name']):
            output.append(f"## {seq['sequence_name']}")
            output.append(f"<a id=\"{seq['sequence_name'].lower()}-sequence\"></a>\n")
            
            output.append("| Property | Value |")
            output.append("|----------|-------|")
            output.append(f"| Start Value | {seq['start_value']} |")
            output.append(f"| Increment By | {seq['increment_by']} |")
            output.append(f"| Min Value | {seq['min_value']} |")
            output.append(f"| Max Value | {seq['max_value']} |")
            output.append(f"| Cycle | {seq['cycle_option']} |")
            
            output.append("\n" + "---" + "\n")
    
    # Add triggers information
    if triggers:
        output.append("# Triggers")
        output.append("\n")
        
        for trigger in sorted(triggers, key=lambda x: (x['event_object_table'], x['trigger_name'])):
            trigger_id = f"{trigger['trigger_name'].lower()}-trigger-on-{trigger['event_object_table'].lower()}"
            output.append(f"## {trigger['trigger_name']} on {trigger['event_object_table']}")
            output.append(f"<a id=\"{trigger_id}\"></a>\n")
            
            output.append("| Property | Value |")
            output.append("|----------|-------|")
            output.append(f"| Timing | {trigger['action_timing']} |")
            output.append(f"| Event | {trigger['event_manipulation']} |")
            
            output.append("\n### Action Statement")
            # Format the SQL statement with proper indentation
            formatted_stmt = trigger['action_statement'].replace("\n", "\n    ")
            output.append(f"```sql\n{formatted_stmt}\n```")
            
            output.append("\n" + "---" + "\n")
    
    return "\n".join(output)

def get_schema_statistics(tables_dict, sequences, views, triggers):
    """
    Generate statistics about the database schema.
    
    Args:
        tables_dict: Dictionary with table details
        sequences: List of sequence information
        views: Dictionary of views
        triggers: List of trigger information
        
    Returns:
        dict: Dictionary with schema statistics
    """
    # Initialize statistics
    stats = {
        'tables': len(tables_dict),
        'views': len(views),
        'sequences': len(sequences),
        'triggers': len(triggers),
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
        connection_string = create_connection_string()
        engine = create_engine(connection_string)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("Database connection established successfully")
        
        # Get schema information
        tables_dict = get_table_details(engine)
        sequences = get_sequences(engine)
        views = get_views(engine)
        triggers = get_triggers(engine)
        
        # Get schema statistics
        stats = get_schema_statistics(tables_dict, sequences, views, triggers)
        
        # Generate PlantUML ER diagram
        plantuml_code = visualize_relationships(tables_dict)
        with open("db_erd.puml", "w") as f:
            f.write(plantuml_code)
        logger.info("PlantUML ER diagram generated successfully at db_erd.puml")
        
        # Generate textual ERD
        textual_erd = generate_textual_erd(tables_dict, sequences, views, triggers)
        with open("db_schema.md", "w") as f:
            f.write(textual_erd)
        logger.info("Textual ERD generated successfully at db_schema.md")
        
        # Save detailed schema information as JSON
        schema_data = {
            'tables': tables_dict,
            'sequences': sequences,
            'views': views,
            'triggers': triggers,
            'statistics': stats
        }
        
        with open("db_schema.json", "w") as f:
            json.dump(schema_data, f, indent=2)
        logger.info("Detailed schema information saved to db_schema.json")
        
        # Generate visual ERD if eralchemy2 is installed
        try:
            from eralchemy2 import render_er
            render_er(connection_string, "db_diagram.png")
            logger.info("Visual ERD generated successfully at db_diagram.png")
        except ImportError:
            logger.warning("eralchemy2 package not installed. Visual ERD was not generated.")
        
        # Print summary statistics
        logger.info(f"Schema Statistics:")
        logger.info(f"  Tables: {stats['tables']}")
        logger.info(f"  Views: {stats['views']}")
        logger.info(f"  Sequences: {stats['sequences']}")
        logger.info(f"  Triggers: {stats['triggers']}")
        logger.info(f"  Total Columns: {stats['columns']}")
        logger.info(f"  Primary Keys: {stats['primary_keys']}")
        logger.info(f"  Foreign Keys: {stats['foreign_keys']}")
        logger.info(f"  Indexes: {stats['indexes']}")
        logger.info(f"  Total Rows: {stats['total_rows']:,}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error generating detailed ERD: {e}")
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