#!/usr/bin/env python
# coding: utf-8

"""
Relationship Analysis for the Russian-Ukrainian War database.
This script connects to the database and generates a detailed analysis of table relationships.
"""

import os
import sys
import logging
from datetime import datetime
from urllib.parse import quote_plus
from sqlalchemy import create_engine, inspect, MetaData, text
import pandas as pd
import json
import graphviz
from collections import defaultdict

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

def get_relationship_data(engine):
    """
    Extract relationship data from the database.
    
    Args:
        engine: SQLAlchemy engine
        
    Returns:
        tuple: (tables_data, relationships_data)
    """
    try:
        inspector = inspect(engine)
        
        tables_data = {}
        relationships = []
        
        # Get row counts for all tables
        row_counts = {}
        with engine.connect() as conn:
            for table_name in inspector.get_table_names():
                try:
                    query = text(f"SELECT COUNT(*) FROM {table_name}")
                    result = conn.execute(query).scalar()
                    row_counts[table_name] = result
                except Exception as e:
                    logger.warning(f"Could not get row count for {table_name}: {e}")
                    row_counts[table_name] = 0
        
        # Process each table
        for table_name in inspector.get_table_names():
            # Basic table info
            tables_data[table_name] = {
                'name': table_name,
                'row_count': row_counts.get(table_name, 0),
                'columns': [],
                'primary_keys': [],
                'foreign_keys': [],
                'incoming_references': [],
                'outgoing_references': 0
            }
            
            # Get column information
            for column in inspector.get_columns(table_name):
                col_info = {
                    'name': column['name'],
                    'type': str(column['type']),
                    'nullable': column['nullable']
                }
                
                tables_data[table_name]['columns'].append(col_info)
            
            # Get primary key information
            pk_constraint = inspector.get_pk_constraint(table_name)
            if pk_constraint and 'constrained_columns' in pk_constraint:
                tables_data[table_name]['primary_keys'] = pk_constraint['constrained_columns']
            
            # Get foreign key information
            for fk in inspector.get_foreign_keys(table_name):
                fk_info = {
                    'name': fk.get('name', ''),
                    'columns': fk['constrained_columns'],
                    'referred_table': fk['referred_table'],
                    'referred_columns': fk['referred_columns']
                }
                
                tables_data[table_name]['foreign_keys'].append(fk_info)
                tables_data[table_name]['outgoing_references'] += 1
                
                # Track the relationship
                relationship = {
                    'source_table': table_name,
                    'source_columns': fk['constrained_columns'],
                    'target_table': fk['referred_table'],
                    'target_columns': fk['referred_columns']
                }
                
                relationships.append(relationship)
        
        # Process incoming references
        for rel in relationships:
            source = rel['source_table']
            target = rel['target_table']
            
            if target in tables_data:
                tables_data[target]['incoming_references'].append({
                    'table': source,
                    'columns': rel['source_columns'],
                    'referenced_columns': rel['target_columns']
                })
        
        return tables_data, relationships
    
    except Exception as e:
        logger.error(f"Error extracting relationship data: {e}")
        return {}, []

def create_relationship_graph(tables_data, relationships):
    """
    Create a graphviz diagram of the database relationships.
    
    Args:
        tables_data: Dictionary with table data
        relationships: List of relationship dictionaries
        
    Returns:
        graphviz.Digraph: Graph object
    """
    dot = graphviz.Digraph(comment='Database Schema Relationships')
    dot.attr('graph', rankdir='LR', size='11,8', ratio='fill')
    dot.attr('node', shape='record', style='filled', fillcolor='lightblue')
    
    # Add tables as nodes
    for table_name, table_info in tables_data.items():
        # Color based on type (central tables vs. lookup tables)
        if len(table_info['incoming_references']) > 3:
            fillcolor = 'palegreen'  # Central tables that many other tables reference
        elif table_info['outgoing_references'] > 3:
            fillcolor = 'lightyellow'  # Tables with many foreign keys
        elif len(table_info['incoming_references']) == 0 and table_info['outgoing_references'] == 0:
            fillcolor = 'lightgray'  # Isolated tables
        else:
            fillcolor = 'lightblue'  # Normal tables
        
        # Create label with table name and primary keys
        label = f"{{<table>{table_name}|"
        
        # Add primary keys
        if table_info['primary_keys']:
            pk_section = "PK: " + ", ".join(table_info['primary_keys'])
            label += pk_section
        
        # Add row count
        label += f"|Rows: {table_info['row_count']:,}}}"
        
        dot.node(table_name, label=label, fillcolor=fillcolor)
    
    # Add relationships as edges
    for rel in relationships:
        source = rel['source_table']
        target = rel['target_table']
        source_cols = ", ".join(rel['source_columns'])
        target_cols = ", ".join(rel['target_columns'])
        
        if source in tables_data and target in tables_data:
            label = f"{source_cols} → {target_cols}"
            dot.edge(source, target, label=label)
    
    return dot

def generate_cardinality_analysis(tables_data, relationships):
    """
    Generate analysis of relationship cardinalities.
    
    Args:
        tables_data: Dictionary with table data
        relationships: List of relationship dictionaries
        
    Returns:
        dict: Dictionary with cardinality analysis
    """
    cardinality = {}
    
    # First pass - determine potential one-to-many relationships
    for table_name, table_info in tables_data.items():
        for incoming in table_info['incoming_references']:
            source_table = incoming['table']
            target_table = table_name
            
            # Check if the referenced column is a primary key
            is_pk_reference = all(col in table_info['primary_keys'] for col in incoming['referenced_columns'])
            
            relation_key = f"{source_table}_to_{target_table}"
            
            if is_pk_reference:
                # Could be many-to-one or one-to-one
                cardinality[relation_key] = {
                    'source': source_table,
                    'target': target_table,
                    'source_columns': incoming['columns'],
                    'target_columns': incoming['referenced_columns'],
                    'preliminary_type': 'M:1'  # Many-to-one is the default for FK to PK
                }
            else:
                # Non-PK reference - could be many-to-many or many-to-one
                cardinality[relation_key] = {
                    'source': source_table,
                    'target': target_table,
                    'source_columns': incoming['columns'],
                    'target_columns': incoming['referenced_columns'],
                    'preliminary_type': 'M:M'  # Assume many-to-many for non-PK reference
                }
    
    # Second pass - refine with uniqueness constraints
    unique_constraints = {}
    
    # Query database for unique constraints
    try:
        with engine.connect() as conn:
            query = """
            SELECT 
                tc.table_name, 
                tc.constraint_name,
                array_agg(kcu.column_name ORDER BY kcu.ordinal_position) as columns
            FROM 
                information_schema.table_constraints tc
            JOIN 
                information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
            WHERE 
                tc.constraint_type IN ('UNIQUE', 'PRIMARY KEY')
                AND tc.table_schema = 'public'
            GROUP BY 
                tc.table_name, tc.constraint_name;
            """
            
            result = conn.execute(text(query))
            
            for row in result:
                table_name, constraint_name, columns = row
                
                if table_name not in unique_constraints:
                    unique_constraints[table_name] = []
                
                unique_constraints[table_name].append(set(columns))
    
    except Exception as e:
        logger.warning(f"Could not get unique constraints: {e}")
    
    # Refine cardinality based on unique constraints
    for relation_key, relation_info in cardinality.items():
        source_table = relation_info['source']
        source_columns = set(relation_info['source_columns'])
        
        # Check if source columns are covered by a unique constraint
        has_unique_constraint = False
        
        if source_table in unique_constraints:
            for constraint_columns in unique_constraints[source_table]:
                if source_columns.issubset(constraint_columns):
                    has_unique_constraint = True
                    break
        
        # Update cardinality type if necessary
        if has_unique_constraint:
            if relation_info['preliminary_type'] == 'M:1':
                relation_info['type'] = '1:1'  # One-to-one relationship
            else:
                relation_info['type'] = '1:M'  # One-to-many relationship
        else:
            relation_info['type'] = relation_info['preliminary_type']  # Keep preliminary type
        
        # Remove preliminary type
        del relation_info['preliminary_type']
    
    return cardinality

def generate_relationship_report(tables_data, relationships, cardinality):
    """
    Generate a detailed relationship report in markdown format.
    
    Args:
        tables_data: Dictionary with table data
        relationships: List of relationship dictionaries
        cardinality: Dictionary with cardinality analysis
        
    Returns:
        str: Markdown formatted report
    """
    output = []
    
    # Generate header
    output.append("# Database Relationship Analysis Report")
    output.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("\n")
    
    # Summary statistics
    output.append("## Summary")
    output.append(f"- Total Tables: {len(tables_data)}")
    output.append(f"- Total Relationships: {len(relationships)}")
    
    # Count relationship types
    cardinality_counts = {'1:1': 0, '1:M': 0, 'M:1': 0, 'M:M': 0}
    for rel_info in cardinality.values():
        rel_type = rel_info['type']
        cardinality_counts[rel_type] = cardinality_counts.get(rel_type, 0) + 1
    
    output.append("- Relationship Types:")
    for rel_type, count in cardinality_counts.items():
        output.append(f"  - {rel_type}: {count}")
    
    # Table with most incoming and outgoing references
    incoming_refs = {name: len(info['incoming_references']) for name, info in tables_data.items()}
    outgoing_refs = {name: info['outgoing_references'] for name, info in tables_data.items()}
    
    if incoming_refs:
        most_referenced = max(incoming_refs.items(), key=lambda x: x[1])
        output.append(f"- Most Referenced Table: {most_referenced[0]} (referenced by {most_referenced[1]} tables)")
    
    if outgoing_refs:
        most_dependent = max(outgoing_refs.items(), key=lambda x: x[1])
        output.append(f"- Most Dependent Table: {most_dependent[0]} (references {most_dependent[1]} tables)")
    
    output.append("\n")
    
    # Tables by number of relationships
    output.append("## Tables by Relationship Count")
    
    # Create a table for incoming references
    output.append("\n### Tables by Incoming References")
    output.append("| Table | Incoming References | Row Count |")
    output.append("|-------|---------------------|-----------|")
    
    for table_name, ref_count in sorted(incoming_refs.items(), key=lambda x: x[1], reverse=True):
        row_count = tables_data[table_name]['row_count']
        output.append(f"| {table_name} | {ref_count} | {row_count:,} |")
    
    # Create a table for outgoing references
    output.append("\n### Tables by Outgoing References")
    output.append("| Table | Outgoing References | Row Count |")
    output.append("|-------|---------------------|-----------|")
    
    for table_name, ref_count in sorted(outgoing_refs.items(), key=lambda x: x[1], reverse=True):
        row_count = tables_data[table_name]['row_count']
        output.append(f"| {table_name} | {ref_count} | {row_count:,} |")
    
    output.append("\n")
    
    # Isolated tables (no relationships)
    isolated_tables = [name for name, info in tables_data.items() 
                      if not info['incoming_references'] and info['outgoing_references'] == 0]
    
    if isolated_tables:
        output.append("## Isolated Tables")
        output.append("These tables have no relationships with other tables.\n")
        
        output.append("| Table | Row Count |")
        output.append("|-------|-----------|")
        
        for table_name in sorted(isolated_tables):
            row_count = tables_data[table_name]['row_count']
            output.append(f"| {table_name} | {row_count:,} |")
        
        output.append("\n")
    
    # Relationship details
    output.append("## Relationship Details")
    
    # Group relationships by type
    rel_by_type = defaultdict(list)
    for rel_key, rel_info in cardinality.items():
        rel_by_type[rel_info['type']].append(rel_info)
    
    # One-to-One relationships
    if '1:1' in rel_by_type:
        output.append("\n### One-to-One Relationships")
        output.append("| Source Table | Source Columns | Target Table | Target Columns |")
        output.append("|--------------|----------------|--------------|----------------|")
        
        for rel in sorted(rel_by_type['1:1'], key=lambda x: (x['source'], x['target'])):
            source = rel['source']
            target = rel['target']
            source_cols = ", ".join(rel['source_columns'])
            target_cols = ", ".join(rel['target_columns'])
            
            output.append(f"| {source} | {source_cols} | {target} | {target_cols} |")
    
    # One-to-Many relationships
    if '1:M' in rel_by_type:
        output.append("\n### One-to-Many Relationships")
        output.append("| Source Table | Source Columns | Target Table | Target Columns |")
        output.append("|--------------|----------------|--------------|----------------|")
        
        for rel in sorted(rel_by_type['1:M'], key=lambda x: (x['source'], x['target'])):
            source = rel['source']
            target = rel['target']
            source_cols = ", ".join(rel['source_columns'])
            target_cols = ", ".join(rel['target_columns'])
            
            output.append(f"| {source} | {source_cols} | {target} | {target_cols} |")
    
    # Many-to-One relationships
    if 'M:1' in rel_by_type:
        output.append("\n### Many-to-One Relationships")
        output.append("| Source Table | Source Columns | Target Table | Target Columns |")
        output.append("|--------------|----------------|--------------|----------------|")
        
        for rel in sorted(rel_by_type['M:1'], key=lambda x: (x['source'], x['target'])):
            source = rel['source']
            target = rel['target']
            source_cols = ", ".join(rel['source_columns'])
            target_cols = ", ".join(rel['target_columns'])
            
            output.append(f"| {source} | {source_cols} | {target} | {target_cols} |")
    
    # Many-to-Many relationships
    if 'M:M' in rel_by_type:
        output.append("\n### Many-to-Many Relationships")
        output.append("| Source Table | Source Columns | Target Table | Target Columns |")
        output.append("|--------------|----------------|--------------|----------------|")
        
        for rel in sorted(rel_by_type['M:M'], key=lambda x: (x['source'], x['target'])):
            source = rel['source']
            target = rel['target']
            source_cols = ", ".join(rel['source_columns'])
            target_cols = ", ".join(rel['target_columns'])
            
            output.append(f"| {source} | {source_cols} | {target} | {target_cols} |")
    
    output.append("\n")
    
    # Detailed table relationships
    output.append("## Table Relationship Details")
    
    for table_name, table_info in sorted(tables_data.items()):
        output.append(f"\n### {table_name}")
        output.append(f"Row Count: {table_info['row_count']:,}")
        
        # Primary Keys
        if table_info['primary_keys']:
            output.append("\n**Primary Keys**: " + ", ".join(table_info['primary_keys']))
        
        # Outgoing References (Foreign Keys)
        if table_info['foreign_keys']:
            output.append("\n**Outgoing References (Foreign Keys):**")
            output.append("| Referenced Table | Foreign Key Columns | Referenced Columns |")
            output.append("|-----------------|---------------------|-------------------|")
            
            for fk in sorted(table_info['foreign_keys'], key=lambda x: x['referred_table']):
                ref_table = fk['referred_table']
                fk_cols = ", ".join(fk['columns'])
                ref_cols = ", ".join(fk['referred_columns'])
                
                output.append(f"| {ref_table} | {fk_cols} | {ref_cols} |")
        
        # Incoming References
        if table_info['incoming_references']:
            output.append("\n**Incoming References:**")
            output.append("| Referencing Table | Foreign Key Columns | Referenced Columns |")
            output.append("|-------------------|---------------------|-------------------|")
            
            for ref in sorted(table_info['incoming_references'], key=lambda x: x['table']):
                src_table = ref['table']
                fk_cols = ", ".join(ref['columns'])
                ref_cols = ", ".join(ref['referenced_columns'])
                
                output.append(f"| {src_table} | {fk_cols} | {ref_cols} |")
    
    return "\n".join(output)

def generate_relationship_analysis():
    """
    Generate a complete relationship analysis of the database.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        connection_string = create_connection_string()
        global engine  # Make it accessible to other functions
        engine = create_engine(connection_string)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("Database connection established successfully")
        
        # Extract relationship data
        tables_data, relationships = get_relationship_data(engine)
        
        if not tables_data or not relationships:
            logger.warning("No tables or relationships found")
            return False
        
        # Generate cardinality analysis
        cardinality = generate_cardinality_analysis(tables_data, relationships)
        
        # Create relationship graph
        dot = create_relationship_graph(tables_data, relationships)
        
        # Render graph to various formats
        dot.render('db_relationship_graph', format='png', cleanup=True)
        dot.render('db_relationship_graph', format='svg', cleanup=True)
        dot.render('db_relationship_graph', format='pdf', cleanup=True)
        
        logger.info("Relationship graph generated successfully")
        
        # Generate detailed relationship report
        report = generate_relationship_report(tables_data, relationships, cardinality)
        
        with open("db_relationship_analysis.md", "w") as f:
            f.write(report)
        
        logger.info("Relationship analysis report generated successfully at db_relationship_analysis.md")
        
        # Save relationship data as JSON
        relationship_data = {
            'tables': tables_data,
            'relationships': relationships,
            'cardinality': cardinality
        }
        
        with open("db_relationships.json", "w") as f:
            json.dump(relationship_data, f, indent=2)
        
        logger.info("Relationship data saved to db_relationships.json")
        
        return True
    
    except Exception as e:
        logger.error(f"Error generating relationship analysis: {e}")
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
    
    try:
        import graphviz
        logger.info("graphviz is installed.")
    except ImportError:
        logger.error("graphviz is not installed. Please install it with: pip install graphviz")
        logger.error("Also make sure the Graphviz software is installed on your system.")
        sys.exit(1)
    
    # Generate relationship analysis
    if generate_relationship_analysis():
        logger.info("Relationship analysis completed successfully.")
    else:
        logger.error("Relationship analysis failed.")