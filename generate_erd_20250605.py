#!/usr/bin/env python
# coding: utf-8

"""
Enhanced Entity-Relationship Diagram (ERD) Generator for the Russian-Ukrainian War database.
This script generates a comprehensive visual ERD with all tables, columns, and relationships.
Generated on: 2025-06-05
"""

import os
import sys
import logging
from datetime import datetime
from urllib.parse import quote_plus
from sqlalchemy import create_engine, inspect, MetaData, text
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'user': 'postgres',
    'password': os.environ.get('DB_PASSWORD', 'GoNKJWp64NkMr9UdgCnT'),
    'host': os.environ.get('DB_HOST', '138.201.62.161'),
    'port': os.environ.get('DB_PORT', '5434'),
    'database': os.environ.get('DB_NAME', 'russian_ukrainian_war'),
}

def create_connection_string():
    """Create a connection string for the database."""
    user = DB_CONFIG['user']
    password = quote_plus(DB_CONFIG['password'])
    host = DB_CONFIG['host']
    port = DB_CONFIG['port']
    database = DB_CONFIG['database']
    
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

def get_database_schema(engine):
    """Get complete database schema including tables, columns, and relationships."""
    inspector = inspect(engine)
    schema = {}
    
    # Get all tables
    tables = inspector.get_table_names()
    
    for table_name in tables:
        table_info = {
            'columns': [],
            'primary_keys': [],
            'foreign_keys': [],
            'indexes': [],
            'row_count': 0
        }
        
        # Get columns
        columns = inspector.get_columns(table_name)
        for col in columns:
            col_info = {
                'name': col['name'],
                'type': str(col['type']),
                'nullable': col.get('nullable', True),
                'default': col.get('default'),
                'primary_key': False
            }
            table_info['columns'].append(col_info)
        
        # Get primary keys
        pk_info = inspector.get_pk_constraint(table_name)
        if pk_info:
            table_info['primary_keys'] = pk_info.get('constrained_columns', [])
            # Mark primary key columns
            for col in table_info['columns']:
                if col['name'] in table_info['primary_keys']:
                    col['primary_key'] = True
        
        # Get foreign keys
        fk_info = inspector.get_foreign_keys(table_name)
        for fk in fk_info:
            fk_data = {
                'name': fk.get('name'),
                'columns': fk.get('constrained_columns', []),
                'ref_table': fk.get('referred_table'),
                'ref_columns': fk.get('referred_columns', [])
            }
            table_info['foreign_keys'].append(fk_data)
        
        # Get indexes
        index_info = inspector.get_indexes(table_name)
        for idx in index_info:
            idx_data = {
                'name': idx.get('name'),
                'columns': idx.get('column_names', []),
                'unique': idx.get('unique', False)
            }
            table_info['indexes'].append(idx_data)
        
        # Get row count
        try:
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
                table_info['row_count'] = result
        except:
            table_info['row_count'] = 0
        
        schema[table_name] = table_info
    
    return schema

def create_visual_erd(schema, output_file):
    """Create a visual ERD using matplotlib."""
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(24, 18))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Table positions (manually arranged for clarity)
    table_positions = {
        'uploaded_document': (20, 80),
        'document_section': (20, 50),
        'document_section_chunk': (20, 20),
        'taxonomy': (50, 20),
        'classifications': (50, 50),
        'logs': (80, 80),
        'alembic_version': (80, 60),
        'taxonomy_granular': (80, 40),
        'taxonomy_clusters_kmeans': (80, 20)
    }
    
    # Color scheme
    colors = {
        'primary': '#1f77b4',
        'foreign': '#ff7f0e',
        'regular': '#2ca02c',
        'nullable': '#d62728',
        'table_bg': '#f0f0f0',
        'table_border': '#333333'
    }
    
    # Draw tables
    table_boxes = {}
    for table_name, table_info in schema.items():
        if table_name not in table_positions:
            continue
            
        x, y = table_positions[table_name]
        
        # Calculate table height based on columns
        num_columns = len(table_info['columns'])
        table_height = 3 + (num_columns * 0.8)
        table_width = 15
        
        # Draw table box
        table_box = FancyBboxPatch(
            (x - table_width/2, y - table_height/2),
            table_width, table_height,
            boxstyle="round,pad=0.1",
            facecolor=colors['table_bg'],
            edgecolor=colors['table_border'],
            linewidth=2
        )
        ax.add_patch(table_box)
        table_boxes[table_name] = (x, y, table_width, table_height)
        
        # Draw table name
        ax.text(x, y + table_height/2 - 0.5, table_name,
                ha='center', va='top', fontsize=12, fontweight='bold')
        
        # Draw row count
        row_count = table_info['row_count']
        ax.text(x, y + table_height/2 - 1.2, f"({row_count:,} rows)",
                ha='center', va='top', fontsize=9, style='italic', color='gray')
        
        # Draw columns
        col_y = y + table_height/2 - 2
        for col in table_info['columns']:
            # Determine column color
            if col['primary_key']:
                col_color = colors['primary']
                prefix = "[PK] "
            elif any(col['name'] in fk['columns'] for fk in table_info['foreign_keys']):
                col_color = colors['foreign']
                prefix = "[FK] "
            elif col['nullable']:
                col_color = colors['nullable']
                prefix = "○ "
            else:
                col_color = colors['regular']
                prefix = "● "
            
            # Column text
            col_text = f"{prefix}{col['name']} : {col['type']}"
            ax.text(x - table_width/2 + 0.5, col_y, col_text,
                    ha='left', va='top', fontsize=8, color=col_color)
            col_y -= 0.8
    
    # Draw relationships
    for table_name, table_info in schema.items():
        if table_name not in table_positions:
            continue
            
        for fk in table_info['foreign_keys']:
            ref_table = fk['ref_table']
            if ref_table not in table_positions:
                continue
            
            # Get positions
            from_x, from_y, from_w, from_h = table_boxes[table_name]
            to_x, to_y, to_w, to_h = table_boxes[ref_table]
            
            # Draw arrow
            arrow = FancyArrowPatch(
                (from_x, from_y),
                (to_x, to_y),
                arrowstyle='-|>',
                connectionstyle='arc3,rad=0.3',
                mutation_scale=15,
                linewidth=1.5,
                color='blue',
                alpha=0.6
            )
            ax.add_patch(arrow)
            
            # Add relationship label
            mid_x = (from_x + to_x) / 2
            mid_y = (from_y + to_y) / 2
            label = f"{fk['columns'][0]} → {fk['ref_columns'][0]}"
            ax.text(mid_x, mid_y, label,
                    ha='center', va='center', fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Add title and legend
    ax.text(50, 95, 'Russian-Ukrainian War Database Schema',
            ha='center', va='top', fontsize=20, fontweight='bold')
    
    ax.text(50, 92, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            ha='center', va='top', fontsize=12, style='italic')
    
    # Create legend
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=colors['primary'], 
                          linewidth=2, label='Primary Key'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=colors['foreign'], 
                          linewidth=2, label='Foreign Key'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=colors['regular'], 
                          linewidth=2, label='Required Field'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=colors['nullable'], 
                          linewidth=2, label='Nullable Field')
    ]
    
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Add statistics
    total_tables = len(schema)
    total_columns = sum(len(t['columns']) for t in schema.values())
    total_relationships = sum(len(t['foreign_keys']) for t in schema.values())
    total_rows = sum(t['row_count'] for t in schema.values())
    
    stats_text = (f"Tables: {total_tables} | "
                 f"Columns: {total_columns} | "
                 f"Relationships: {total_relationships} | "
                 f"Total Rows: {total_rows:,}")
    
    ax.text(50, 5, stats_text,
            ha='center', va='top', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray'))
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"ERD saved to {output_file}")

def create_plantuml_erd(schema, output_file):
    """Create a PlantUML ERD file."""
    
    puml_content = ["@startuml", "!theme plain", "skinparam backgroundColor #FAFAFA"]
    puml_content.append("skinparam classAttributeIconSize 0")
    puml_content.append("skinparam classFontSize 14")
    puml_content.append("")
    
    # Define tables
    for table_name, table_info in schema.items():
        puml_content.append(f"entity {table_name} {{")
        
        # Primary keys
        for col in table_info['columns']:
            if col['primary_key']:
                puml_content.append(f"  * **{col['name']}** : {col['type']}")
        
        puml_content.append("  --")
        
        # Other columns
        for col in table_info['columns']:
            if not col['primary_key']:
                nullable = "NULL" if col['nullable'] else "NOT NULL"
                # Check if it's a foreign key
                is_fk = any(col['name'] in fk['columns'] for fk in table_info['foreign_keys'])
                prefix = "  # " if is_fk else "  "
                puml_content.append(f"{prefix}{col['name']} : {col['type']} <<{nullable}>>")
        
        puml_content.append(f"  --")
        puml_content.append(f"  Rows: {table_info['row_count']:,}")
        puml_content.append("}")
        puml_content.append("")
    
    # Define relationships
    for table_name, table_info in schema.items():
        for fk in table_info['foreign_keys']:
            puml_content.append(f"{table_name} }}o--|| {fk['ref_table']} : {fk['columns'][0]}")
    
    puml_content.append("")
    puml_content.append("@enduml")
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(puml_content))
    
    logger.info(f"PlantUML ERD saved to {output_file}")

def main():
    """Main function to generate ERD."""
    try:
        # Create database connection
        connection_string = create_connection_string()
        engine = create_engine(connection_string)
        
        logger.info("Connected to database successfully")
        
        # Get schema information
        logger.info("Fetching database schema...")
        schema = get_database_schema(engine)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create visual ERD
        visual_output = f"db_diagram_{timestamp}.png"
        create_visual_erd(schema, visual_output)
        
        # Create PlantUML ERD
        puml_output = f"db_erd_{timestamp}.puml"
        create_plantuml_erd(schema, puml_output)
        
        # Also create copies without timestamp for easy reference
        import shutil
        shutil.copy(visual_output, "db_diagram.png")
        shutil.copy(puml_output, "db_erd.puml")
        
        logger.info("ERD generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error generating ERD: {e}")
        raise
    finally:
        if 'engine' in locals():
            engine.dispose()

if __name__ == "__main__":
    main()