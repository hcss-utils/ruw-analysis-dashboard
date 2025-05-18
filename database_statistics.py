#!/usr/bin/env python
# coding: utf-8

"""
Database Statistics Analysis for the Russian-Ukrainian War database.
This script connects to the database and generates detailed statistics about 
table sizes, column distributions, and data quality metrics.
"""

import os
import sys
import logging
from datetime import datetime
from urllib.parse import quote_plus
from sqlalchemy import create_engine, inspect, MetaData, text
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration - copied from config.py
DB_CONFIG = {
    'user': 'postgres',
    'password': os.environ.get('DB_PASSWORD', 'GoNKJWp64NkMr9UdgCnT'),
    'host': os