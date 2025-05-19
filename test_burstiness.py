#!/usr/bin/env python
# coding: utf-8

"""
Comprehensive test script for the burstiness functionality.
This script tests all components of the burstiness feature, including:
1. Burst detection algorithms
2. Data fetching and processing
3. Visualization creation
4. Timeline and network analysis
5. Historical event integration
"""

import unittest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import os
import sys
import warnings
import json
from unittest.mock import patch, MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add the project root to the path to import modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the modules to test
from utils.burst_detection import (
    kleinberg_burst_detection,
    kleinberg_multi_state_burst_detection,
    detect_burst_co_occurrences,
    validate_bursts_statistically,
    find_cascade_patterns,
    prepare_time_periods,
    normalize_burst_scores
)

from visualizations.bursts import (
    create_burst_heatmap,
    create_burst_summary_chart,
    create_burst_timeline,
    create_burst_comparison_chart,
    create_citespace_timeline,
    create_enhanced_citespace_timeline,
    create_co_occurrence_network,
    create_predictive_visualization,
    load_historical_events,
    prepare_document_links
)

from database.data_fetchers_freshness import (
    get_taxonomy_elements_for_burst,
    get_keywords_for_burst,
    get_named_entities_for_burst,
    get_burst_data_for_periods,
    calculate_burst_summaries
)

# Mock data for testing
def create_mock_time_series_data(n_periods=10, n_elements=5, with_bursts=True):
    """Create mock time series data for testing burst detection"""
    data = []
    
    # Create dates from today backwards
    end_date = datetime.now()
    dates = [end_date - timedelta(days=i) for i in range(n_periods * 7)]
    
    for element_id in range(n_elements):
        element_name = f"Element-{element_id}"
        
        for date in dates:
            # Base count
            count = np.random.randint(10, 50)
            
            # Add bursts for some elements at specific times
            if with_bursts and element_id < 3:  # Only first 3 elements have bursts
                # Add a burst in the middle of the time series
                if date >= dates[int(len(dates) * 0.4)] and date <= dates[int(len(dates) * 0.6)]:
                    count = count * 5  # Significant burst
            
            data.append({
                'date': date,
                'element': element_name,
                'count': count
            })
    
    return pd.DataFrame(data)

def create_mock_burst_data(n_periods=10, n_elements=5, n_data_types=3):
    """Create mock burst data dictionary for testing visualizations"""
    data_types = ['taxonomy', 'keywords', 'named_entities']
    burst_data = {}
    
    # Create period labels
    today = datetime.now()
    if n_periods <= 10:
        period_labels = [f"Week {i+1}" for i in range(n_periods)]
    else:
        period_labels = [(today - timedelta(days=7*i)).strftime('%b %d') for i in range(n_periods)]
    
    # Create data for each type
    for data_type_idx, data_type in enumerate(data_types[:n_data_types]):
        burst_data[data_type] = {}
        
        for element_id in range(n_elements):
            element_name = f"{data_type}-{element_id}"
            element_data = []
            
            for period_idx, period in enumerate(period_labels):
                # Base intensity - higher for earlier elements
                base_intensity = max(0, 80 - (element_id * 15))
                
                # Create some variation
                if period_idx > 0 and period_idx < len(period_labels) - 1:
                    variation = np.random.normal(0, 15)
                else:
                    variation = np.random.normal(0, 5)
                
                # Ensure intensity is within bounds
                intensity = min(100, max(0, base_intensity + variation))
                
                # Create a mock date for this period
                date = today - timedelta(days=7*period_idx)
                
                # Create row with all necessary columns
                element_data.append({
                    'date': date,
                    'period': period,
                    'element': element_name,
                    'count': int(100 * (intensity / 100)),
                    'burst_intensity': intensity,
                    'smoothed_count': int(100 * (intensity / 100)) + 0.5,
                    'prob': 0.01 * intensity,
                    'burst': 1 if intensity > 50 else 0,
                    'burst_score': intensity / 20
                })
            
            # Create DataFrame for this element
            burst_data[data_type][element_name] = pd.DataFrame(element_data)
    
    return burst_data

def create_mock_historical_events(n_events=5):
    """Create mock historical events data"""
    events = []
    base_date = datetime.now() - timedelta(days=90)
    
    for i in range(n_events):
        event_date = base_date + timedelta(days=i*15)
        events.append({
            "date": event_date.strftime("%Y-%m-%d"),
            "period": event_date.strftime("%b %Y"),
            "event": f"Test Event {i+1}",
            "impact": 1.0 - (i * 0.15),
            "description": f"Description for test event {i+1}"
        })
    
    return events

class TestBurstDetectionAlgorithms(unittest.TestCase):
    """Test suite for burst detection algorithms"""
    
    def setUp(self):
        """Set up test data"""
        self.time_series = create_mock_time_series_data()
        
    def test_kleinberg_basic(self):
        """Test basic Kleinberg burst detection algorithm"""
        # Test with default parameters
        result = kleinberg_burst_detection(self.time_series)
        
        # Check that result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check that required columns are added
        for col in ['smoothed_count', 'prob', 'burst', 'burst_score', 'burst_intensity']:
            self.assertIn(col, result.columns)
        
        # Check that burst intensity values are between 0 and 100
        self.assertTrue(all(0 <= val <= 100 for val in result['burst_intensity']))
        
        # Test with custom parameters
        result_custom = kleinberg_burst_detection(self.time_series, s=3.0, gamma=1.5)
        
        # Should have different burst scores with different parameters
        if len(result['burst_score']) > 0 and len(result_custom['burst_score']) > 0:
            if result['burst_score'].max() > 0 and result_custom['burst_score'].max() > 0:
                # This may not always be true due to random data, but it's a reasonable test
                self.assertNotEqual(result['burst_score'].sum(), result_custom['burst_score'].sum())
    
    def test_kleinberg_multi_state(self):
        """Test multi-state Kleinberg burst detection algorithm"""
        # Test with default parameters
        result = kleinberg_multi_state_burst_detection(self.time_series)
        
        # Check that result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check that required columns are added
        for col in ['smoothed_count', 'burst_state', 'burst_intensity']:
            self.assertIn(col, result.columns)
        
        # Check that burst intensity values are between 0 and 100
        self.assertTrue(all(0 <= val <= 100 for val in result['burst_intensity']))
        
        # Test with custom parameters
        result_custom = kleinberg_multi_state_burst_detection(
            self.time_series, num_states=4, gamma=1.5
        )
        
        # Check that burst states are within the expected range
        self.assertTrue(all(0 <= val < 4 for val in result_custom['burst_state']))
    
    def test_statistical_validation(self):
        """Test statistical validation of bursts"""
        # Test with default parameters
        result = validate_bursts_statistically(self.time_series)
        
        # Check that result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check that required columns are added
        for col in ['global_z_score', 'global_ci_lower', 'global_ci_upper', 'global_exceeds_ci',
                   'local_z_score', 'local_ci_lower', 'local_ci_upper', 'local_exceeds_ci',
                   'stat_burst_intensity', 'burst_probability', 'burst_significant']:
            self.assertIn(col, result.columns)
        
        # Test with custom parameters
        result_custom = validate_bursts_statistically(
            self.time_series, confidence_level=0.99, window_size=3
        )
        
        # Statistical burst intensity should be between 0 and 100
        self.assertTrue(all(0 <= val <= 100 for val in result_custom['stat_burst_intensity']))
    
    def test_co_occurrence_detection(self):
        """Test burst co-occurrence detection"""
        # Create mock burst data dictionary
        burst_data = create_mock_burst_data()
        
        # Test with default parameters
        co_occurrences = detect_burst_co_occurrences(burst_data)
        
        # Should be a dictionary
        self.assertIsInstance(co_occurrences, dict)
        
        # Test that co-occurrences have the right structure
        for key, value in co_occurrences.items():
            self.assertIsInstance(key, tuple)
            self.assertEqual(len(key), 2)  # Should be a pair of elements
            self.assertIn('strength', value)
            self.assertIn('common_periods', value)
            self.assertIn('num_common_periods', value)
    
    def test_cascade_patterns(self):
        """Test finding cascade patterns in bursts"""
        # Create mock burst data dictionary
        burst_data = create_mock_burst_data()
        
        # Test with default parameters
        cascades = find_cascade_patterns(burst_data)
        
        # Should be a list
        self.assertIsInstance(cascades, list)
        
        # Test cascade pattern structure if any found
        if cascades:
            cascade = cascades[0]
            for field in ['leader', 'follower', 'leader_period', 'follower_period', 
                         'leader_intensity', 'follower_intensity', 'lag']:
                self.assertIn(field, cascade)

    def test_time_period_preparation(self):
        """Test preparing time periods for analysis"""
        # Test for weeks
        boundaries, labels = prepare_time_periods('week', 5)
        self.assertEqual(len(boundaries), 5)
        self.assertEqual(len(labels), 5)
        self.assertIsInstance(boundaries[0], tuple)
        self.assertEqual(len(boundaries[0]), 2)  # Start and end dates
        
        # Test for months
        boundaries, labels = prepare_time_periods('month', 3)
        self.assertEqual(len(boundaries), 3)
        self.assertEqual(len(labels), 3)
        
        # Test for quarters
        boundaries, labels = prepare_time_periods('quarter', 2)
        self.assertEqual(len(boundaries), 2)
        self.assertEqual(len(labels), 2)
    
    def test_normalize_burst_scores(self):
        """Test normalizing burst scores across elements"""
        # Create mock burst data for two elements
        burst_data = {
            'elem1': pd.DataFrame({
                'burst_intensity': [50, 75, 100]
            }),
            'elem2': pd.DataFrame({
                'burst_intensity': [25, 30, 35]
            })
        }
        
        # Normalize
        normalized = normalize_burst_scores(burst_data)
        
        # Check results
        self.assertIn('normalized_intensity', normalized['elem1'].columns)
        self.assertIn('normalized_intensity', normalized['elem2'].columns)
        
        # Check that max normalized value is 100
        self.assertAlmostEqual(normalized['elem1']['normalized_intensity'].max(), 100.0)
        
        # Check that normalization preserves relative proportions
        ratio_raw = burst_data['elem2']['burst_intensity'].max() / burst_data['elem1']['burst_intensity'].max()
        ratio_normalized = normalized['elem2']['normalized_intensity'].max() / normalized['elem1']['normalized_intensity'].max()
        self.assertAlmostEqual(ratio_raw, ratio_normalized, places=5)

class TestVisualizationFunctions(unittest.TestCase):
    """Test suite for visualization functions"""
    
    def setUp(self):
        """Set up test data"""
        self.burst_data = create_mock_burst_data()
        self.historical_events = create_mock_historical_events()
        
        # Create summary DataFrame
        summary_data = []
        for data_type, elements in self.burst_data.items():
            for element, df in elements.items():
                if not df.empty:
                    for _, row in df.iterrows():
                        summary_data.append({
                            'element': element,
                            'period': row['period'],
                            'burst_intensity': row['burst_intensity']
                        })
        self.summary_df = pd.DataFrame(summary_data)
        
        # Create summary DataFrames for each data type
        self.taxonomy_summary = pd.DataFrame([
            {'element': 'Category 1', 'max_burst_period': 'Week 1', 'max_burst_intensity': 90.0,
             'max_burst_date': datetime.now(), 'total_count': 1000, 'avg_intensity': 70.0},
            {'element': 'Category 2', 'max_burst_period': 'Week 2', 'max_burst_intensity': 80.0,
             'max_burst_date': datetime.now(), 'total_count': 800, 'avg_intensity': 60.0},
        ])
        
        self.keyword_summary = pd.DataFrame([
            {'element': 'Keyword 1', 'max_burst_period': 'Week 1', 'max_burst_intensity': 85.0,
             'max_burst_date': datetime.now(), 'total_count': 1200, 'avg_intensity': 65.0},
            {'element': 'Keyword 2', 'max_burst_period': 'Week 3', 'max_burst_intensity': 75.0,
             'max_burst_date': datetime.now(), 'total_count': 900, 'avg_intensity': 55.0},
        ])
        
        self.entity_summary = pd.DataFrame([
            {'element': 'Entity 1', 'max_burst_period': 'Week 2', 'max_burst_intensity': 95.0,
             'max_burst_date': datetime.now(), 'total_count': 1500, 'avg_intensity': 75.0},
            {'element': 'Entity 2', 'max_burst_period': 'Week 1', 'max_burst_intensity': 70.0,
             'max_burst_date': datetime.now(), 'total_count': 700, 'avg_intensity': 50.0},
        ])
    
    def test_create_burst_heatmap(self):
        """Test creating a burst heatmap visualization"""
        fig = create_burst_heatmap(self.summary_df)
        
        # Should be a Plotly figure
        self.assertIsInstance(fig, go.Figure)
        
        # Test with custom color scale
        custom_color_scale = [[0, 'blue'], [1, 'red']]
        fig = create_burst_heatmap(self.summary_df, color_scale=custom_color_scale)
        self.assertIsInstance(fig, go.Figure)
        
        # Test with empty DataFrame
        empty_fig = create_burst_heatmap(pd.DataFrame())
        self.assertIsInstance(empty_fig, go.Figure)
    
    def test_create_burst_summary_chart(self):
        """Test creating a burst summary chart"""
        # Create mock summary data
        summary_df = pd.DataFrame([
            {'element': 'Elem 1', 'max_burst_intensity': 90.0, 'avg_intensity': 70.0,
             'max_burst_period': 'Week 1', 'total_count': 1000},
            {'element': 'Elem 2', 'max_burst_intensity': 80.0, 'avg_intensity': 60.0,
             'max_burst_period': 'Week 2', 'total_count': 800},
        ])
        
        fig = create_burst_summary_chart(summary_df)
        self.assertIsInstance(fig, go.Figure)
        
        # Test with custom parameters
        fig = create_burst_summary_chart(summary_df, title="Custom Title", color='#ff0000')
        self.assertIsInstance(fig, go.Figure)
        
        # Test with empty DataFrame
        empty_fig = create_burst_summary_chart(pd.DataFrame())
        self.assertIsInstance(empty_fig, go.Figure)
    
    def test_create_burst_timeline(self):
        """Test creating a burst timeline visualization"""
        # Extract one data type for testing
        taxonomy_data = self.burst_data.get('taxonomy', {})
        
        fig = create_burst_timeline(taxonomy_data)
        self.assertIsInstance(fig, go.Figure)
        
        # Test with custom parameters
        fig = create_burst_timeline(taxonomy_data, title="Custom Timeline", top_n=3, color_base='#00ff00')
        self.assertIsInstance(fig, go.Figure)
        
        # Test with empty data
        empty_fig = create_burst_timeline({})
        self.assertIsInstance(empty_fig, go.Figure)
    
    def test_create_burst_comparison_chart(self):
        """Test creating a comparison chart across data types"""
        fig = create_burst_comparison_chart(
            self.taxonomy_summary, self.keyword_summary, self.entity_summary
        )
        self.assertIsInstance(fig, go.Figure)
        
        # Test with custom title
        fig = create_burst_comparison_chart(
            self.taxonomy_summary, self.keyword_summary, self.entity_summary,
            title="Custom Comparison"
        )
        self.assertIsInstance(fig, go.Figure)
        
        # Test with some empty DataFrames
        fig = create_burst_comparison_chart(
            pd.DataFrame(), self.keyword_summary, self.entity_summary
        )
        self.assertIsInstance(fig, go.Figure)
        
        # Test with all empty DataFrames
        empty_fig = create_burst_comparison_chart(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )
        self.assertIsInstance(empty_fig, go.Figure)
    
    def test_create_citespace_timeline(self):
        """Test creating a CiteSpace-style timeline"""
        fig = create_citespace_timeline(self.summary_df)
        self.assertIsInstance(fig, go.Figure)
        
        # Test with custom parameters
        fig = create_citespace_timeline(
            self.summary_df, 
            title="Custom CiteSpace Timeline",
            color_scale=["blue", "green", "red"]
        )
        self.assertIsInstance(fig, go.Figure)
        
        # Test with empty DataFrame
        empty_fig = create_citespace_timeline(pd.DataFrame())
        self.assertIsInstance(empty_fig, go.Figure)
    
    def test_create_enhanced_citespace_timeline(self):
        """Test creating an enhanced CiteSpace timeline with events"""
        fig = create_enhanced_citespace_timeline(self.summary_df)
        self.assertIsInstance(fig, go.Figure)
        
        # Test with historical events
        fig = create_enhanced_citespace_timeline(
            self.summary_df,
            historical_events=self.historical_events
        )
        self.assertIsInstance(fig, go.Figure)
        
        # Create mock document links
        doc_links = {}
        for element in self.summary_df['element'].unique()[:3]:
            doc_links[element] = [
                {'period': 'Week 1', 'document_id': 1001, 'title': f"Doc about {element}"}
            ]
        
        # Test with document links
        fig = create_enhanced_citespace_timeline(
            self.summary_df,
            historical_events=self.historical_events,
            document_links=doc_links
        )
        self.assertIsInstance(fig, go.Figure)
        
        # Test with empty DataFrame
        empty_fig = create_enhanced_citespace_timeline(pd.DataFrame())
        self.assertIsInstance(empty_fig, go.Figure)
    
    def test_create_co_occurrence_network(self):
        """Test creating a co-occurrence network visualization"""
        fig = create_co_occurrence_network(self.burst_data)
        self.assertIsInstance(fig, go.Figure)
        
        # Test with custom parameters
        fig = create_co_occurrence_network(
            self.burst_data,
            min_burst_intensity=10.0,
            min_periods=1,
            min_strength=0.1,
            title="Custom Network"
        )
        self.assertIsInstance(fig, go.Figure)
        
        # Test with empty data
        empty_fig = create_co_occurrence_network({})
        self.assertIsInstance(empty_fig, go.Figure)
    
    def test_create_predictive_visualization(self):
        """Test creating a predictive visualization"""
        fig = create_predictive_visualization(self.burst_data)
        self.assertIsInstance(fig, go.Figure)
        
        # Test with custom parameters
        fig = create_predictive_visualization(
            self.burst_data,
            prediction_periods=3,
            confidence_level=0.8,
            title="Custom Prediction"
        )
        self.assertIsInstance(fig, go.Figure)
        
        # Test with empty data
        empty_fig = create_predictive_visualization({})
        self.assertIsInstance(empty_fig, go.Figure)
    
    def test_load_historical_events(self):
        """Test loading historical events"""
        # Test loading default events
        events = load_historical_events()
        self.assertIsInstance(events, list)
        self.assertTrue(len(events) > 0)
        
        # Test loading from a file (mock)
        with patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(self.historical_events))):
            events = load_historical_events('mock_file.json')
            self.assertIsInstance(events, list)
            self.assertEqual(len(events), len(self.historical_events))
    
    def test_prepare_document_links(self):
        """Test preparing document links"""
        # Test with empty data
        links = prepare_document_links({})
        self.assertIsInstance(links, dict)
        
        # Test with burst data but no document IDs
        links = prepare_document_links(self.burst_data)
        self.assertIsInstance(links, dict)
        
        # Test with document IDs
        doc_ids = {'taxonomy-0': [1001, 1002], 'keywords-0': [2001, 2002]}
        links = prepare_document_links(self.burst_data, doc_ids)
        self.assertIsInstance(links, dict)
        if 'taxonomy-0' in links:
            self.assertIsInstance(links['taxonomy-0'], list)
            if links['taxonomy-0']:
                self.assertIn('document_id', links['taxonomy-0'][0])


class TestDataFetchersIntegration(unittest.TestCase):
    """Test suite for data fetchers (with mocking)"""
    
    def setUp(self):
        """Set up test mocks"""
        self.mock_time_series = create_mock_time_series_data()
        self.mock_burst_data = create_mock_burst_data()
        
        # Mock the get_engine function
        self.engine_patcher = patch('database.data_fetchers_freshness.get_engine')
        self.mock_engine = self.engine_patcher.start()
        
        # Setup mock connection and query results
        self.mock_conn = MagicMock()
        self.mock_engine.return_value.connect.return_value.__enter__.return_value = self.mock_conn
        
        # Mock pd.read_sql to return our mock data
        self.read_sql_patcher = patch('pandas.read_sql')
        self.mock_read_sql = self.read_sql_patcher.start()
        self.mock_read_sql.return_value = self.mock_time_series
    
    def tearDown(self):
        """Clean up mocks"""
        self.engine_patcher.stop()
        self.read_sql_patcher.stop()
    
    def test_get_taxonomy_elements_for_burst(self):
        """Test fetching taxonomy elements data for burst analysis"""
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        # Test with default parameters
        result = get_taxonomy_elements_for_burst(start_date, end_date)
        
        # Should return our mock data
        self.assertEqual(result.equals(self.mock_time_series), True)
        
        # Check that the SQL query was called with correct parameters
        self.mock_read_sql.assert_called()
        
        # Test with filter parameters
        result = get_taxonomy_elements_for_burst(
            start_date, end_date, 
            filter_value='russian',
            group_by='subcategory',
            language='RU',
            database='test_db'
        )
        
        # Should still return our mock data
        self.assertEqual(result.equals(self.mock_time_series), True)
    
    def test_get_keywords_for_burst(self):
        """Test fetching keywords data for burst analysis"""
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        # Test with default parameters
        result = get_keywords_for_burst(start_date, end_date)
        
        # Should return our mock data
        self.assertEqual(result.equals(self.mock_time_series), True)
        
        # Check that the SQL query was called
        self.mock_read_sql.assert_called()
        
        # Test with filter parameters
        result = get_keywords_for_burst(
            start_date, end_date, 
            filter_value='western',
            language='EN',
            database='test_db'
        )
        
        # Should still return our mock data
        self.assertEqual(result.equals(self.mock_time_series), True)
    
    def test_get_named_entities_for_burst(self):
        """Test fetching named entities data for burst analysis"""
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        # Test with default parameters
        result = get_named_entities_for_burst(start_date, end_date)
        
        # Should return our mock data
        self.assertEqual(result.equals(self.mock_time_series), True)
        
        # Check that the SQL query was called
        self.mock_read_sql.assert_called()
        
        # Test with filter parameters
        result = get_named_entities_for_burst(
            start_date, end_date, 
            filter_value='military',
            entity_types=['PERSON', 'ORG'],
            language='EN',
            database='test_db'
        )
        
        # Should still return our mock data
        self.assertEqual(result.equals(self.mock_time_series), True)
    
    @patch('database.data_fetchers_freshness.get_taxonomy_elements_for_burst')
    @patch('database.data_fetchers_freshness.get_keywords_for_burst')
    @patch('database.data_fetchers_freshness.get_named_entities_for_burst')
    @patch('database.data_fetchers_freshness.kleinberg_burst_detection')
    def test_get_burst_data_for_periods(self, mock_burst_detection, mock_get_entities, 
                                       mock_get_keywords, mock_get_taxonomy):
        """Test getting burst data for multiple periods"""
        # Configure the mocks
        mock_get_taxonomy.return_value = self.mock_time_series
        mock_get_keywords.return_value = self.mock_time_series
        mock_get_entities.return_value = self.mock_time_series
        
        # Mock the burst detection to return a DataFrame with burst_intensity
        burst_result = self.mock_time_series.copy()
        burst_result['burst_intensity'] = 75.0
        mock_burst_detection.return_value = burst_result
        
        # Test the function
        result = get_burst_data_for_periods('week', 5)
        
        # Should return a dictionary with the right structure
        self.assertIsInstance(result, dict)
        self.assertIn('taxonomy', result)
        self.assertIn('keywords', result)
        self.assertIn('named_entities', result)
        
        # Test with custom parameters
        result = get_burst_data_for_periods(
            'month', 3,
            filter_value='russian',
            data_types=['taxonomy', 'keywords'],
            language='RU',
            taxonomy_level='subcategory',
            entity_types=['PERSON'],
            keywords_top_n=10,
            entities_top_n=10
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('taxonomy', result)
        self.assertIn('keywords', result)
    
    def test_calculate_burst_summaries(self):
        """Test calculating burst summaries"""
        # Test with our mock burst data
        result = calculate_burst_summaries(self.mock_burst_data)
        
        # Should return a dictionary with the right structure
        self.assertIsInstance(result, dict)
        for data_type in self.mock_burst_data.keys():
            self.assertIn(data_type, result)
            if result[data_type].empty:
                continue
            # Check that summary has the right columns
            for col in ['element', 'max_burst_period', 'max_burst_intensity', 
                       'max_burst_date', 'total_count', 'avg_intensity']:
                self.assertIn(col, result[data_type].columns)


if __name__ == '__main__':
    # Suppress ResourceWarning from unittest
    warnings.simplefilter("ignore", ResourceWarning)
    
    # Run tests
    unittest.main()