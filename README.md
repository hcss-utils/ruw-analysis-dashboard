# Russian-Ukrainian War Data Analysis Dashboard

A comprehensive data analysis dashboard for exploring, searching, and visualizing data related to the Russian-Ukrainian War.

## Features

- **Explore Tab**: Hierarchical data exploration with sunburst visualization
- **Search Tab**: Full-text search with categorized results
- **Compare Tab**: Comparative analysis of different data subsets
- **Burstiness Tab**: Trend detection using Kleinberg's burst detection algorithm
- **Sources Tab**: Analysis of source distribution and statistics
- **Keyword Consolidation**: Standardization of keyword variations for consistent analysis

## Key Components

### Data Analysis

- **Burst Detection**: Identifies significant spikes in frequency over time
- **Co-occurrence Analysis**: Shows relationships between elements that burst together
- **Timeline Visualization**: CiteSpace-style timeline of significant events
- **Keyword Mapping**: Consolidates variant forms of keywords/entities to canonical forms

### Technical Features

- **Demo Mode**: Works without a database connection using sample data
- **Responsive Design**: Adapts to different screen sizes
- **Optimized Performance**: Caching and efficient database queries
- **Heroku Deployment**: Ready for cloud deployment

## Documentation

For detailed information on specific features, refer to these documents:

- [Keyword Mapping and Consolidation](KEYWORD_MAPPING.md)
- [Heroku Deployment Guide](HEROKU_DEPLOYMENT.md)
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md)

## Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL database (optional, can run in demo mode)

### Installation

1. Clone the repository
2. Create a virtual environment and install the dependencies
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Set environment variables
   ```bash
   export DATABASE_URL=your_database_url  # or "demo" for demo mode
   export PORT=8051  # Optional, defaults to 8051
   ```
4. Run the application
   ```bash
   python app.py
   ```

## Deployment

The dashboard is optimized for deployment on Heroku:

1. Create a Heroku app
2. Set the `DATABASE_URL` environment variable
3. Deploy the code to Heroku
4. The app will automatically start in demo mode if no valid database connection is available

See [HEROKU_DEPLOYMENT.md](HEROKU_DEPLOYMENT.md) for detailed deployment instructions.
## License

This project is licensed under the terms of the MIT License. See [LICENSE](LICENSE) for more information.

