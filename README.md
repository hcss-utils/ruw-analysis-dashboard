# ruw-analysis-dashboard

## Overview

This is a Flask application built with Python.

A web application originally hosted on Heroku.

## Technology Stack

- **Language**: Python
- **Framework**: Flask
- **Platform**: Originally deployed on Heroku

## Project Structure

- `requirements.txt` - Python dependencies
- `Procfile` - Heroku process configuration

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hcss-utils/ruw-analysis-dashboard.git
   cd ruw-analysis-dashboard
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Run the application:
   ```bash
   python app.py  # or as specified in Procfile
   ```

## Environment Variables

The following environment variables are required:

- `DATABASE_URL`

## Deployment

### Deploying to Heroku

1. Create a new Heroku app:
   ```bash
   heroku create your-app-name
   ```

2. Set environment variables:
   ```bash
   heroku config:set KEY=value
   ```

3. Deploy:
   ```bash
   git push heroku main
   ```

### Alternative Deployment Options

- **Vercel**: For static sites and Next.js apps
- **Netlify**: For static sites
- **Railway**: Similar to Heroku, good for full-stack apps
- **Render**: Another Heroku alternative

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository.

---

*This application was originally hosted on Heroku and has been archived here for preservation and future use.*
