# Heroku Deployment Instructions

Follow these steps to deploy the Russian-Ukrainian War Data Analysis Dashboard to Heroku:

## Prerequisites
- You must have a Heroku account
- You must have the Heroku CLI installed

## Step 1: Login to Heroku
```bash
heroku login
```

## Step 2: Create a new Heroku app
```bash
cd "/mnt/c/Apps/ruw-analyze - refactor - 250209/"
heroku create ruw-analysis-dashboard
```

## Step 3: Set up database (if needed)
```bash
heroku addons:create heroku-postgresql:mini
```

## Step 4: Configure environment variables
```bash
# Set any required environment variables
heroku config:set DATABASE_URL=your_database_url
```

## Step 5: Deploy the application
```bash
# Push the code to Heroku
git push heroku refactored-250518:main
```

## Step 6: Open the application
```bash
heroku open
```

## Troubleshooting

If you encounter any issues during deployment, check the logs:
```bash
heroku logs --tail
```

If the app doesn't start properly, you may need to manually start it:
```bash
heroku ps:scale web=1
```

## Database Connection
If you're connecting to an external database, make sure the database is accessible from Heroku's IP ranges and properly configured in your app.

## Optional Dependencies
The app is configured to work without NetworkX, scikit-learn, and SciPy if they're not available, but basic visualizations may be less sophisticated. If you want the full functionality, you can use a larger dyno type and specify all dependencies:

```bash
heroku ps:scale web=standard-1x
```