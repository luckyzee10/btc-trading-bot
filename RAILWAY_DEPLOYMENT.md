# 🚂 Railway Deployment Guide

Deploy your Bitcoin Trading Bot to Railway with just a few clicks!

## 🚀 Quick Deployment Steps

### 1. Push to GitHub First
```bash
# Add Railway config files
git add railway.json nixpacks.toml RAILWAY_DEPLOYMENT.md
git commit -m "Add Railway deployment configuration"

# Push to your GitHub repository
git push origin main
```

### 2. Deploy to Railway

**Option A: One-Click Deploy**
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template)

**Option B: Manual Deploy**
1. Go to [railway.app](https://railway.app)
2. Sign up/Login with GitHub
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Choose your `btc-trading-bot` repository
6. Railway will automatically detect it's a Python app!

### 3. Configure Settings
Once deployed:
1. Go to your project dashboard
2. Click on your service
3. Go to "Settings" tab
4. Set these environment variables (optional):
   - `TZ=UTC` (for consistent timezone)
   - `PYTHONUNBUFFERED=1` (for better logging)

## 📊 Monitoring Your Bot

### View Logs
- Go to your Railway project
- Click "View Logs" 
- You'll see live output from your trading bot

### Download Trade Data
1. Go to "Data" tab in Railway dashboard
2. Connect to your service
3. Download `trades.csv` file

## 💰 Pricing

**Railway Pricing for Trading Bot:**
- **Hobby Plan**: $5/month (recommended)
- **Free Plan**: $5 credit (enough for testing)
- Running 24/7 costs approximately $3-5/month

## 🔧 Management Commands

### Check Bot Status
Railway dashboard shows if your service is running

### Restart Bot
```bash
# Using Railway CLI
railway service restart
```

### View Recent Trades
Check the logs for recent analysis results

## 🛠 Troubleshooting

### Bot Not Starting?
1. Check logs in Railway dashboard
2. Verify `requirements.txt` is correct
3. Check `railway.json` configuration

### Want to Stop for Maintenance?
1. Go to Railway dashboard
2. Click "Settings" → "Sleep Service"
3. Resume when ready

## 📈 Expected Performance

Your bot will:
- Run continuously 24/7
- Analyze BTC every hour (168 times per week)
- Log all signals to `trades.csv`
- Restart automatically if there are any errors
- Cost approximately $3-5/month to run

## 🔄 Updating Your Bot

To update your bot with new features:
```bash
# Make changes to your code
git add .
git commit -m "Update trading strategy"
git push origin main
```

Railway will automatically redeploy with your changes! 