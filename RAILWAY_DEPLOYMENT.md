# 🚂 Railway Deployment Guide

Deploy your Bitcoin Trading Bot to Railway with persistent data storage!

## 🚀 Quick Deployment Steps

### 1. Push to GitHub First
```bash
# Add Railway config files
git add railway.json nixpacks.toml RAILWAY_DEPLOYMENT.md download_trades.py
git commit -m "Add Railway deployment with persistent storage"

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

### 3. Enable Persistent Storage
**Important**: Your trades and logs are now saved to persistent volumes!
1. Go to your project dashboard
2. Click on your service
3. Go to "Settings" tab
4. The volume `trading-data` is automatically created
5. Your CSV files and logs persist through restarts 🎉

## 📊 Accessing Your Trading Data

### Method 1: Railway Dashboard Logs
- Go to your Railway project
- Click "View Logs"
- Look for trading analysis results in real-time

### Method 2: Connect to Service (Download Files)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and connect
railway login
railway link [your-project-id]

# Run the data summary script
railway run python download_trades.py

# Or access files directly
railway run ls /app/data
railway run cat /app/data/trades.csv
```

### Method 3: File Browser (If Available)
Some Railway plans include file browsers to download files directly.

## 🗂️ **What Gets Saved Persistently:**

✅ **`/app/data/trades.csv`** - All your buy/sell signals  
✅ **`/app/data/trading_bot.log`** - Detailed bot execution logs  
✅ **`/app/data/periodic_bot.log`** - Scheduler logs  

These files **survive restarts** and accumulate data over your week-long test!

## 💰 Pricing

**Railway Pricing for Trading Bot:**
- **Hobby Plan**: $5/month (recommended for persistent volumes)
- **Free Plan**: Limited storage, good for testing
- Running 24/7 with persistent storage: approximately $5-8/month

## 📈 Expected Data Collection

Over a week, you'll collect:
- **~168 hourly analyses** (24 hours × 7 days)
- **Technical indicator values** for each hour
- **All buy/sell signals** with detailed reasons
- **Complete trading performance** metrics

## 🔧 Monitoring Commands

### View Live Trading Results
```bash
# See current data summary
railway run python download_trades.py

# View recent logs
railway logs

# Check if bot is running
railway status
```

### Download Weekly Results
```bash
# Download your trades CSV
railway run cat /app/data/trades.csv > my_btc_trades.csv

# Download full logs
railway run cat /app/data/trading_bot.log > my_bot_logs.txt
```

## 🛠 Troubleshooting

### Bot Not Saving Data?
1. Check Railway dashboard for volume attachment
2. Look for "Data directory:" messages in logs
3. Verify persistent volume is mounted at `/app/data`

### Can't Access Files?
1. Install Railway CLI: `npm install -g @railway/cli`
2. Login: `railway login`
3. Link project: `railway link`
4. Access files: `railway run ls /app/data`

## 🔄 Updating Your Bot

To update your bot while preserving data:
```bash
# Make changes to your code
git add .
git commit -m "Update trading strategy"
git push origin main
```

Railway will redeploy but **keep your trading data**! 📈

## 🎯 **Week-Long Test Results**

After a week, you'll have:
- Complete BTC market analysis for 168 hours
- All trading signals with precise timestamps
- Performance metrics and P&L tracking
- Detailed logs of every decision made

Your data persists forever (or until you delete the Railway project)! 