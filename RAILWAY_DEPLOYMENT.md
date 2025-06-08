# 🚂 Railway Deployment Guide - PostgreSQL Edition

Deploy your Bitcoin Trading Bot to Railway with PostgreSQL database!

## 🚀 Quick Deployment Steps

### 1. Push to GitHub First
```bash
# Add PostgreSQL configuration
git add -A
git commit -m "Configure bot for PostgreSQL database"

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

### 3. Add PostgreSQL Database
**Critical Step**: Add PostgreSQL to your project
1. In your Railway project dashboard
2. Click "New Service" → "Database" → "PostgreSQL"
3. Railway automatically creates `DATABASE_URL` environment variable
4. Your bot will connect automatically! 🎉

### 4. Verify Connection
Check your bot logs for:
```
Database tables initialized successfully
Bitcoin Trading Bot initialized successfully with PostgreSQL
```

## 🗄️ **Database Schema**

Your bot automatically creates these tables:

### `trades` Table
- All buy/sell signals with timestamps
- Technical indicator values
- P&L calculations
- Trade reasoning

### `market_data` Table  
- OHLCV candle data
- Complete technical analysis
- Historical price movements

### `bot_state` Table
- Current position tracking
- Entry prices
- State persistence across restarts

## 📊 Accessing Your Trading Data

### Method 1: Railway Dashboard Logs
- View real-time trading decisions
- Monitor bot health and status

### Method 2: Database Queries (Railway CLI)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Connect to your project
railway login
railway link [your-project-id]

# View trading summary
railway run python download_trades.py

# Query database directly
railway run psql $DATABASE_URL -c "SELECT COUNT(*) FROM trades;"
```

### Method 3: Export Data
```bash
# Export all trades to CSV
railway run python download_trades.py

# Or run specific exports
railway run python -c "
from download_trades import export_trades_to_csv
export_trades_to_csv('my_trades.csv')
"
```

## 📈 **What Gets Stored:**

✅ **Complete Trading History** - Every signal with full context  
✅ **Technical Analysis** - All indicators for every hour  
✅ **P&L Tracking** - Automatic profit/loss calculations  
✅ **Market Data** - OHLCV data with technical indicators  
✅ **Bot State** - Position tracking across restarts  

**All data persists forever** and survives any restarts!

## 💰 Pricing

**Railway Pricing:**
- **PostgreSQL Database**: $5/month (512MB storage)
- **App Hosting**: $5/month 
- **Total Cost**: ~$10/month for professional-grade setup

**Storage Capacity:**
- 512MB = ~10+ years of hourly trading data
- Much more reliable than file storage

## 🔍 **Sample Database Queries**

### View All Trades
```sql
SELECT timestamp, signal, price, reason 
FROM trades 
ORDER BY timestamp DESC;
```

### Calculate Total P&L
```sql
SELECT 
    SUM(CASE WHEN signal = 'SELL' THEN price ELSE -price END) as total_pnl
FROM trades;
```

### Get Trading Performance
```sql
SELECT 
    signal, 
    COUNT(*) as count,
    AVG(price) as avg_price
FROM trades 
GROUP BY signal;
```

### Latest Technical Analysis
```sql
SELECT timestamp, close_price, rsi, ema_200 
FROM market_data 
WHERE rsi IS NOT NULL 
ORDER BY timestamp DESC 
LIMIT 1;
```

## 🔧 Monitoring Commands

### View Live Results
```bash
# Complete trading summary
railway run python download_trades.py

# Just the stats
railway run psql $DATABASE_URL -c "
SELECT 
    (SELECT COUNT(*) FROM trades) as total_trades,
    (SELECT COUNT(*) FROM trades WHERE signal = 'BUY') as buys,
    (SELECT COUNT(*) FROM trades WHERE signal = 'SELL') as sells;
"
```

### Download Week Results
```bash
# Export everything to CSV files
railway run python download_trades.py
# Choose option 4 for complete export
```

## 🛠 Database Management

### Backup Your Data
```bash
# Full database backup
railway run pg_dump $DATABASE_URL > my_trading_bot_backup.sql
```

### Reset Database (if needed)
```bash
# Warning: This deletes all data!
railway run psql $DATABASE_URL -c "
DROP TABLE IF EXISTS trades, market_data, bot_state;
"
```

### View Database Size
```bash
railway run psql $DATABASE_URL -c "
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public';
"
```

## 🎯 **Week-Long Test Results**

After 7 days, your PostgreSQL database will contain:

- **~168 market analyses** (every hour)
- **Complete trading signals** with detailed reasoning
- **Technical indicator history** for analysis
- **Profit/Loss calculations** for all completed trades
- **Market data archive** for backtesting

### Advanced Analytics Possible:
- Signal accuracy analysis
- Technical indicator effectiveness
- Market condition correlations
- Performance over different time periods

## 🔄 Zero-Downtime Updates

Update your bot while preserving all data:
```bash
git add .
git commit -m "Improve trading strategy"
git push origin main
```

Railway redeploys automatically - **zero data loss**! 

Your PostgreSQL database contains your complete trading history and continues growing even during updates.

## 🏆 **Advantages of PostgreSQL vs CSV:**

✅ **ACID Compliance** - No data corruption  
✅ **Concurrent Access** - Multiple queries safely  
✅ **Rich Queries** - Complex analysis possible  
✅ **Automatic Backups** - Railway handles this  
✅ **Scalability** - Handles massive datasets  
✅ **Professional Grade** - Production-ready storage  

Your trading bot now has enterprise-level data persistence! 🚀 