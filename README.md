# Bitcoin Trading Bot

A Python-based paper trading bot that analyzes BTC/USDT price movements on Binance using technical indicators and generates buy/sell signals. This bot is designed for educational purposes and paper trading only - no real orders are placed.

## Features

- **Real-time Data**: Fetches live BTC/USDT 1-hour candles from Binance
- **Technical Analysis**: Implements multiple indicators:
  - RSI (Relative Strength Index) - 14 period
  - Bollinger Bands - 20 period, 2 standard deviations
  - EMA (Exponential Moving Average) - 200 period
  - ATR (Average True Range) - 14 period
- **Smart Signals**: Generates buy/sell signals based on confluence of indicators
- **Trade Logging**: Logs all trades to CSV with timestamps, prices, and reasons
- **Paper Trading**: Tracks positions and P&L without real money
- **Periodic Execution**: Can run automatically every hour

## Trading Strategy

### Buy Signals
A BUY signal is generated when ALL of the following conditions are met:
- RSI < 30 (oversold condition)
- Price is below the lower Bollinger Band
- ATR is above its recent average (increased volatility)
- Price is above the 200-period EMA (uptrend confirmation)

### Sell Signals
A SELL signal is generated when ANY of the following conditions are met:
- RSI > 50 (momentum weakening)
- Price crosses back above the middle Bollinger Band

## Installation

1. **Clone or download the files**

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**Important**: Always activate the virtual environment first:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Single Analysis Run
To run the bot once and see the current market analysis:
```bash
python btc_trading_bot.py
```

### Continuous Hourly Analysis
To run the bot automatically every hour:
```bash
python run_bot_hourly.py
```

## Files Generated

- **`trades.csv`**: All buy/sell signals with timestamps, prices, and technical indicator values
- **`trading_bot.log`**: Detailed bot execution logs
- **`periodic_bot.log`**: Logs from the hourly scheduler (if using periodic mode)

## Sample Output

```
============================================================
BITCOIN TRADING BOT - STATUS REPORT
============================================================
Timestamp: 2024-01-15 14:00:00+00:00
Current Price: $42,350.75
Signal: BUY
Position: long
Entry Price: $42,350.75

TECHNICAL INDICATORS:
RSI (14): 28.45
Bollinger Bands: Upper $43,200.50 | Middle $42,100.25 | Lower $40,999.75
EMA (200): $41,800.30
ATR (14): 1250.45

Reason: RSI oversold (28.45), price below BB lower (42350.75 < 40999.75), high volatility (ATR 1250.45 > 1180.20), above EMA200

Candles Analyzed: 500
============================================================
```

## CSV Output Format

The `trades.csv` file contains the following columns:
- `timestamp`: When the signal was generated
- `price`: BTC price at signal time
- `signal`: BUY or SELL
- `reason`: Detailed explanation of why the signal was generated
- `rsi`: RSI value at signal time
- `bb_upper`: Upper Bollinger Band value
- `bb_middle`: Middle Bollinger Band value
- `bb_lower`: Lower Bollinger Band value
- `ema_200`: EMA 200 value
- `atr`: ATR value

## Customization

You can modify the trading parameters in the `BTCTradingBot` class:

```python
# Technical indicator parameters
self.rsi_period = 14        # RSI lookback period
self.bb_period = 20         # Bollinger Bands period
self.bb_std = 2             # Bollinger Bands standard deviation
self.ema_period = 200       # EMA period
self.atr_period = 14        # ATR period
```

## Automation Options

### Option 1: Manual Execution
Run the bot manually whenever you want to check the market:
```bash
python btc_trading_bot.py
```

### Option 2: Periodic Scheduler
Use the built-in scheduler to run every hour:
```bash
python run_bot_hourly.py
```

### Option 3: Cron Job (Linux/macOS)
Add to your crontab to run every hour:
```bash
0 * * * * cd /path/to/your/bot && source venv/bin/activate && python btc_trading_bot.py
```

## Important Notes

⚠️ **This is a paper trading bot for educational purposes only**
- No real trades are executed
- No API keys or real money are required
- This is not financial advice
- Always do your own research before making investment decisions

## Requirements

- Python 3.7+
- Internet connection (for fetching market data from Binance)
- All dependencies listed in `requirements.txt`

## Troubleshooting

1. **"Command not found: pip"**: Use `pip3` instead of `pip`, or ensure Python is properly installed
2. **"externally-managed-environment"**: Use a virtual environment (see installation instructions above)
3. **Connection Issues**: Ensure you have a stable internet connection
4. **Missing Dependencies**: Make sure you've activated the virtual environment and run `pip install -r requirements.txt`
5. **Permission Errors**: Make sure the bot can write to the current directory for logs and CSV files

## License

This project is for educational purposes. Use at your own risk. 