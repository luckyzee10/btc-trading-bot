#!/usr/bin/env python3
"""
Utility script to download/view trading data from Railway
"""

import os
import pandas as pd
from btc_trading_bot import DATA_DIR

def main():
    """Display trading results and file locations."""
    print("🏦 Bitcoin Trading Bot - Data Summary")
    print("="*50)
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
        return
    
    print(f"📁 Data Directory: {DATA_DIR}")
    
    # List all files in data directory
    files = os.listdir(DATA_DIR)
    print(f"📄 Files found: {files}")
    
    # Display trades if available
    trades_file = os.path.join(DATA_DIR, 'trades.csv')
    if os.path.exists(trades_file):
        print("\n📊 TRADING RESULTS:")
        print("-" * 30)
        
        try:
            df = pd.read_csv(trades_file)
            if len(df) > 0:
                print(f"Total signals: {len(df)}")
                print(f"Buy signals: {len(df[df['signal'] == 'BUY'])}")
                print(f"Sell signals: {len(df[df['signal'] == 'SELL'])}")
                print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                
                print("\nRecent trades:")
                print(df.tail().to_string(index=False))
                
                # Calculate basic P&L if we have buy/sell pairs
                buys = df[df['signal'] == 'BUY']
                sells = df[df['signal'] == 'SELL']
                if len(buys) > 0 and len(sells) > 0:
                    print(f"\nLatest buy price: ${buys['price'].iloc[-1]:,.2f}")
                    if len(sells) > 0:
                        print(f"Latest sell price: ${sells['price'].iloc[-1]:,.2f}")
            else:
                print("No trades recorded yet.")
        except Exception as e:
            print(f"Error reading trades: {e}")
    else:
        print("❌ No trades.csv file found")
    
    # Show log file info
    log_file = os.path.join(DATA_DIR, 'trading_bot.log')
    if os.path.exists(log_file):
        print(f"\n📝 Log file size: {os.path.getsize(log_file)} bytes")
        print("Last 5 log lines:")
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-5:]:
                print(f"  {line.strip()}")

if __name__ == "__main__":
    main() 