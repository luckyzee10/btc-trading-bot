#!/usr/bin/env python3
"""
Utility script to download/view trading data from PostgreSQL database
"""

import os
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta

def get_db_connection():
    """Get database connection."""
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print("❌ DATABASE_URL environment variable not found!")
        return None
    return psycopg2.connect(db_url)

def display_trading_summary():
    """Display comprehensive trading results from PostgreSQL."""
    print("🏦 Bitcoin Trading Bot - PostgreSQL Data Summary")
    print("="*60)
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                
                # Get total trades
                cur.execute("SELECT COUNT(*) as total_trades FROM trades")
                total_trades = cur.fetchone()['total_trades']
                
                if total_trades == 0:
                    print("📊 No trades found in database yet.")
                    return
                
                print(f"📊 TRADING STATISTICS:")
                print("-" * 30)
                print(f"Total signals: {total_trades}")
                
                # Get signal breakdown
                cur.execute("SELECT signal, COUNT(*) as count FROM trades GROUP BY signal")
                signal_counts = {row['signal']: row['count'] for row in cur.fetchall()}
                print(f"Buy signals: {signal_counts.get('BUY', 0)}")
                print(f"Sell signals: {signal_counts.get('SELL', 0)}")
                
                # Get date range
                cur.execute("SELECT MIN(timestamp) as first_trade, MAX(timestamp) as last_trade FROM trades")
                date_range = cur.fetchone()
                print(f"Trading period: {date_range['first_trade']} to {date_range['last_trade']}")
                
                # Calculate P&L for completed trades
                cur.execute("""
                    SELECT * FROM trades 
                    WHERE signal IN ('BUY', 'SELL') 
                    ORDER BY timestamp
                """)
                trades = cur.fetchall()
                
                total_pnl = 0
                completed_trades = 0
                current_position = None
                buy_price = None
                
                print(f"\n💰 PROFIT & LOSS ANALYSIS:")
                print("-" * 30)
                
                for trade in trades:
                    if trade['signal'] == 'BUY' and current_position is None:
                        current_position = 'long'
                        buy_price = float(trade['price'])
                    elif trade['signal'] == 'SELL' and current_position == 'long':
                        sell_price = float(trade['price'])
                        pnl = sell_price - buy_price
                        total_pnl += pnl
                        completed_trades += 1
                        current_position = None
                        print(f"Trade #{completed_trades}: Bought at ${buy_price:,.2f}, Sold at ${sell_price:,.2f} = ${pnl:,.2f}")
                
                print(f"\nCompleted trades: {completed_trades}")
                print(f"Total P&L: ${total_pnl:,.2f}")
                if completed_trades > 0:
                    avg_pnl = total_pnl / completed_trades
                    print(f"Average P&L per trade: ${avg_pnl:,.2f}")
                
                # Current position
                cur.execute("SELECT * FROM bot_state ORDER BY updated_at DESC LIMIT 1")
                current_state = cur.fetchone()
                if current_state:
                    print(f"\n🔄 CURRENT POSITION:")
                    print("-" * 30)
                    print(f"Position: {current_state['position'] or 'None'}")
                    if current_state['entry_price']:
                        print(f"Entry price: ${float(current_state['entry_price']):,.2f}")
                        print(f"Last signal: {current_state['last_signal_time']}")
                
                # Recent trades
                print(f"\n📈 RECENT TRADES (Last 5):")
                print("-" * 30)
                cur.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 5")
                recent_trades = cur.fetchall()
                
                for trade in recent_trades:
                    print(f"{trade['timestamp']} | {trade['signal']} at ${float(trade['price']):,.2f}")
                    print(f"  Reason: {trade['reason']}")
                    print(f"  RSI: {trade['rsi']}, EMA200: ${float(trade['ema_200']):,.2f}")
                    print()
                
                # Market data stats
                cur.execute("SELECT COUNT(*) as data_points FROM market_data")
                market_data_count = cur.fetchone()['data_points']
                print(f"📊 Market data points collected: {market_data_count}")
                
                # Latest market analysis
                cur.execute("""
                    SELECT * FROM market_data 
                    WHERE rsi IS NOT NULL 
                    ORDER BY timestamp DESC LIMIT 1
                """)
                latest_analysis = cur.fetchone()
                if latest_analysis:
                    print(f"\n🔍 LATEST TECHNICAL ANALYSIS:")
                    print("-" * 30)
                    print(f"Timestamp: {latest_analysis['timestamp']}")
                    print(f"BTC Price: ${float(latest_analysis['close_price']):,.2f}")
                    print(f"RSI: {latest_analysis['rsi']}")
                    print(f"Bollinger Bands: ${float(latest_analysis['bb_lower']):,.2f} - ${float(latest_analysis['bb_upper']):,.2f}")
                    print(f"EMA 200: ${float(latest_analysis['ema_200']):,.2f}")
                    print(f"ATR: {latest_analysis['atr']}")
                
    except Exception as e:
        print(f"❌ Error accessing database: {e}")

def export_trades_to_csv(filename="btc_trades_export.csv"):
    """Export all trades to CSV file."""
    try:
        with get_db_connection() as conn:
            # Use pandas to read directly from database
            df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp", conn)
            df.to_csv(filename, index=False)
            print(f"✅ Trades exported to {filename}")
            print(f"   Total records: {len(df)}")
            
    except Exception as e:
        print(f"❌ Error exporting trades: {e}")

def export_market_data_to_csv(filename="btc_market_data_export.csv"):
    """Export all market data to CSV file."""
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM market_data ORDER BY timestamp", conn)
            df.to_csv(filename, index=False)
            print(f"✅ Market data exported to {filename}")
            print(f"   Total records: {len(df)}")
            
    except Exception as e:
        print(f"❌ Error exporting market data: {e}")

def main():
    """Main function with menu options."""
    if not get_db_connection():
        return
    
    print("🤖 Bitcoin Trading Bot - Data Viewer")
    print("="*50)
    print("1. View trading summary")
    print("2. Export trades to CSV")
    print("3. Export market data to CSV")
    print("4. View all data")
    print("")
    
    choice = input("Choose option (1-4) or press Enter for summary: ").strip()
    
    if choice == "1" or choice == "":
        display_trading_summary()
    elif choice == "2":
        export_trades_to_csv()
    elif choice == "3":
        export_market_data_to_csv()
    elif choice == "4":
        display_trading_summary()
        print("\n" + "="*60)
        export_trades_to_csv()
        export_market_data_to_csv()
    else:
        print("Invalid choice. Showing summary...")
        display_trading_summary()

if __name__ == "__main__":
    main() 