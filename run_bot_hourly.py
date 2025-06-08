#!/usr/bin/env python3
"""
Periodic Bitcoin Trading Bot Runner
Runs the BTC trading bot every hour automatically.
Uses PostgreSQL for data persistence.
"""

import time
import schedule
from btc_trading_bot import BTCTradingBot
import logging

# Configure logging (PostgreSQL handles data persistence)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def run_bot_analysis():
    """Run a single bot analysis cycle."""
    try:
        logging.info("=" * 50)
        logging.info("STARTING SCHEDULED BOT ANALYSIS")
        logging.info("=" * 50)
        
        # Initialize and run bot
        bot = BTCTradingBot()
        result = bot.run_analysis()
        
        # Print results
        bot.print_current_status(result)
        
        logging.info("Scheduled analysis completed successfully")
        
    except Exception as e:
        logging.error(f"Error in scheduled analysis: {e}")

def main():
    """Main function to set up periodic execution."""
    print("Bitcoin Trading Bot - Hourly Scheduler")
    print("This will run the trading bot analysis every hour.")
    print("All data is stored in PostgreSQL database.")
    print("Press Ctrl+C to stop the scheduler.")
    print("-" * 50)
    
    # Schedule the bot to run every hour
    schedule.every().hour.do(run_bot_analysis)
    
    # Run once immediately
    print("Running initial analysis...")
    run_bot_analysis()
    
    # Keep the scheduler running
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        print("\nScheduler stopped by user.")
        logging.info("Scheduler stopped by user")

if __name__ == "__main__":
    main() 