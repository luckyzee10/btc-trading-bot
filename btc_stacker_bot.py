#!/usr/bin/env python3
"""
BTC STACKER BOT - BITCOIN ACCUMULATION STRATEGY
=============================================
Modified version of the Champion Strategy optimized for BTC accumulation:
- 🎯 Focus on increasing BTC holdings over time
- 💎 Long-term bitcoin accumulation strategy
- 📊 Performance measured in BTC gains
- 🔄 Dynamic entry/exit based on BTC opportunity cost
"""

import ccxt
import pandas as pd
import numpy as np
import ta
import time
import json
import pickle
import logging
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import signal
import requests
from typing import Dict, List, Tuple, Optional
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('btc_stacker.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# === STRATEGY CONFIGURATION ===
STATE_MATRIX_FILE = 'stacker_matrix.pkl'
TRADES_LOG_FILE = 'stacker_trades.csv'
PORTFOLIO_LOG_FILE = 'stacker_portfolio.csv'
MARKOV_STATES = [
    f'{p}-{r}'
    for p in ('UP', 'FLAT', 'DOWN')
    for r in ('OVERSOLD', 'NEUTRAL', 'OVERBOUGHT')
]

class BTCStackerBot:
    def __init__(self):
        """Initialize the BTC stacking bot with accumulation settings."""
        
        # Load configuration
        self.api_key = os.getenv('EXCHANGE_API_KEY')
        self.api_secret = os.getenv('EXCHANGE_API_SECRET')
        self.exchange_name = os.getenv('EXCHANGE_NAME', 'binance')
        self.symbol = os.getenv('TRADING_SYMBOL', 'BTC/USDT')
        self.webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        
        # Stacking Strategy Configuration
        self.position_size_btc = 0.01  # Base position size in BTC
        self.min_profit_btc = 0.0001   # Minimum profit target in BTC
        self.max_position_btc = 0.1    # Maximum BTC position size
        self.stop_loss_btc = 0.005     # Stop loss in BTC terms
        
        # Accumulation Settings
        self.target_btc_allocation = 0.75  # Target 75% of portfolio in BTC
        self.rebalance_threshold = 0.1     # Rebalance when 10% off target
        self.dca_enabled = True            # Enable dollar-cost averaging
        self.dca_interval_hours = 24       # DCA every 24 hours
        
        # Safety limits
        self.max_daily_trades = 10     # Conservative trading frequency
        self.emergency_stop_loss = 0.2  # 20% emergency stop in BTC terms
        
        # State variables
        self.running = True
        self.last_dca_time = 0
        self.total_btc_accumulated = 0
        self.initial_btc_balance = 0
        self.active_positions = []
        self.transition_matrix = {}
        
        # Performance tracking
        self.trades_today = 0
        self.total_trades = 0
        self.winning_trades_btc = 0
        self.losing_trades_btc = 0
        self.btc_profit_loss = 0
        
        # Cooldown settings
        self.last_trade_timestamp = 0
        self.trade_cooldown_seconds = 14400  # 4 hours between trades
        
        logger.info("🔷 BTC STACKER BOT INITIALIZED")
        logger.info(f"📊 Strategy: Bitcoin Accumulation")
        logger.info(f"💱 Exchange: {self.exchange_name.upper()}")
        logger.info(f"🎯 Target BTC Allocation: {self.target_btc_allocation*100}%")
        
        # Initialize exchange and logging
        self.setup_exchange()
        self.setup_csv_logs()
        self.initialize_transition_matrix()
        
        # Setup emergency handlers
        signal.signal(signal.SIGINT, self.emergency_shutdown)
        signal.signal(signal.SIGTERM, self.emergency_shutdown)

    def setup_exchange(self):
        """Initialize exchange connection with error handling."""
        try:
            exchange_class = getattr(ccxt, self.exchange_name)
            self.exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True
            })
            
            # Test connection and get initial balance
            balance = self.exchange.fetch_balance()
            self.initial_btc_balance = balance.get('BTC', {}).get('total', 0)
            
            logger.info(f"✅ Exchange connected successfully")
            logger.info(f"💰 Initial BTC Balance: {self.initial_btc_balance:.8f} BTC")
            
            self.log_portfolio_state()
            
        except Exception as e:
            logger.error(f"❌ Exchange setup failed: {e}")
            raise

    def setup_csv_logs(self):
        """Initialize CSV files for trade and portfolio tracking."""
        if not os.path.exists(TRADES_LOG_FILE):
            with open(TRADES_LOG_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'type', 'price', 'btc_amount', 
                               'usdt_value', 'btc_profit_loss', 'total_btc_holdings'])
        
        if not os.path.exists(PORTFOLIO_LOG_FILE):
            with open(PORTFOLIO_LOG_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'btc_balance', 'usdt_balance', 
                               'btc_price', 'total_btc_value'])

    def log_trade(self, trade_data: dict):
        """Log trade with focus on BTC metrics."""
        try:
            with open(TRADES_LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    trade_data['timestamp'],
                    trade_data['type'],
                    trade_data['price'],
                    trade_data['btc_amount'],
                    trade_data['usdt_value'],
                    trade_data['btc_profit_loss'],
                    trade_data['total_btc_holdings']
                ])
        except Exception as e:
            logger.error(f"❌ Failed to log trade: {e}")

    def log_portfolio_state(self):
        """Log current portfolio state with focus on BTC holdings."""
        try:
            balance = self.exchange.fetch_balance()
            ticker = self.exchange.fetch_ticker(self.symbol)
            
            btc_balance = balance.get('BTC', {}).get('total', 0)
            usdt_balance = balance.get('USDT', {}).get('total', 0)
            btc_price = ticker['last']
            
            # Calculate total value in BTC terms
            total_btc_value = btc_balance + (usdt_balance / btc_price)
            
            # Log to console
            logger.info(f"📊 Portfolio State:")
            logger.info(f"   BTC: {btc_balance:.8f}")
            logger.info(f"   USDT: {usdt_balance:.2f}")
            logger.info(f"   Total BTC Value: {total_btc_value:.8f}")
            logger.info(f"   BTC Price: ${btc_price:.2f}")
            
            # Log to CSV
            with open(PORTFOLIO_LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    btc_balance,
                    usdt_balance,
                    btc_price,
                    total_btc_value
                ])
            
            return total_btc_value
            
        except Exception as e:
            logger.error(f"❌ Failed to log portfolio state: {e}")
            return 0

    def should_dca(self) -> bool:
        """Check if it's time for dollar-cost averaging."""
        if not self.dca_enabled:
            return False
            
        hours_since_last_dca = (time.time() - self.last_dca_time) / 3600
        return hours_since_last_dca >= self.dca_interval_hours

    def execute_dca(self):
        """Execute DCA buy of Bitcoin."""
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            
            if usdt_balance < 10:  # Minimum USDT for DCA
                return False
                
            # Use 10% of available USDT for DCA
            dca_amount_usdt = usdt_balance * 0.1
            
            # Create market buy order
            ticker = self.exchange.fetch_ticker(self.symbol)
            btc_amount = dca_amount_usdt / ticker['last']
            
            order = self.exchange.create_market_buy_order(
                self.symbol, 
                btc_amount, 
                {'type': 'market'}
            )
            
            logger.info(f"✅ DCA Buy: {btc_amount:.8f} BTC @ ${ticker['last']:.2f}")
            self.last_dca_time = time.time()
            
            # Log DCA trade
            self.log_trade({
                'timestamp': datetime.now().isoformat(),
                'type': 'DCA_BUY',
                'price': ticker['last'],
                'btc_amount': btc_amount,
                'usdt_value': dca_amount_usdt,
                'btc_profit_loss': 0,
                'total_btc_holdings': balance.get('BTC', {}).get('total', 0) + btc_amount
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ DCA execution failed: {e}")
            return False

    def check_portfolio_balance(self) -> Optional[str]:
        """Check if portfolio needs rebalancing."""
        try:
            balance = self.exchange.fetch_balance()
            ticker = self.exchange.fetch_ticker(self.symbol)
            
            btc_balance = balance.get('BTC', {}).get('total', 0)
            usdt_balance = balance.get('USDT', {}).get('total', 0)
            btc_price = ticker['last']
            
            total_value_usdt = (btc_balance * btc_price) + usdt_balance
            current_btc_allocation = (btc_balance * btc_price) / total_value_usdt
            
            # Check if we're too far from target allocation
            deviation = current_btc_allocation - self.target_btc_allocation
            
            if abs(deviation) > self.rebalance_threshold:
                if deviation < 0:
                    return 'BUY'  # Need more BTC
                else:
                    return 'SELL'  # Need more USDT
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Portfolio balance check failed: {e}")
            return None

    def generate_signal(self, market_data: Dict) -> Tuple[Optional[str], str]:
        """Generate trading signal optimized for BTC accumulation."""
        try:
            if not market_data:
                return None, "No market data"
            
            price = market_data['price']
            rsi = market_data['rsi']
            bb_upper = market_data['bb_upper']
            bb_lower = market_data['bb_lower']
            ema_200 = market_data['ema_200']
            current_state = market_data['markov_state']
            
            # Get current balances
            balance = self.exchange.fetch_balance()
            btc_balance = balance.get('BTC', {}).get('free', 0)
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            
            # Check portfolio balance first
            rebalance_signal = self.check_portfolio_balance()
            if rebalance_signal:
                return rebalance_signal, "Portfolio rebalancing"
            
            # Check for DCA opportunity
            if self.should_dca():
                return 'BUY', "Dollar-cost averaging"
            
            signal = None
            reason = ""
            
            # BUY signals - Accumulation focused
            if (rsi < 30 and  # Strong oversold condition
                price < bb_lower and 
                price > ema_200 * 0.95 and  # Allow slight dips below EMA
                usdt_balance > 100):
                
                signal = 'BUY'
                reason = f"Technical: Strong buy setup for accumulation"
            
            # SELL signals - Only sell if significantly overbought
            elif (rsi > 75 and 
                  price > bb_upper * 1.02 and 
                  btc_balance > self.position_size_btc):
                
                signal = 'SELL'
                reason = f"Technical: Strong overbought, temporary profit taking"
            
            # Add Markov prediction overlay
            if current_state:
                next_state, probability = self.predict_next_state(current_state)
                
                if probability >= 0.75:  # Higher confidence threshold
                    if (next_state.startswith('DOWN') and 
                        btc_balance > self.position_size_btc):
                        signal = 'SELL'
                        reason = f"Markov: High probability downtrend"
                    elif (next_state.startswith('UP') and 
                          usdt_balance > 100):
                        signal = 'BUY'
                        reason = f"Markov: High probability uptrend"
            
            return signal, reason
            
        except Exception as e:
            logger.error(f"❌ Signal generation failed: {e}")
            return None, f"Error: {e}"

    def execute_trade(self, signal: str, market_data: Dict, reason: str):
        """Execute trade with focus on BTC accumulation."""
        try:
            if self.trades_today >= self.max_daily_trades:
                logger.warning("⚠️ Daily trade limit reached")
                return False
            
            price = market_data['price']
            
            if signal == 'BUY':
                # Calculate buy amount in BTC
                balance = self.exchange.fetch_balance()
                usdt_balance = balance.get('USDT', {}).get('free', 0)
                btc_amount = min(
                    (usdt_balance * 0.95) / price,  # Use 95% of USDT
                    self.max_position_btc  # Cap at maximum position
                )
                
                if btc_amount < 0.0001:  # Minimum BTC amount
                    logger.warning("⚠️ BUY amount too small")
                    return False
                
                # Execute market buy
                order = self.exchange.create_market_buy_order(
                    self.symbol,
                    btc_amount,
                    {'type': 'market'}
                )
                
                logger.info(f"✅ BUY: {btc_amount:.8f} BTC @ ${price:.2f}")
                logger.info(f"💎 Accumulation: {reason}")
                
                # Update tracking
                self.total_btc_accumulated += btc_amount
                self.trades_today += 1
                self.total_trades += 1
                
                # Log trade
                self.log_trade({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'BUY',
                    'price': price,
                    'btc_amount': btc_amount,
                    'usdt_value': btc_amount * price,
                    'btc_profit_loss': 0,
                    'total_btc_holdings': balance.get('BTC', {}).get('total', 0) + btc_amount
                })
                
                return True
                
            elif signal == 'SELL':
                balance = self.exchange.fetch_balance()
                btc_balance = balance.get('BTC', {}).get('free', 0)
                
                # Only sell a portion of holdings
                btc_to_sell = min(
                    btc_balance * 0.25,  # Sell up to 25% of holdings
                    self.position_size_btc  # Cap at position size
                )
                
                if btc_to_sell < 0.0001:
                    logger.warning("⚠️ SELL amount too small")
                    return False
                
                # Execute market sell
                order = self.exchange.create_market_sell_order(
                    self.symbol,
                    btc_to_sell,
                    {'type': 'market'}
                )
                
                logger.info(f"✅ SELL: {btc_to_sell:.8f} BTC @ ${price:.2f}")
                logger.info(f"💰 Taking profits: {reason}")
                
                # Update tracking
                self.trades_today += 1
                self.total_trades += 1
                
                # Calculate profit/loss in BTC terms
                btc_profit = btc_to_sell * (1 - 0.001)  # Account for fees
                self.btc_profit_loss += btc_profit
                
                if btc_profit > 0:
                    self.winning_trades_btc += 1
                else:
                    self.losing_trades_btc += 1
                
                # Log trade
                self.log_trade({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'SELL',
                    'price': price,
                    'btc_amount': btc_to_sell,
                    'usdt_value': btc_to_sell * price,
                    'btc_profit_loss': btc_profit,
                    'total_btc_holdings': balance.get('BTC', {}).get('total', 0) - btc_to_sell
                })
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
            return False

    def run_trading_loop(self):
        """Main trading loop optimized for BTC accumulation."""
        logger.info("🚀 Starting BTC Stacker Bot!")
        logger.info("💎 Focus: Bitcoin Accumulation")
        
        while self.running:
            try:
                # Reset daily counter if needed
                current_date = datetime.now().date()
                if self.last_date != current_date:
                    self.trades_today = 0
                    self.last_date = current_date
                
                # Get market data
                market_data = self.get_current_market_data()
                if not market_data:
                    time.sleep(30)
                    continue
                
                # Generate trading signal
                signal, reason = self.generate_signal(market_data)
                
                if signal:
                    # Check cooldown
                    if time.time() - self.last_trade_timestamp >= self.trade_cooldown_seconds:
                        success = self.execute_trade(signal, market_data, reason)
                        if success:
                            self.last_trade_timestamp = time.time()
                    else:
                        logger.info("⏳ Trade cooldown in effect")
                
                # Log portfolio state every hour
                if time.time() - self.last_portfolio_log_time > 3600:
                    self.log_portfolio_state()
                    self.last_portfolio_log_time = time.time()
                    
                    # Log accumulation progress
                    btc_gained = self.total_btc_accumulated - self.initial_btc_balance
                    logger.info(f"📈 BTC Accumulated: {btc_gained:.8f} BTC")
                    logger.info(f"🎯 Total Trades: {self.total_trades}")
                
                time.sleep(300)  # 5-minute check interval
                
            except Exception as e:
                logger.error(f"❌ Trading loop error: {e}")
                time.sleep(60)

def main():
    """Main function to start the BTC stacking bot."""
    print("💎 BTC STACKER BOT")
    print("=" * 50)
    print("🎯 Focus: Bitcoin Accumulation")
    print("📈 Strategy: Dynamic BTC stacking")
    print("⚡ Features:")
    print("  - Automated DCA")
    print("  - Portfolio rebalancing")
    print("  - Technical analysis")
    print("  - Markov predictions")
    
    try:
        bot = BTCStackerBot()
        bot.run_trading_loop()
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Bot crashed: {e}")
        logging.error(f"Bot crashed: {e}")

if __name__ == "__main__":
    main() 