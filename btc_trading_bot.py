#!/usr/bin/env python3
"""
Bitcoin Trading Bot
A paper trading bot that analyzes BTC/USDT on Binance using technical indicators
and generates buy/sell signals based on predefined conditions.
"""

import ccxt
import pandas as pd
import numpy as np
import ta
import csv
import time
import logging
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Tuple

# Create data directory for persistent storage
DATA_DIR = os.getenv('DATA_DIR', '/app/data') if os.path.exists('/app/data') else 'data'
os.makedirs(DATA_DIR, exist_ok=True)

# Configure logging with persistent path
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(DATA_DIR, 'trading_bot.log')),
        logging.StreamHandler()
    ]
)

class BTCTradingBot:
    def __init__(self):
        """Initialize the Bitcoin trading bot."""
        # Initialize Binance exchange (no API keys needed for public data)
        self.exchange = ccxt.binance({
            'sandbox': False,  # Use real Binance for market data
            'enableRateLimit': True,
        })
        
        # Trading parameters
        self.symbol = 'BTC/USDT'
        self.timeframe = '1h'
        self.candles_limit = 500  # Fetch more candles to ensure we have enough data
        
        # Technical indicator parameters
        self.rsi_period = 14
        self.bb_period = 20
        self.bb_std = 2
        self.ema_period = 200
        self.atr_period = 14
        
        # Trading state
        self.position = None  # None, 'long', 'short'
        self.entry_price = None
        self.last_signal_time = None
        
        # CSV file for trade logging (persistent storage)
        self.trades_file = os.path.join(DATA_DIR, 'trades.csv')
        self._initialize_trades_csv()
        
        logging.info("Bitcoin Trading Bot initialized successfully")
        logging.info(f"Data directory: {DATA_DIR}")
        logging.info(f"Trades file: {self.trades_file}")

    def _initialize_trades_csv(self):
        """Initialize the trades CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.trades_file):
            with open(self.trades_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'price', 'signal', 'reason', 'rsi', 
                    'bb_upper', 'bb_middle', 'bb_lower', 'ema_200', 'atr'
                ])
            logging.info(f"Created trades CSV file: {self.trades_file}")

    def fetch_ohlcv_data(self) -> pd.DataFrame:
        """
        Fetch OHLCV data from Binance.
        
        Returns:
            DataFrame with OHLCV data and datetime index
        """
        try:
            # Fetch candles
            candles = self.exchange.fetch_ohlcv(
                self.symbol, 
                self.timeframe, 
                limit=self.candles_limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logging.info(f"Fetched {len(df)} candles for {self.symbol}")
            return df
            
        except Exception as e:
            logging.error(f"Error fetching OHLCV data: {e}")
            raise

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicator columns
        """
        try:
            # Make a copy to avoid modifying the original
            df = df.copy()
            
            # RSI (14)
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=self.rsi_period).rsi()
            
            # Bollinger Bands (20, 2 std dev)
            bb_indicator = ta.volatility.BollingerBands(
                df['close'], 
                window=self.bb_period, 
                window_dev=self.bb_std
            )
            df['bb_upper'] = bb_indicator.bollinger_hband()
            df['bb_middle'] = bb_indicator.bollinger_mavg()
            df['bb_lower'] = bb_indicator.bollinger_lband()
            
            # EMA (200)
            df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=self.ema_period).ema_indicator()
            
            # ATR (14)
            df['atr'] = ta.volatility.AverageTrueRange(
                df['high'], 
                df['low'], 
                df['close'], 
                window=self.atr_period
            ).average_true_range()
            
            # ATR moving average for comparison
            df['atr_ma'] = df['atr'].rolling(window=20).mean()
            
            logging.info("Technical indicators calculated successfully")
            return df
            
        except Exception as e:
            logging.error(f"Error calculating indicators: {e}")
            raise

    def generate_signals(self, df: pd.DataFrame) -> Tuple[Optional[str], str, Dict]:
        """
        Generate trading signals based on technical indicators.
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            Tuple of (signal, reason, indicator_values)
        """
        try:
            # Get the latest values (most recent candle)
            latest = df.iloc[-1]
            current_price = latest['close']
            
            # Current indicator values
            indicators = {
                'rsi': latest['rsi'],
                'bb_upper': latest['bb_upper'],
                'bb_middle': latest['bb_middle'],
                'bb_lower': latest['bb_lower'],
                'ema_200': latest['ema_200'],
                'atr': latest['atr'],
                'atr_ma': latest['atr_ma']
            }
            
            # Check for NaN values
            if any(pd.isna(value) for value in indicators.values()):
                return None, "Insufficient data - indicators contain NaN values", indicators
            
            signal = None
            reason = ""
            
            # BUY conditions:
            # - RSI < 30
            # - Price is below lower Bollinger Band
            # - ATR is above its recent average
            # - Price is above EMA 200
            if (indicators['rsi'] < 30 and 
                current_price < indicators['bb_lower'] and 
                indicators['atr'] > indicators['atr_ma'] and 
                current_price > indicators['ema_200']):
                
                signal = 'BUY'
                reason = f"RSI oversold ({indicators['rsi']:.2f}), price below BB lower ({current_price:.2f} < {indicators['bb_lower']:.2f}), high volatility (ATR {indicators['atr']:.2f} > {indicators['atr_ma']:.2f}), above EMA200"
            
            # SELL conditions:
            # - RSI > 50 OR price crosses back above mid Bollinger Band
            elif (indicators['rsi'] > 50 or current_price > indicators['bb_middle']):
                if self.position == 'long':  # Only sell if we have a position
                    signal = 'SELL'
                    if indicators['rsi'] > 50:
                        reason = f"RSI overbought ({indicators['rsi']:.2f})"
                    else:
                        reason = f"Price above BB middle ({current_price:.2f} > {indicators['bb_middle']:.2f})"
            
            return signal, reason, indicators
            
        except Exception as e:
            logging.error(f"Error generating signals: {e}")
            return None, f"Error: {e}", {}

    def log_trade(self, timestamp: datetime, price: float, signal: str, reason: str, indicators: Dict):
        """
        Log trade to CSV file.
        
        Args:
            timestamp: Trade timestamp
            price: Current price
            signal: BUY/SELL signal
            reason: Reason for the signal
            indicators: Dictionary of indicator values
        """
        try:
            with open(self.trades_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    price,
                    signal,
                    reason,
                    round(indicators.get('rsi', 0), 2),
                    round(indicators.get('bb_upper', 0), 2),
                    round(indicators.get('bb_middle', 0), 2),
                    round(indicators.get('bb_lower', 0), 2),
                    round(indicators.get('ema_200', 0), 2),
                    round(indicators.get('atr', 0), 2)
                ])
            
            logging.info(f"Trade logged: {signal} at {price} - {reason}")
            
        except Exception as e:
            logging.error(f"Error logging trade: {e}")

    def execute_trading_logic(self) -> Dict:
        """
        Execute the main trading logic.
        
        Returns:
            Dictionary with analysis results
        """
        try:
            logging.info("Starting trading analysis...")
            
            # Fetch market data
            df = self.fetch_ohlcv_data()
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            # Generate signals
            signal, reason, indicators = self.generate_signals(df)
            
            # Get current price and timestamp
            current_price = df['close'].iloc[-1]
            current_time = df.index[-1]
            
            # Process signal
            if signal:
                # Update position state
                if signal == 'BUY' and self.position != 'long':
                    self.position = 'long'
                    self.entry_price = current_price
                    self.last_signal_time = current_time
                    
                    # Log the trade
                    self.log_trade(current_time, current_price, signal, reason, indicators)
                    
                elif signal == 'SELL' and self.position == 'long':
                    profit_loss = current_price - self.entry_price if self.entry_price else 0
                    self.position = None
                    self.entry_price = None
                    self.last_signal_time = current_time
                    
                    # Log the trade with P&L info
                    reason_with_pnl = f"{reason} | P&L: {profit_loss:.2f} USDT"
                    self.log_trade(current_time, current_price, signal, reason_with_pnl, indicators)
            
            # Prepare result summary
            result = {
                'timestamp': current_time,
                'price': current_price,
                'signal': signal,
                'reason': reason,
                'position': self.position,
                'entry_price': self.entry_price,
                'indicators': indicators,
                'candles_analyzed': len(df)
            }
            
            # Log current status
            status_msg = f"Analysis complete - Price: {current_price:.2f}, Signal: {signal or 'HOLD'}, Position: {self.position or 'None'}"
            logging.info(status_msg)
            
            return result
            
        except Exception as e:
            logging.error(f"Error in trading logic: {e}")
            raise

    def run_analysis(self) -> Dict:
        """
        Public method to run a single analysis cycle.
        
        Returns:
            Dictionary with analysis results
        """
        return self.execute_trading_logic()

    def print_current_status(self, result: Dict):
        """
        Print a formatted status report.
        
        Args:
            result: Result dictionary from execute_trading_logic
        """
        print("\n" + "="*60)
        print("BITCOIN TRADING BOT - STATUS REPORT")
        print("="*60)
        print(f"Timestamp: {result['timestamp']}")
        print(f"Current Price: ${result['price']:,.2f}")
        print(f"Signal: {result['signal'] or 'HOLD'}")
        print(f"Position: {result['position'] or 'None'}")
        if result['entry_price']:
            print(f"Entry Price: ${result['entry_price']:,.2f}")
            unrealized_pnl = result['price'] - result['entry_price']
            print(f"Unrealized P&L: ${unrealized_pnl:,.2f}")
        
        print("\nTECHNICAL INDICATORS:")
        indicators = result['indicators']
        if indicators:
            print(f"RSI (14): {indicators.get('rsi', 0):.2f}")
            print(f"Bollinger Bands: Upper ${indicators.get('bb_upper', 0):,.2f} | Middle ${indicators.get('bb_middle', 0):,.2f} | Lower ${indicators.get('bb_lower', 0):,.2f}")
            print(f"EMA (200): ${indicators.get('ema_200', 0):,.2f}")
            print(f"ATR (14): {indicators.get('atr', 0):.2f}")
        
        if result['reason']:
            print(f"\nReason: {result['reason']}")
        
        print(f"\nCandles Analyzed: {result['candles_analyzed']}")
        print("="*60)


def main():
    """Main function to run the trading bot."""
    try:
        # Initialize the bot
        bot = BTCTradingBot()
        
        print("Bitcoin Trading Bot Started!")
        print("This is a PAPER TRADING bot - no real trades will be executed.")
        print(f"Monitoring {bot.symbol} on {bot.timeframe} timeframe")
        print(f"Trades will be logged to: {bot.trades_file}")
        print("-" * 60)
        
        # Run a single analysis
        result = bot.run_analysis()
        
        # Print status
        bot.print_current_status(result)
        
        print("\nBot analysis completed successfully!")
        print("To run this periodically, you can set up a cron job or scheduler.")
        print("Example: Run every hour with 'python btc_trading_bot.py'")
        
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except Exception as e:
        logging.error(f"Bot error: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 