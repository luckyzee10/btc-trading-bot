#!/usr/bin/env python3
"""
Bitcoin Trading Bot
A paper trading bot that analyzes BTC/USDT on Binance using technical indicators
and generates buy/sell signals based on predefined conditions.
Uses PostgreSQL for data persistence.
Enhanced with Markov Chain state modeling and prediction.
"""

import ccxt
import pandas as pd
import numpy as np
import ta
import time
import logging
from datetime import datetime, timedelta
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from typing import Dict, List, Optional, Tuple
import pickle
from collections import defaultdict

# === MARKOV CHAIN HELPERS =======================================
STATE_MATRIX_FILE = 'transition_matrix.pkl'
MARKOV_STATES = [
    f'{p}-{r}'
    for p in ('UP', 'FLAT', 'DOWN')
    for r in ('OVERSOLD', 'NEUTRAL', 'OVERBOUGHT')
]

# Markov trading configuration
MARKOV_CONFIDENCE_THRESHOLD = 0.70
BULLISH_STATES = {'UP-NEUTRAL', 'UP-OVERBOUGHT', 'FLAT-NEUTRAL'}
BEARISH_STATES = {'DOWN-NEUTRAL', 'DOWN-OVERSOLD', 'FLAT-OVERBOUGHT'}

def label_state(pct_change: float, rsi: float) -> str:
    """Return combined state label (price movement + RSI zone)."""
    if pct_change > 1:
        price_state = 'UP'
    elif pct_change < -1:
        price_state = 'DOWN'
    else:
        price_state = 'FLAT'

    if rsi < 30:
        rsi_state = 'OVERSOLD'
    elif rsi > 70:
        rsi_state = 'OVERBOUGHT'
    else:
        rsi_state = 'NEUTRAL'

    return f'{price_state}-{rsi_state}'

def build_transition_matrix(df: pd.DataFrame) -> dict:
    """Return {state:{next_state:prob}} from labeled DataFrame."""
    counts = defaultdict(lambda: defaultdict(int))
    for cur, nxt in zip(df['markov_state'][:-1], df['markov_state'][1:]):
        counts[cur][nxt] += 1

    matrix = {s: {} for s in MARKOV_STATES}
    for s, nxts in counts.items():
        total = sum(nxts.values())
        if total > 0:
            for nxt, c in nxts.items():
                matrix[s][nxt] = c / total
        else:
            # If no transitions from this state, use uniform distribution
            for nxt in MARKOV_STATES:
                matrix[s][nxt] = 1 / len(MARKOV_STATES)
    
    # Fill missing states with uniform distribution
    for s in MARKOV_STATES:
        if not matrix[s]:
            matrix[s] = {n: 1/len(MARKOV_STATES) for n in MARKOV_STATES}
    
    return matrix

def save_matrix(matrix: dict, path: str = STATE_MATRIX_FILE):
    """Save transition matrix to pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(matrix, f)

def load_matrix(path: str = STATE_MATRIX_FILE) -> dict:
    """Load transition matrix from pickle file."""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        # Return uniform probabilities until we have real data
        return {s: {n: 1/len(MARKOV_STATES) for n in MARKOV_STATES} for s in MARKOV_STATES}

def predict_next_state(current_state: str, transition_matrix: dict) -> Tuple[str, float]:
    """Predict most probable next state and its probability."""
    probs = transition_matrix.get(current_state, {})
    if probs:
        next_state = max(probs, key=probs.get)
        prob_val = probs[next_state]
        return next_state, prob_val
    else:
        # Fallback to most common state
        return 'FLAT-NEUTRAL', 0.33

# === END MARKOV HELPERS =======================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class BTCTradingBot:
    def __init__(self):
        """Initialize the Bitcoin trading bot with PostgreSQL."""
        # Initialize exchange (Kraken is more cloud-friendly than Binance)
        self.exchange = ccxt.kraken({
            'sandbox': False,  # Use real Kraken for market data
            'enableRateLimit': True,
        })
        
        # Trading parameters
        self.symbol = 'BTC/USDT'  # Kraken uses this symbol too
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
        
        # Portfolio simulation - Start with exact BTC amount
        self.starting_btc_amount = 0.07983511  # Exact BTC starting amount
        self.current_balance = 100.0  # Small USD balance for trading fees/flexibility
        self.btc_holdings = 0.0  # Will be set to starting_btc_amount on initialization
        self.position_size_pct = 0.95  # Use 95% of balance for each trade
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Database connection
        self.db_url = os.getenv('DATABASE_URL')
        if not self.db_url:
            logging.error("DATABASE_URL environment variable not found!")
            raise ValueError("PostgreSQL connection required. Add DATABASE_URL environment variable.")
        
        # Initialize database
        self._initialize_database()
        
        # Markov Chain setup
        self.transition_matrix = load_matrix()
        logging.info("Markov transition matrix loaded")
        
        # Weekly Matrix Rebuilding Settings - NEW!
        self.enable_weekly_rebuild = True  # Flag to enable/disable weekly rebuilds
        self.rebuild_frequency_hours = 336  # 14 days - OPTIMIZED from comparison study
        self.matrix_training_window = 1500  # Use 1500 hours for matrix training - OPTIMIZED
        self.last_matrix_rebuild_time = None  # Track when matrix was last rebuilt
        self.matrix_rebuild_count = 0  # Count how many times matrix was rebuilt
        
        # Ensure we have historical data for matrix (back-fill if needed)
        self.ensure_transition_matrix()
        
        logging.info("Bitcoin Trading Bot initialized successfully with PostgreSQL and Markov Chain")
        logging.info(f"Adaptive matrix rebuilding: {'ENABLED' if self.enable_weekly_rebuild else 'DISABLED'}")
        if self.enable_weekly_rebuild:
            logging.info(f"Matrix will rebuild every {self.rebuild_frequency_hours} hours ({self.rebuild_frequency_hours//24} days) - OPTIMIZED")
            logging.info(f"Training window: {self.matrix_training_window} hours ({self.matrix_training_window//24:.1f} days) - OPTIMIZED")

    def _get_db_connection(self):
        """Get database connection."""
        return psycopg2.connect(self.db_url)

    def _initialize_database(self):
        """Initialize the database tables if they don't exist."""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Create trades table with portfolio tracking
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS trades (
                            id SERIAL PRIMARY KEY,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            price DECIMAL(15, 2) NOT NULL,
                            signal VARCHAR(10) NOT NULL,
                            btc_amount DECIMAL(15, 8),
                            usd_amount DECIMAL(15, 2),
                            portfolio_value DECIMAL(15, 2),
                            balance DECIMAL(15, 2),
                            btc_holdings DECIMAL(15, 8)
                        )
                    """)
                    
                    # Create bot_state table for position tracking
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS bot_state (
                            id SERIAL PRIMARY KEY,
                            position VARCHAR(10),
                            entry_price DECIMAL(15, 2),
                            last_signal_time TIMESTAMP,
                            current_balance DECIMAL(15, 2),
                            btc_holdings DECIMAL(15, 8),
                            trade_count INTEGER,
                            winning_trades INTEGER,
                            losing_trades INTEGER,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Create market_data table for storing OHLCV data
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS market_data (
                            id SERIAL PRIMARY KEY,
                            timestamp TIMESTAMP UNIQUE NOT NULL,
                            open_price DECIMAL(15, 2) NOT NULL,
                            high_price DECIMAL(15, 2) NOT NULL,
                            low_price DECIMAL(15, 2) NOT NULL,
                            close_price DECIMAL(15, 2) NOT NULL,
                            volume DECIMAL(20, 8) NOT NULL,
                            rsi DECIMAL(6, 2),
                            bb_upper DECIMAL(15, 2),
                            bb_middle DECIMAL(15, 2),
                            bb_lower DECIMAL(15, 2),
                            ema_200 DECIMAL(15, 2),
                            atr DECIMAL(10, 2),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    conn.commit()
                    logging.info("Database tables initialized successfully")
                    
                    # Load bot state
                    self._load_bot_state()
                    
        except Exception as e:
            logging.error(f"Error initializing database: {e}")
            raise

    def _initialize_btc_holdings(self):
        """Initialize BTC holdings based on $10,000 worth at current price."""
        try:
            current_price = self.get_current_price()
            if current_price > 0:
                # Calculate BTC amount for $10,000
                initial_btc_amount = self.starting_btc_amount
                self.btc_holdings = initial_btc_amount
                self.position = 'long'  # Start with a long position in BTC
                self.entry_price = current_price
                
                logging.info(f"Initialized with {initial_btc_amount:.6f} BTC at ${current_price:,.2f}")
                
                # Save initial state
                self._save_bot_state()
                return True
            else:
                logging.error("Could not fetch current BTC price for initialization")
                return False
        except Exception as e:
            logging.error(f"Error initializing BTC holdings: {e}")
            return False

    def _load_bot_state(self):
        """Load bot state from database."""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT * FROM bot_state ORDER BY updated_at DESC LIMIT 1")
                    state = cur.fetchone()
                    
                    if state:
                        self.position = state['position']
                        self.entry_price = float(state['entry_price']) if state['entry_price'] else None
                        self.last_signal_time = state['last_signal_time']
                        self.current_balance = float(state['current_balance']) if state['current_balance'] else 100.0
                        self.btc_holdings = float(state['btc_holdings']) if state['btc_holdings'] else 0.0
                        self.trade_count = int(state['trade_count']) if state['trade_count'] else 0
                        self.winning_trades = int(state['winning_trades']) if state['winning_trades'] else 0
                        self.losing_trades = int(state['losing_trades']) if state['losing_trades'] else 0
                        logging.info(f"Loaded bot state: position={self.position}, entry_price={self.entry_price}, current_balance=${self.current_balance:,.2f}, btc_holdings={self.btc_holdings:.6f}, trade_count={self.trade_count}")
                    else:
                        logging.info("No previous bot state found - initializing with exact BTC amount")
                        # Initialize with BTC holdings instead of USD
                        if not self._initialize_btc_holdings():
                            logging.error("Failed to initialize BTC holdings")
                        
        except Exception as e:
            logging.error(f"Error loading bot state: {e}")

    def _save_bot_state(self):
        """Save current bot state to database."""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO bot_state (position, entry_price, last_signal_time, current_balance, btc_holdings, trade_count, winning_trades, losing_trades)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        str(self.position) if self.position else None,
                        float(self.entry_price) if self.entry_price else None,
                        self.last_signal_time,
                        float(self.current_balance),
                        float(self.btc_holdings),
                        int(self.trade_count),
                        int(self.winning_trades),
                        int(self.losing_trades)
                    ))
                    conn.commit()
                    
        except Exception as e:
            logging.error(f"Error saving bot state: {e}")

    def fetch_ohlcv_data(self) -> pd.DataFrame:
        """
        Fetch OHLCV data from Binance and store in database.
        
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
            
            # Store latest data in database
            self._store_market_data(df)
            
            logging.info(f"Fetched {len(df)} candles for {self.symbol}")
            return df
            
        except Exception as e:
            logging.error(f"Error fetching OHLCV data: {e}")
            raise

    def _store_market_data(self, df: pd.DataFrame):
        """Store market data in PostgreSQL."""
        try:
            # Get the latest candle only to avoid duplicates
            latest_candle = df.iloc[-1]
            
            with self._get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO market_data (timestamp, open_price, high_price, low_price, close_price, volume)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (timestamp) DO UPDATE SET
                            open_price = EXCLUDED.open_price,
                            high_price = EXCLUDED.high_price,
                            low_price = EXCLUDED.low_price,
                            close_price = EXCLUDED.close_price,
                            volume = EXCLUDED.volume
                    """, (
                        latest_candle.name,
                        float(latest_candle['open']),
                        float(latest_candle['high']),
                        float(latest_candle['low']),
                        float(latest_candle['close']),
                        float(latest_candle['volume'])
                    ))
                    conn.commit()
                    
        except Exception as e:
            logging.error(f"Error storing market data: {e}")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators and update database.
        
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
            
            # Update latest indicators in database
            self._update_indicators_in_db(df.iloc[-1])
            
            logging.info("Technical indicators calculated successfully")
            return df
            
        except Exception as e:
            logging.error(f"Error calculating indicators: {e}")
            raise

    def _update_indicators_in_db(self, latest_row):
        """Update technical indicators in database for the latest timestamp."""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE market_data SET
                            rsi = %s,
                            bb_upper = %s,
                            bb_middle = %s,
                            bb_lower = %s,
                            ema_200 = %s,
                            atr = %s
                        WHERE timestamp = %s
                    """, (
                        float(latest_row['rsi']) if not pd.isna(latest_row['rsi']) else None,
                        float(latest_row['bb_upper']) if not pd.isna(latest_row['bb_upper']) else None,
                        float(latest_row['bb_middle']) if not pd.isna(latest_row['bb_middle']) else None,
                        float(latest_row['bb_lower']) if not pd.isna(latest_row['bb_lower']) else None,
                        float(latest_row['ema_200']) if not pd.isna(latest_row['ema_200']) else None,
                        float(latest_row['atr']) if not pd.isna(latest_row['atr']) else None,
                        latest_row.name
                    ))
                    conn.commit()
                    
        except Exception as e:
            logging.error(f"Error updating indicators in database: {e}")

    def generate_signals(self, df: pd.DataFrame) -> Tuple[Optional[str], str, Dict]:
        """
        Generate trading signals based on technical indicators and Markov prediction.
        
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
            if any(pd.isna(value) for value in indicators.values() if value is not None):
                return None, "Insufficient data - indicators contain NaN values", indicators
            
            signal = None
            reason = ""
            
            # BUY conditions (original logic maintained):
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
            
            # SELL conditions (original logic maintained):
            # - RSI > 50 OR price crosses back above mid Bollinger Band
            elif (indicators['rsi'] > 50 or current_price > indicators['bb_middle']):
                if self.position == 'long':  # Only sell if we have a position
                    signal = 'SELL'
                    if indicators['rsi'] > 50:
                        reason = f"RSI overbought ({indicators['rsi']:.2f})"
                    else:
                        reason = f"Price above BB middle ({current_price:.2f} > {indicators['bb_middle']:.2f})"
            
            # === MARKOV PREDICTION & GATING LOGIC ===
            if len(df) >= 2:
                # Calculate current state
                prev_price = df['close'].iloc[-2]
                current_pct_change = ((current_price - prev_price) / prev_price) * 100
                current_rsi = indicators['rsi']
                current_state = label_state(current_pct_change, current_rsi)
                
                # Get prediction using helper function
                next_state, probability = predict_next_state(current_state, self.transition_matrix)
                
                # Log Markov prediction for observation
                logging.info(f"[MARKOV] Current State: {current_state} → Predicted Next: {next_state} ({probability:.2%})")
                
                # Add to indicators for tracking
                indicators['markov_current_state'] = current_state
                indicators['markov_next_state'] = next_state
                indicators['markov_probability'] = probability
                
                # MARKOV GATING LOGIC
                # 1. If no existing signal, consider Markov-based BUY
                if (signal is None and 
                    next_state in BULLISH_STATES and 
                    probability >= MARKOV_CONFIDENCE_THRESHOLD and 
                    self.position != 'long'):
                    
                    signal = 'BUY'
                    reason = f"Markov: {current_state}→{next_state} ({probability:.0%})"
                    logging.info(f"[MARKOV TRADE] BUY triggered by {current_state}→{next_state} ({probability:.2%})")
                
                # 2. If currently long, consider Markov-based SELL
                elif (self.position == 'long' and 
                      next_state in BEARISH_STATES and 
                      probability >= MARKOV_CONFIDENCE_THRESHOLD):
                    
                    signal = 'SELL'
                    reason = f"Markov exit: {current_state}→{next_state} ({probability:.0%})"
                    logging.info(f"[MARKOV TRADE] SELL triggered by {current_state}→{next_state} ({probability:.2%})")
            # === END MARKOV LOGIC ===
            
            return signal, reason, indicators
            
        except Exception as e:
            logging.error(f"Error generating signals: {e}")
            return None, f"Error: {e}", {}

    def get_current_price(self):
        """Get current BTC price"""
        try:
            ticker = self.exchange.fetch_ticker('BTC/USDT')
            return ticker['last']
        except Exception as e:
            logging.error(f"Error fetching current price: {str(e)}")
            return 0
    
    def log_trade(self, signal, price, btc_amount, usd_amount):
        """Log trade to database with portfolio tracking"""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Convert all values to regular Python types (avoid numpy types)
                    portfolio_value = self.current_balance + (self.btc_holdings * price if self.position == 'long' else 0)
                    
                    cur.execute("""
                        INSERT INTO trades (timestamp, price, signal, btc_amount, usd_amount, portfolio_value, balance, btc_holdings)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        datetime.now(),
                        float(price),
                        str(signal),
                        float(btc_amount),
                        float(usd_amount),
                        float(portfolio_value),
                        float(self.current_balance),
                        float(self.btc_holdings)
                    ))
                    conn.commit()
        except Exception as e:
            logging.error(f"Error logging trade: {str(e)}")

    def get_trading_stats(self) -> Dict:
        """Get comprehensive trading statistics"""
        try:
            # Calculate current portfolio value
            current_portfolio_value = self.current_balance
            current_price = self.get_current_price()
            
            if self.position == 'long' and self.btc_holdings > 0:
                # Add current value of BTC holdings
                current_portfolio_value += self.btc_holdings * current_price
            elif self.position == 'short' and self.btc_holdings > 0:
                # For short positions, calculate unrealized P&L
                unrealized_pnl = (self.entry_price - current_price) * self.btc_holdings
                current_portfolio_value += unrealized_pnl + (self.btc_holdings * self.entry_price)
            
            # Calculate initial portfolio value (starting BTC amount * current price)
            initial_portfolio_value = self.starting_btc_amount * current_price
            
            # Calculate performance metrics (compared to holding initial BTC)
            total_return = ((current_portfolio_value - initial_portfolio_value) / initial_portfolio_value) * 100
            win_rate = (self.winning_trades / self.trade_count * 100) if self.trade_count > 0 else 0
            
            return {
                'starting_btc_amount': self.starting_btc_amount,
                'initial_portfolio_value': initial_portfolio_value,
                'current_balance': self.current_balance,
                'btc_holdings': self.btc_holdings,
                'current_portfolio_value': current_portfolio_value,
                'current_btc_price': current_price,
                'total_return_pct': total_return,
                'total_pnl': current_portfolio_value - initial_portfolio_value,
                'trade_count': self.trade_count,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate_pct': win_rate,
                'current_position': self.position,
                'entry_price': self.entry_price,
                'started_with_btc': True  # Flag to indicate we started with BTC
            }
        except Exception as e:
            logging.error(f"Error calculating trading stats: {str(e)}")
            return {}

    def execute_trade(self, signal, current_price):
        """Execute trade based on signal with portfolio simulation"""
        if signal == 'BUY' and self.position != 'long':
            if self.position == 'short':
                # Close short position first
                pnl = (self.entry_price - current_price) * self.btc_holdings
                self.current_balance += pnl + (self.btc_holdings * self.entry_price)  # Close short
                logging.info(f"Closed SHORT at ${current_price:.2f}, P&L: ${pnl:.2f}")
                
                if pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
            
            # Open long position
            trade_amount = self.current_balance * self.position_size_pct
            btc_to_buy = trade_amount / current_price
            
            self.btc_holdings = btc_to_buy
            self.current_balance -= trade_amount
            self.position = 'long'
            self.entry_price = current_price
            self.trade_count += 1
            
            # Log to database
            self.log_trade('BUY', current_price, btc_to_buy, trade_amount)
            logging.info(f"BUY: ${trade_amount:.2f} worth of BTC ({btc_to_buy:.6f} BTC) at ${current_price:.2f}")
            
        elif signal == 'SELL' and self.position != 'short':
            if self.position == 'long':
                # Close long position first
                trade_value = self.btc_holdings * current_price
                pnl = trade_value - (self.btc_holdings * self.entry_price)
                self.current_balance += trade_value
                logging.info(f"Closed LONG at ${current_price:.2f}, P&L: ${pnl:.2f}")
                
                if pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
            
            # Open short position (simulated)
            trade_amount = self.current_balance * self.position_size_pct
            btc_to_short = trade_amount / current_price
            
            self.btc_holdings = btc_to_short  # Amount we're "shorting"
            self.current_balance -= trade_amount  # Collateral
            self.position = 'short'
            self.entry_price = current_price
            self.trade_count += 1
            
            # Log to database
            self.log_trade('SELL', current_price, btc_to_short, trade_amount)
            logging.info(f"SELL: ${trade_amount:.2f} worth of BTC ({btc_to_short:.6f} BTC) at ${current_price:.2f}")
            
        # Save updated state
        self._save_bot_state()

    def execute_trading_logic(self) -> Dict:
        """
        Execute the main trading logic:
        1. Fetch current market data
        2. Calculate indicators  
        3. Generate signals
        4. Execute trades
        5. Update portfolio state
        
        Returns:
            Dictionary with execution results
        """
        try:
            # NEW: Check and rebuild matrix if needed (weekly adjustment)
            matrix_rebuilt = self.check_and_rebuild_matrix()
            if matrix_rebuilt:
                logging.info("✅ Transition matrix successfully updated with recent market data")
            
            # Fetch market data
            df = self.fetch_ohlcv_data()
            
            if df is None or len(df) == 0:
                logging.warning("No market data available")
                return {'error': 'No market data'}
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            # Generate signals
            signal, reason, indicators = self.generate_signals(df)
            
            # Get current price and timestamp
            current_price = df['close'].iloc[-1]
            current_time = df.index[-1]
            
            # Process signal
            if signal:
                # Execute trade
                self.execute_trade(signal, current_price)
            
            # Get trading stats
            stats = self.get_trading_stats()
            
            # Prepare result summary
            result = {
                'timestamp': current_time,
                'price': current_price,
                'signal': signal,
                'reason': reason,
                'position': self.position,
                'entry_price': self.entry_price,
                'indicators': indicators,
                'candles_analyzed': len(df),
                'trading_stats': stats,
                'matrix_rebuilt': matrix_rebuilt,  # NEW: Include matrix rebuild status
                'matrix_rebuild_count': self.matrix_rebuild_count  # NEW: Include rebuild count
            }
            
            # Log current status
            status_msg = f"Analysis complete - Price: {current_price:.2f}, Signal: {signal or 'HOLD'}, Position: {self.position or 'None'}"
            if matrix_rebuilt:
                status_msg += f" [Matrix Rebuilt #{self.matrix_rebuild_count}]"
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
        
        # Show trading stats
        stats = result.get('trading_stats', {})
        if stats:
            print(f"\nTRADING STATISTICS:")
            print(f"Total trades: {stats.get('total_trades', 0)}")
            print(f"Buy signals: {stats.get('buy_signals', 0)}")
            print(f"Sell signals: {stats.get('sell_signals', 0)}")
        
        print(f"\nCandles Analyzed: {result['candles_analyzed']}")
        print("="*60)

        # Display comprehensive results
        stats = self.get_trading_stats()
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"\n=== BTC Trading Bot Analysis - {current_time} ===")
        print(f"Current BTC Price: ${result.get('price', 'N/A')}")
        print(f"Signal: {result.get('signal', 'N/A')}")
        print(f"Reason: {result.get('reason', 'N/A')}")
        
        print(f"\n=== Portfolio Performance ===")
        print(f"Started with: {self.starting_btc_amount:.6f} BTC")
        print(f"Initial Value: ${stats.get('initial_portfolio_value', 0):,.2f}")
        print(f"Current Balance: ${stats.get('current_balance', 0):,.2f}")
        print(f"BTC Holdings: {stats.get('btc_holdings', 0):.6f} BTC")
        print(f"Current BTC Price: ${stats.get('current_btc_price', 0):,.2f}")
        print(f"Portfolio Value: ${stats.get('current_portfolio_value', 0):,.2f}")
        print(f"Total Return: {stats.get('total_return_pct', 0):+.2f}%")
        print(f"Total P&L: ${stats.get('total_pnl', 0):+,.2f}")
        
        print(f"\n=== Trading Statistics ===")
        print(f"Total Trades: {stats.get('trade_count', 0)}")
        print(f"Winning Trades: {stats.get('winning_trades', 0)}")
        print(f"Losing Trades: {stats.get('losing_trades', 0)}")
        print(f"Win Rate: {stats.get('win_rate_pct', 0):.1f}%")
        print(f"Current Position: {stats.get('current_position') or 'None'}")
        if stats.get('entry_price'):
            print(f"Entry Price: ${stats.get('entry_price'):,.2f}")
        
        print(f"\n=== Technical Indicators ===")
        indicators = result.get('indicators', {})
        print(f"RSI (14): {indicators.get('rsi', 'N/A')}")
        print(f"EMA (200): ${indicators.get('ema_200', 'N/A')}")
        print(f"Bollinger Bands: ${indicators.get('bb_lower', 'N/A')} - ${indicators.get('bb_upper', 'N/A')}")
        print(f"ATR (14): {indicators.get('atr', 'N/A')}")
        
        print(f"\n=== Markov Chain Prediction ===")
        print(f"Current State: {indicators.get('markov_current_state', 'N/A')}")
        print(f"Predicted Next: {indicators.get('markov_next_state', 'N/A')}")
        print(f"Confidence: {indicators.get('markov_probability', 0)*100:.1f}%")
        
        # NEW: Display matrix rebuild information
        if result.get('matrix_rebuilt', False):
            print(f"\n🔄 === Matrix Rebuild Alert ===")
            print(f"Matrix was rebuilt this cycle (Rebuild #{result.get('matrix_rebuild_count', 0)})")
            print(f"Bot adapted to recent market changes!")
        elif self.matrix_rebuild_count > 0:
            print(f"\n🔄 === Adaptive Learning Status ===")
            print(f"Total Matrix Rebuilds: {self.matrix_rebuild_count}")
            if self.last_matrix_rebuild_time:
                next_rebuild = self.last_matrix_rebuild_time + timedelta(hours=self.rebuild_frequency_hours)
                print(f"Next Rebuild: {next_rebuild.strftime('%Y-%m-%d %H:%M')} UTC")
        
        logging.info(f"Analysis complete. Portfolio value: ${stats.get('current_portfolio_value', 0):,.2f}, Return: {stats.get('total_return_pct', 0):+.2f}%")

    def ensure_transition_matrix(self):
        """Ensure we have historical data for transition matrix (back-fill if needed)"""
        try:
            if os.path.exists(STATE_MATRIX_FILE):
                logging.info("Transition matrix file already exists")
                return
            
            logging.info("Building initial Markov matrix from historical data...")
            
            # Fetch extended historical data for better matrix
            candles = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=1000)
            
            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=self.rsi_period).rsi()
            
            # Calculate price change percentage
            df['pct_change'] = df['close'].pct_change() * 100
            
            # Label states
            df['markov_state'] = df.apply(
                lambda row: label_state(row['pct_change'], row['rsi']) 
                if not pd.isna(row['pct_change']) and not pd.isna(row['rsi']) 
                else None, 
                axis=1
            )
            
            # Remove NaN rows
            df_clean = df.dropna(subset=['markov_state'])
            
            if len(df_clean) < 100:
                logging.warning("Insufficient clean data for matrix building, using uniform distribution")
                return
            
            # Build transition matrix
            matrix = build_transition_matrix(df_clean)
            
            # Save matrix
            save_matrix(matrix)
            self.transition_matrix = matrix
            
            logging.info(f"Transition matrix built from {len(df_clean)} data points and saved to {STATE_MATRIX_FILE}")
            
            # Log some matrix stats for verification
            total_states = len([s for s in matrix.keys() if any(matrix[s].values())])
            logging.info(f"Matrix contains {total_states} states with transition data")
            
        except Exception as e:
            logging.error(f"Error ensuring transition matrix: {e}")
            # Continue with uniform distribution if matrix building fails

    def check_and_rebuild_matrix(self):
        """Check if matrix needs rebuilding and rebuild if necessary - NEW WEEKLY ADJUSTMENT!"""
        if not self.enable_weekly_rebuild:
            return False
        
        try:
            current_time = datetime.now()
            
            # Check if it's time to rebuild
            if (self.last_matrix_rebuild_time is None or 
                (current_time - self.last_matrix_rebuild_time).total_seconds() >= self.rebuild_frequency_hours * 3600):
                
                logging.info("Time for weekly matrix rebuild...")
                
                # Get historical data from database for matrix rebuilding
                with self._get_db_connection() as conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        # Get most recent matrix_training_window hours of data
                        cur.execute("""
                            SELECT timestamp, close_price, rsi
                            FROM market_data 
                            WHERE rsi IS NOT NULL 
                            ORDER BY timestamp DESC 
                            LIMIT %s
                        """, (self.matrix_training_window,))
                        
                        rows = cur.fetchall()
                
                if len(rows) < 100:
                    logging.warning(f"Insufficient database data for matrix rebuild ({len(rows)} rows), skipping...")
                    return False
                
                # Convert to DataFrame (reverse to chronological order)
                df = pd.DataFrame(rows[::-1])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # Rename column to match expected structure
                df['close'] = df['close_price']
                
                # Calculate price change percentage
                df['pct_change'] = df['close'].pct_change() * 100
                
                # Label states
                df['markov_state'] = df.apply(
                    lambda row: label_state(row['pct_change'], row['rsi']) 
                    if not pd.isna(row['pct_change']) and not pd.isna(row['rsi']) 
                    else None, 
                    axis=1
                )
                
                # Remove NaN rows
                df_clean = df.dropna(subset=['markov_state'])
                
                if len(df_clean) < 100:
                    logging.warning(f"Insufficient clean data for matrix rebuild ({len(df_clean)} samples), skipping...")
                    return False
                
                # Build new transition matrix
                new_matrix = build_transition_matrix(df_clean)
                
                # Update tracking
                self.matrix_rebuild_count += 1
                self.last_matrix_rebuild_time = current_time
                
                # Save new matrix
                save_matrix(new_matrix)
                self.transition_matrix = new_matrix
                
                # Log the rebuild
                logging.info(f"Matrix Rebuild #{self.matrix_rebuild_count} completed successfully")
                logging.info(f"Training data: {len(df_clean)} samples from {df.index[0]} to {df.index[-1]}")
                logging.info(f"Next rebuild scheduled for: {current_time + timedelta(hours=self.rebuild_frequency_hours)}")
                
                # Save rebuild history for inspection
                rebuild_filename = f'transition_matrix_rebuild_{self.matrix_rebuild_count}_{current_time.strftime("%Y%m%d_%H%M")}.pkl'
                with open(rebuild_filename, 'wb') as f:
                    pickle.dump(new_matrix, f)
                
                return True
                
        except Exception as e:
            logging.error(f"Error in matrix rebuild: {e}")
            return False
        
        return False


def main():
    """Main function to run the trading bot."""
    try:
        # Initialize the bot
        bot = BTCTradingBot()
        
        print("Bitcoin Trading Bot Started!")
        print("This is a PAPER TRADING bot - no real trades will be executed.")
        print(f"Monitoring {bot.symbol} on {bot.timeframe} timeframe")
        print("Trades will be logged to PostgreSQL database")
        print("-" * 60)
        
        # Run a single analysis
        result = bot.run_analysis()
        
        # Print status
        bot.print_current_status(result)
        
        print("\nBot analysis completed successfully!")
        print("Data is persisted in PostgreSQL database.")
        
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except Exception as e:
        logging.error(f"Bot error: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 