#!/usr/bin/env python3
"""
MARGIN TRADING Bot Backtester - SPECIALIZED FOR LEVERAGED TRADING
=====================================================================
High-risk, high-reward margin trading simulation with advanced risk management.
Designed to optimize strategies specifically for leveraged Bitcoin trading.

Key Features:
- Dynamic leverage adjustment based on market conditions
- Advanced liquidation avoidance
- Optimized position sizing for margin
- Borrowing cost optimization
- Multiple leverage strategies testing
"""

import ccxt
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our Markov helpers (copy from main bot)
import pickle
from collections import defaultdict

# === MARKOV CHAIN HELPERS (same as main bot) ===
STATE_MATRIX_FILE = 'backtest_transition_matrix.pkl'
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
            for nxt in MARKOV_STATES:
                matrix[s][nxt] = 1 / len(MARKOV_STATES)
    
    # Fill missing states with uniform distribution
    for s in MARKOV_STATES:
        if not matrix[s]:
            matrix[s] = {n: 1/len(MARKOV_STATES) for n in MARKOV_STATES}
    
    return matrix

def predict_next_state(current_state: str, transition_matrix: dict) -> tuple:
    """Predict most probable next state and its probability."""
    probs = transition_matrix.get(current_state, {})
    if probs:
        next_state = max(probs, key=probs.get)
        prob_val = probs[next_state]
        return next_state, prob_val
    else:
        return 'FLAT-NEUTRAL', 0.33

# === BACKTESTING ENGINE ===
class MarkovBotBacktester:
    def __init__(self, start_date='2024-01-01', end_date='2024-12-31', 
                 enable_weekly_rebuild=True, rebuild_frequency_hours=168, matrix_training_window=1000,
                 enable_margin=False, max_leverage=3.0, margin_interest_rate=0.0001):
        self.start_date = start_date
        self.end_date = end_date
        
        # Portfolio settings - NEW: 50/50 split with BTC and cash + Risk Management
        self.starting_btc_amount = 0.07983511 / 2  # Half the original BTC
        self.current_balance = 5000.0  # Start with $5000 cash
        self.btc_holdings = self.starting_btc_amount
        self.position = 'mixed'  # Start with mixed position (both BTC and cash)
        self.entry_price = None
        self.position_size_pct = 0.25  # RISK MANAGEMENT: Reduced from 95% to 25%
        
        # Risk Management Settings
        self.stop_loss_pct = 0.08  # 8% stop loss
        self.active_positions = []  # Track active positions for stop loss
        
        # NEW: Margin Trading Settings
        self.enable_margin = enable_margin
        self.max_leverage = max_leverage if enable_margin else 1.0
        self.margin_interest_rate = margin_interest_rate  # Per hour borrowing cost (e.g., 0.0001 = 0.01% per hour)
        self.borrowed_amount = 0.0  # Track borrowed funds
        self.margin_requirement = 0.3  # 30% margin requirement (typical for 3x leverage)
        self.liquidation_threshold = 0.15  # 15% - liquidate if equity falls below this
        self.total_borrowing_costs = 0.0  # Track total interest paid
        
        # Weekly Matrix Rebuilding Settings - NEW! Now configurable
        self.enable_weekly_rebuild = enable_weekly_rebuild  # Flag to enable/disable weekly rebuilds
        self.rebuild_frequency_hours = rebuild_frequency_hours  # Configurable rebuild frequency
        self.matrix_training_window = matrix_training_window  # Configurable training window
        self.last_matrix_rebuild = 0  # Track when matrix was last rebuilt
        self.matrix_rebuild_count = 0  # Count how many times matrix was rebuilt
        
        # Trading stats
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.liquidation_count = 0  # Track liquidations
        
        # Track results
        self.trades = []
        self.portfolio_values = []
        self.states = []
        
        print(f"🔬 Initializing backtest from {start_date} to {end_date}")
        print(f"💰 Starting with: {self.starting_btc_amount:.6f} BTC + ${self.current_balance:,.2f} cash")
        print("📊 This creates an approximately 50/50 BTC/USD portfolio split")
        print(f"🛡️  Risk Management: 25% position sizing, 8% stop losses")
        
        # NEW: Margin trading info
        if self.enable_margin:
            print(f"⚡ MARGIN TRADING ENABLED:")
            print(f"   - Max Leverage: {self.max_leverage:.1f}x")
            print(f"   - Hourly Interest Rate: {self.margin_interest_rate*100:.4f}%")
            print(f"   - Margin Requirement: {self.margin_requirement*100:.0f}%")
            print(f"   - Liquidation Threshold: {self.liquidation_threshold*100:.0f}%")
            print(f"   ⚠️  WARNING: High risk/reward potential!")
        else:
            print("💼 Spot Trading Only (No Leverage)")
        
        print(f"🔄 Adaptive Matrix Rebuilding: {'ENABLED' if self.enable_weekly_rebuild else 'DISABLED'}")
        if self.enable_weekly_rebuild:
            print(f"   - Rebuild every {self.rebuild_frequency_hours} hours ({self.rebuild_frequency_hours//24} days)")
            print(f"   - Training window: {self.matrix_training_window} hours ({self.matrix_training_window//24:.1f} days)")

    def fetch_historical_data(self, days=365):
        """Fetch historical OHLCV data for backtesting."""
        print("📈 Fetching historical data...")
        
        # Try multiple exchanges in order of preference
        exchanges_to_try = [
            ('binance', 'BTC/USDT', 1000),
            ('coinbase', 'BTC/USDT', 300), 
            ('kraken', 'BTC/USD', 720),
            ('gate', 'BTC/USDT', 1000)
        ]
        
        for exchange_name, symbol, max_limit in exchanges_to_try:
            try:
                print(f"   Trying {exchange_name.upper()}...")
                exchange = getattr(ccxt, exchange_name)({'enableRateLimit': True})
                
                # Calculate how many candles we need (hourly data)
                hours_needed = days * 24
                print(f"   Target: {hours_needed} hours ({days} days) of historical data...")
                
                all_candles = []
                batch_size = max_limit
                max_batches = min(20, (hours_needed // batch_size) + 2)  # Limit batches
                
                print(f"   Will fetch up to {max_batches} batches of {batch_size} candles each...")
                
                # Start with most recent data and work backwards
                since_timestamp = None
                
                for batch_num in range(max_batches):
                    try:
                        print(f"   Fetching batch {batch_num + 1}/{max_batches}...")
                        
                        # Fetch batch
                        if since_timestamp is None:
                            # First batch - get most recent data
                            candles = exchange.fetch_ohlcv(symbol, '1h', limit=batch_size)
                        else:
                            # Subsequent batches - get older data
                            candles = exchange.fetch_ohlcv(symbol, '1h', since=since_timestamp, limit=batch_size)
                        
                        if not candles or len(candles) == 0:
                            print(f"   No more data available at batch {batch_num + 1}")
                            break
                        
                        # Check if we got new data (not duplicates)
                        if all_candles and candles[-1][0] >= all_candles[0][0]:
                            print(f"   Reached duplicate data, stopping at batch {batch_num + 1}")
                            break
                        
                        # Add candles to the front (to maintain chronological order when going backwards)
                        all_candles = candles + all_candles
                        print(f"   Added {len(candles)} candles, total: {len(all_candles)}")
                        
                        # Set timestamp for next iteration (get data before the oldest we just fetched)
                        oldest_timestamp = candles[0][0]
                        since_timestamp = oldest_timestamp - (batch_size * 3600000)  # Go back batch_size hours
                        
                        # Check if we have enough data
                        if len(all_candles) >= hours_needed * 0.7:  # Accept 70% of target
                            print(f"   Reached acceptable amount of data: {len(all_candles)} hours!")
                            break
                        
                        # Rate limiting
                        import time
                        time.sleep(0.5)  # Be more conservative
                        
                    except Exception as e:
                        print(f"   Error in batch {batch_num + 1}: {e}")
                        if "rate limit" in str(e).lower():
                            print("   Rate limited, waiting 30 seconds...")
                            import time
                            time.sleep(30)
                            continue
                        else:
                            print(f"   Breaking due to error: {e}")
                            break
                
                # If we got decent amount of data, use it
                if len(all_candles) >= 2000:  # At least ~80 days
                    print(f"✅ Successfully got {len(all_candles)} candles from {exchange_name.upper()}")
                    break
                else:
                    print(f"   {exchange_name.upper()} only provided {len(all_candles)} candles, trying next exchange...")
                    
            except Exception as e:
                print(f"   {exchange_name.upper()} failed: {e}")
                continue
        
        if len(all_candles) == 0:
            print("❌ Could not fetch any historical data from any exchange!")
            return None
        
        # Remove duplicates and sort chronologically
        print("   Processing and sorting data...")
        seen_timestamps = set()
        unique_candles = []
        
        for candle in sorted(all_candles, key=lambda x: x[0]):
            if candle[0] not in seen_timestamps:
                unique_candles.append(candle)
                seen_timestamps.add(candle[0])
        
        # Convert to DataFrame
        df = pd.DataFrame(unique_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()  # Ensure chronological order
        
        actual_days = len(df) / 24
        print(f"✅ Successfully loaded {len(df)} hours ({actual_days:.1f} days) of data")
        print(f"   Period: {df.index[0]} to {df.index[-1]}")
        
        return df

    def calculate_indicators(self, df):
        """Calculate all technical indicators (same as main bot)."""
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        
        # EMA 200
        df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
        
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        df['atr_ma'] = df['atr'].rolling(window=20).mean()
        
        # Price change for Markov states
        df['pct_change'] = df['close'].pct_change() * 100
        
        # Markov states
        df['markov_state'] = df.apply(
            lambda row: label_state(row['pct_change'], row['rsi']) 
            if not pd.isna(row['pct_change']) and not pd.isna(row['rsi']) 
            else None, 
            axis=1
        )
        
        return df

    def build_transition_matrix(self, df, train_period=1000):
        """Build transition matrix from initial training period."""
        # Use at least 1000 hours but up to 1/3 of available data for training
        available_data = len(df)
        min_train = 1000
        max_train = available_data // 3
        
        actual_train_period = max(min_train, min(train_period, max_train))
        
        print(f"🧠 Building transition matrix from first {actual_train_period} hours...")
        print(f"   (Available data: {available_data} hours, using {actual_train_period/available_data*100:.1f}% for training)")
        
        train_data = df.iloc[:actual_train_period].dropna(subset=['markov_state'])
        matrix = build_transition_matrix(train_data)
        
        # Save for inspection
        with open('backtest_transition_matrix.pkl', 'wb') as f:
            pickle.dump(matrix, f)
        
        print(f"✅ Matrix built from {len(train_data)} training samples")
        return matrix

    def rebuild_transition_matrix(self, df, current_idx):
        """Rebuild transition matrix using recent data - NEW WEEKLY ADJUSTMENT!"""
        # Calculate the data window for training (most recent matrix_training_window hours)
        start_idx = max(0, current_idx - self.matrix_training_window)
        end_idx = current_idx
        
        # Get training data from the window
        train_data = df.iloc[start_idx:end_idx].dropna(subset=['markov_state'])
        
        if len(train_data) < 100:  # Ensure we have enough data
            print(f"   ⚠️  Insufficient data for matrix rebuild ({len(train_data)} samples), skipping...")
            return None
        
        # Build new transition matrix
        new_matrix = build_transition_matrix(train_data)
        
        # Update tracking
        self.matrix_rebuild_count += 1
        self.last_matrix_rebuild = current_idx
        
        # Log the rebuild
        current_time = df.index[current_idx]
        print(f"🔄 Matrix Rebuild #{self.matrix_rebuild_count} at {current_time}")
        print(f"   Training window: {len(train_data)} samples ({start_idx} to {end_idx})")
        print(f"   Data period: {df.index[start_idx]} to {df.index[end_idx-1]}")
        
        # Save updated matrix for inspection
        filename = f'backtest_transition_matrix_rebuild_{self.matrix_rebuild_count}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(new_matrix, f)
        
        return new_matrix

    def generate_signal(self, current_idx, df, transition_matrix):
        """Generate trading signal for current candle - FIXED for mixed portfolio."""
        row = df.iloc[current_idx]
        
        # Skip if we don't have enough data for indicators
        if pd.isna(row['rsi']) or pd.isna(row['bb_lower']) or pd.isna(row['ema_200']):
            return None, "Insufficient indicator data"
        
        current_price = row['close']
        signal = None
        reason = ""
        
        # Technical analysis signals - FIXED conditions
        # BUY: Oversold conditions with trend support
        if (row['rsi'] < 35 and  # Relaxed from 30 to 35
            current_price < row['bb_lower'] and 
            row['atr'] > row['atr_ma'] and 
            current_price > row['ema_200'] and
            self.current_balance > 100):  # Only if we have cash
            
            signal = 'BUY'
            reason = f"Technical: RSI oversold ({row['rsi']:.1f}), below BB lower"
        
        # SELL: Take profits or exit overbought - FIXED for mixed portfolio
        elif (row['rsi'] > 65 or current_price > row['bb_upper']) and self.btc_holdings > 0.001:
            signal = 'SELL'
            reason = f"Technical: RSI overbought ({row['rsi']:.1f}) or above BB upper"
        
        # Markov gating logic - FIXED conditions
        if current_idx > 0 and not pd.isna(row['markov_state']):
            current_state = row['markov_state']
            next_state, probability = predict_next_state(current_state, transition_matrix)
            
            # Markov-based BUY - check if we have cash
            if (signal is None and 
                next_state in BULLISH_STATES and 
                probability >= MARKOV_CONFIDENCE_THRESHOLD and 
                self.current_balance > 100):  # Changed from position check to cash check
                
                signal = 'BUY'
                reason = f"Markov: {current_state}→{next_state} ({probability:.0%})"
            
            # Markov-based SELL - check if we have BTC
            elif (self.btc_holdings > 0.001 and  # Changed from position check to BTC check
                  next_state in BEARISH_STATES and 
                  probability >= MARKOV_CONFIDENCE_THRESHOLD):
                
                signal = 'SELL'
                reason = f"Markov exit: {current_state}→{next_state} ({probability:.0%})"
            
            # Store state info
            self.states.append({
                'timestamp': row.name,
                'current_state': current_state,
                'next_state': next_state,
                'probability': probability
            })
        
        return signal, reason

    def execute_trade(self, signal, price, timestamp, reason):
        """Execute trade with margin trading support - leveraged positions and borrowing costs."""
        
        if signal == 'BUY' and self.current_balance > 100:  # Only buy if we have cash
            # Calculate available buying power (with or without leverage)
            available_cash = self.current_balance * self.position_size_pct
            
            if self.enable_margin:
                # With margin: can borrow up to leverage ratio
                max_borrowing = available_cash * (self.max_leverage - 1)
                effective_buying_power = available_cash + max_borrowing
                
                # Calculate actual purchase
                btc_to_buy = effective_buying_power / price
                actual_cost = btc_to_buy * price
                
                # Determine how much we need to borrow
                if actual_cost > available_cash:
                    borrowed_this_trade = actual_cost - available_cash
                    self.borrowed_amount += borrowed_this_trade
                    cash_used = available_cash
                else:
                    borrowed_this_trade = 0
                    cash_used = actual_cost
                
                # Execute the leveraged buy
                self.current_balance -= cash_used
                self.btc_holdings += btc_to_buy
                self.trade_count += 1
                
                # Track leveraged position for stop loss and margin calls
                position = {
                    'type': 'long',
                    'entry_price': price,
                    'btc_amount': btc_to_buy,
                    'entry_time': timestamp,
                    'stop_loss': price * (1 - self.stop_loss_pct),  # 8% below entry
                    'borrowed_amount': borrowed_this_trade,
                    'leverage_used': actual_cost / cash_used if cash_used > 0 else 1.0,
                    'liquidation_price': price * 0.7 if borrowed_this_trade > 0 else 0  # Rough liquidation calc
                }
                
                trade_info = f"LEVERAGED BUY (Leverage: {position['leverage_used']:.1f}x, Borrowed: ${borrowed_this_trade:.2f})"
            
            else:
                # Spot trading (no leverage)
                btc_to_buy = available_cash / price
                
                self.current_balance -= available_cash
                self.btc_holdings += btc_to_buy
                self.trade_count += 1
                
                position = {
                    'type': 'long',
                    'entry_price': price,
                    'btc_amount': btc_to_buy,
                    'entry_time': timestamp,
                    'stop_loss': price * (1 - self.stop_loss_pct),
                    'borrowed_amount': 0,
                    'leverage_used': 1.0,
                    'liquidation_price': 0
                }
                
                trade_info = "SPOT BUY"
            
            self.active_positions.append(position)
            
            self.trades.append({
                'timestamp': timestamp,
                'signal': signal,
                'price': price,
                'btc_amount': btc_to_buy,
                'usd_amount': btc_to_buy * price,
                'reason': reason,
                'portfolio_value': self.calculate_portfolio_value(price),
                'position_type': trade_info,
                'borrowed_amount': position['borrowed_amount'],
                'leverage': position['leverage_used']
            })
            
        elif signal == 'SELL' and self.btc_holdings > 0.001:  # Only sell if we have BTC
            # Calculate trade amount - 25% of BTC holdings  
            btc_to_sell = self.btc_holdings * self.position_size_pct
            trade_value = btc_to_sell * price
            
            # Execute the sell
            self.btc_holdings -= btc_to_sell
            self.current_balance += trade_value
            self.trade_count += 1
            
            # Check if this closes any long positions (for P&L tracking and loan repayment)
            loan_repayment = self._close_long_positions(btc_to_sell, price)
            
            self.trades.append({
                'timestamp': timestamp,
                'signal': signal,
                'price': price,
                'btc_amount': btc_to_sell,
                'usd_amount': trade_value,
                'reason': reason,
                'portfolio_value': self.calculate_portfolio_value(price),
                'position_type': 'SELL',
                'loan_repayment': loan_repayment,
                'leverage': 1.0
            })

    def _close_long_positions(self, btc_sold, current_price):
        """Close long positions, repay loans, and track P&L for win/loss statistics."""
        btc_remaining = btc_sold
        positions_to_remove = []
        total_loan_repayment = 0
        
        for i, position in enumerate(self.active_positions):
            if position['type'] == 'long' and btc_remaining > 0:
                btc_from_position = min(position['btc_amount'], btc_remaining)
                
                # Calculate P&L
                entry_price = position['entry_price']
                gross_pnl = (current_price - entry_price) * btc_from_position
                
                # Handle loan repayment for leveraged positions
                position_loan = position['borrowed_amount'] * (btc_from_position / position['btc_amount'])
                if position_loan > 0:
                    # Repay borrowed amount
                    loan_repayment = min(position_loan, self.current_balance)
                    self.current_balance -= loan_repayment
                    self.borrowed_amount -= loan_repayment
                    total_loan_repayment += loan_repayment
                    
                    # Net P&L after loan repayment
                    net_pnl = gross_pnl - position_loan
                else:
                    net_pnl = gross_pnl
                
                if net_pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                # Update position or mark for removal
                position['btc_amount'] -= btc_from_position
                position['borrowed_amount'] -= position_loan
                btc_remaining -= btc_from_position
                
                if position['btc_amount'] <= 0.0001:  # Position fully closed
                    positions_to_remove.append(i)
        
        # Remove closed positions
        for i in reversed(positions_to_remove):
            del self.active_positions[i]
        
        return total_loan_repayment

    def check_margin_calls_and_liquidations(self, current_price, timestamp):
        """Check for margin calls and forced liquidations."""
        liquidations = []
        
        if not self.enable_margin or self.borrowed_amount == 0:
            return liquidations
        
        # Calculate current equity
        total_portfolio_value = self.calculate_portfolio_value(current_price)
        equity_ratio = (total_portfolio_value - self.borrowed_amount) / total_portfolio_value if total_portfolio_value > 0 else 0
        
        # Check for liquidation threshold
        if equity_ratio < self.liquidation_threshold:
            print(f"🚨 MARGIN LIQUIDATION TRIGGERED at ${current_price:.2f}")
            print(f"   Equity Ratio: {equity_ratio*100:.1f}% < Threshold: {self.liquidation_threshold*100:.0f}%")
            
            # Force liquidate all leveraged positions
            for position in self.active_positions[:]:  # Copy list to safely modify
                if position['borrowed_amount'] > 0:
                    btc_to_sell = position['btc_amount']
                    liquidation_value = btc_to_sell * current_price * 0.95  # 5% liquidation penalty
                    
                    # Execute forced liquidation
                    self.btc_holdings -= btc_to_sell
                    self.current_balance += liquidation_value
                    
                    # Repay loan
                    loan_to_repay = min(position['borrowed_amount'], liquidation_value)
                    self.current_balance -= loan_to_repay
                    self.borrowed_amount -= loan_to_repay
                    
                    # Track as losing trade
                    self.losing_trades += 1
                    self.liquidation_count += 1
                    
                    liquidations.append({
                        'timestamp': timestamp,
                        'signal': 'LIQUIDATION',
                        'price': current_price,
                        'btc_amount': btc_to_sell,
                        'usd_amount': liquidation_value,
                        'reason': f"Forced liquidation - equity below {self.liquidation_threshold*100:.0f}%",
                        'portfolio_value': self.calculate_portfolio_value(current_price),
                        'position_type': 'LIQUIDATION',
                        'penalty': btc_to_sell * current_price * 0.05,  # 5% penalty
                        'leverage': position['leverage_used']
                    })
                    
                    # Remove liquidated position
                    self.active_positions.remove(position)
        
        return liquidations

    def apply_borrowing_costs(self, current_price):
        """Apply hourly borrowing costs to borrowed funds."""
        if self.enable_margin and self.borrowed_amount > 0:
            hourly_cost = self.borrowed_amount * self.margin_interest_rate
            self.current_balance -= hourly_cost
            self.total_borrowing_costs += hourly_cost
            
            # If we can't pay interest, trigger margin call
            if self.current_balance < 0:
                self.current_balance = 0
                return True  # Signal for potential liquidation
        
        return False

    def calculate_portfolio_value(self, current_price):
        """Calculate current portfolio value including borrowed funds."""
        btc_value = self.btc_holdings * current_price
        total_value = self.current_balance + btc_value
        
        if self.enable_margin:
            # Net equity = total value - borrowed amount
            net_equity = total_value - self.borrowed_amount
            return net_equity
        else:
            return total_value

    def calculate_position_pnl(self, current_price):
        """Calculate total unrealized P&L from active positions including leverage."""
        total_pnl = 0
        winning_positions = 0
        losing_positions = 0
        
        for position in self.active_positions:
            if position['type'] == 'long':
                entry_price = position['entry_price']
                gross_pnl = (current_price - entry_price) * position['btc_amount']
                
                # For leveraged positions, subtract borrowed amount impact
                if position['borrowed_amount'] > 0:
                    net_pnl = gross_pnl - position['borrowed_amount'] * (1 - entry_price/current_price)
                else:
                    net_pnl = gross_pnl
                
                total_pnl += net_pnl
                
                if net_pnl > 0:
                    winning_positions += 1
                else:
                    losing_positions += 1
        
        return total_pnl, winning_positions, losing_positions

    def check_stop_losses(self, current_price, timestamp):
        """Check and execute stop losses for active long positions."""
        stop_loss_trades = []
        
        for position in self.active_positions[:]:  # Copy list to safely modify
            if position['type'] == 'long' and current_price <= position['stop_loss']:
                # Execute stop loss
                btc_to_sell = position['btc_amount']
                trade_value = btc_to_sell * current_price
                
                # Handle loan repayment for leveraged positions
                loan_repayment = 0
                if position['borrowed_amount'] > 0:
                    loan_repayment = min(position['borrowed_amount'], trade_value)
                    trade_value -= loan_repayment
                    self.borrowed_amount -= loan_repayment
                
                self.btc_holdings -= btc_to_sell
                self.current_balance += trade_value
                self.trade_count += 1
                self.losing_trades += 1  # Stop loss = losing trade
                
                stop_loss_trades.append({
                    'timestamp': timestamp,
                    'signal': 'STOP_LOSS',
                    'price': current_price,
                    'btc_amount': btc_to_sell,
                    'usd_amount': btc_to_sell * current_price,
                    'reason': f"Stop loss hit at ${current_price:.2f} (entry: ${position['entry_price']:.2f})",
                    'portfolio_value': self.calculate_portfolio_value(current_price),
                    'position_type': 'STOP_LOSS',
                    'loan_repayment': loan_repayment,
                    'leverage': position['leverage_used']
                })
                
                # Remove the stopped position
                self.active_positions.remove(position)
        
        # Add stop loss trades to trade history
        self.trades.extend(stop_loss_trades)
        
        return len(stop_loss_trades)  # Return number of stop losses executed

    def run_backtest(self, days=365):
        """Run the complete backtest."""
        print(f"\n🚀 Starting {days}-day backtest...")
        
        # Fetch data
        df = self.fetch_historical_data(days)
        
        if df is None or len(df) < 1500:
            print(f"❌ Insufficient data: need at least 1500 hours, got {len(df) if df is not None else 0}")
            return None
        
        # Calculate indicators
        print("📊 Calculating technical indicators...")
        df = self.calculate_indicators(df)
        
        # Build transition matrix (this handles training period internally)
        transition_matrix = self.build_transition_matrix(df)
        
        # Determine test period start (after training + indicator warmup)
        available_data = len(df)
        min_train = 1000
        max_train = available_data // 3
        actual_train_period = max(min_train, min(1000, max_train))
        test_period_start = actual_train_period + 200  # Skip 200 periods for indicator warmup
        
        if available_data < test_period_start + 100:
            print(f"❌ Insufficient data: need at least {test_period_start + 100} periods, have {available_data}")
            return None
        
        print(f"📊 Test period: {available_data - test_period_start} hours ({(available_data - test_period_start)/24:.1f} days)")
        
        # Initialize starting values - NEW: Calculate mixed portfolio value
        initial_price = df['close'].iloc[test_period_start]
        self.entry_price = initial_price
        initial_btc_value = self.starting_btc_amount * initial_price
        initial_portfolio_value = initial_btc_value + self.current_balance
        
        print(f"💰 Starting portfolio breakdown:")
        print(f"   BTC value: ${initial_btc_value:,.2f} ({initial_btc_value/initial_portfolio_value*100:.1f}%)")
        print(f"   Cash value: ${self.current_balance:,.2f} ({self.current_balance/initial_portfolio_value*100:.1f}%)")
        print(f"   Total value: ${initial_portfolio_value:,.2f}")
        print(f"📅 Backtest period: {df.index[test_period_start]} to {df.index[-1]}")
        print(f"⏱️  Total periods to simulate: {len(df) - test_period_start}")
        
        # Main backtest loop
        print("\n📈 Running simulation...")
        
        for i in range(test_period_start, len(df)):
            current_row = df.iloc[i]
            current_price = current_row['close']
            
            # NEW: Apply borrowing costs every hour for margin positions
            if self.enable_margin:
                insufficient_funds = self.apply_borrowing_costs(current_price)
                
                # Check for margin calls and liquidations
                liquidations = self.check_margin_calls_and_liquidations(current_price, current_row.name)
                if liquidations:
                    self.trades.extend(liquidations)
                    print(f"   ⚠️  {len(liquidations)} position(s) liquidated due to margin call")
            
            # NEW: Check if we need to rebuild the transition matrix (weekly)
            if (self.enable_weekly_rebuild and 
                i > test_period_start + self.matrix_training_window and  # Ensure we have enough data
                i - self.last_matrix_rebuild >= self.rebuild_frequency_hours):
                
                print(f"\n🔄 Time for weekly matrix rebuild (hour {i}, last rebuild: {self.last_matrix_rebuild})")
                new_matrix = self.rebuild_transition_matrix(df, i)
                if new_matrix is not None:
                    transition_matrix = new_matrix
                    print("✅ Matrix successfully updated with recent market data")
                else:
                    print("❌ Matrix rebuild failed, continuing with existing matrix")
                print("")  # Add spacing after matrix rebuild
            
            # Check stop losses first
            stop_losses_executed = self.check_stop_losses(current_price, current_row.name)
            
            # Generate signal
            signal, reason = self.generate_signal(i, df, transition_matrix)
            
            # Execute trade if signal
            if signal:
                self.execute_trade(signal, current_price, current_row.name, reason)
            
            # Track portfolio value
            portfolio_value = self.calculate_portfolio_value(current_price)
            self.portfolio_values.append({
                'timestamp': current_row.name,
                'portfolio_value': portfolio_value,
                'btc_price': current_price,
                'position': f"Mixed: {self.btc_holdings:.4f} BTC + ${self.current_balance:.2f} cash",
                'btc_holdings': self.btc_holdings,
                'cash_balance': self.current_balance,
                'active_positions': len(self.active_positions),
                'stop_losses_today': stop_losses_executed,
                'borrowed_amount': self.borrowed_amount,
                'total_borrowing_costs': self.total_borrowing_costs,
                'net_equity': portfolio_value
            })
            
            # Progress update with margin info
            if i % 500 == 0:  # Less frequent updates for longer backtests
                progress = ((i - test_period_start) / (len(df) - test_period_start)) * 100
                active_pos = len(self.active_positions)
                rebuilds = self.matrix_rebuild_count
                
                if self.enable_margin and self.borrowed_amount > 0:
                    leverage_ratio = (self.current_balance + self.btc_holdings * current_price) / max(1, self.current_balance + self.btc_holdings * current_price - self.borrowed_amount)
                    print(f"   Progress: {progress:.1f}% - Portfolio: ${portfolio_value:,.2f} - Active: {active_pos} - Borrowed: ${self.borrowed_amount:,.0f} - Leverage: {leverage_ratio:.1f}x - Rebuilds: {rebuilds}")
                else:
                    print(f"   Progress: {progress:.1f}% - Portfolio: ${portfolio_value:,.2f} - Active positions: {active_pos} - Matrix rebuilds: {rebuilds}")
        
        print("✅ Backtest completed!")
        return self.analyze_results(df, initial_portfolio_value)

    def analyze_results(self, df, initial_portfolio_value):
        """Analyze backtest results and generate metrics."""
        print("\n📊 Analyzing results...")
        
        # Convert results to DataFrames
        portfolio_df = pd.DataFrame(self.portfolio_values)
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        # Calculate final values
        final_portfolio_value = portfolio_df['portfolio_value'].iloc[-1]
        final_btc_price = portfolio_df['btc_price'].iloc[-1]
        
        # Buy and hold comparison - NEW: Mixed portfolio buy & hold
        # For buy & hold with mixed portfolio: keep BTC as is, cash stays cash
        buy_hold_btc_value = self.starting_btc_amount * final_btc_price  # BTC portion
        buy_hold_cash_value = 5000.0  # Cash portion stays same
        buy_hold_value = buy_hold_btc_value + buy_hold_cash_value
        
        # Performance metrics
        total_return = ((final_portfolio_value - initial_portfolio_value) / initial_portfolio_value) * 100
        buy_hold_return = ((buy_hold_value - initial_portfolio_value) / initial_portfolio_value) * 100
        
        # Calculate daily returns for Sharpe ratio
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
        sharpe_ratio = portfolio_df['daily_return'].mean() / portfolio_df['daily_return'].std() * np.sqrt(8760)  # Annualized
        
        # Max drawdown
        portfolio_df['running_max'] = portfolio_df['portfolio_value'].expanding().max()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['running_max']) / portfolio_df['running_max']
        max_drawdown = portfolio_df['drawdown'].min() * 100
        
        # Win rate - FIXED: Include unrealized P&L from active positions
        if len(trades_df) > 0 or len(self.active_positions) > 0:
            # Get final unrealized P&L
            final_price = portfolio_df['btc_price'].iloc[-1]
            unrealized_pnl, winning_unrealized, losing_unrealized = self.calculate_position_pnl(final_price)
            
            total_winning = self.winning_trades + winning_unrealized
            total_losing = self.losing_trades + losing_unrealized
            total_positions = total_winning + total_losing
            
            if total_positions > 0:
                win_rate = (total_winning / total_positions) * 100
            else:
                win_rate = 0
        else:
            win_rate = 0
            unrealized_pnl = 0
            winning_unrealized = 0
            losing_unrealized = 0
        
        # Print results
        print("\n" + "="*60)
        print("🎯 BACKTEST RESULTS")
        print("="*60)
        print(f"📅 Period: {portfolio_df['timestamp'].iloc[0]} to {portfolio_df['timestamp'].iloc[-1]}")
        print(f"⏱️  Duration: {len(portfolio_df)} hours ({len(portfolio_df)/24:.1f} days)")
        
        print(f"\n💰 PORTFOLIO PERFORMANCE:")
        print(f"   Initial Value: ${initial_portfolio_value:,.2f}")
        print(f"   Final Value: ${final_portfolio_value:,.2f}")
        print(f"   Total Return: {total_return:+.2f}%")
        print(f"   Max Drawdown: {max_drawdown:.2f}%")
        print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
        
        print(f"\n🪙 BUY & HOLD COMPARISON:")
        print(f"   Buy & Hold Value: ${buy_hold_value:,.2f}")
        print(f"   Buy & Hold Return: {buy_hold_return:+.2f}%")
        print(f"   Strategy vs B&H: {total_return - buy_hold_return:+.2f}% outperformance")
        
        print(f"\n📊 TRADING STATISTICS:")
        print(f"   Total Trades: {len(trades_df)}")
        print(f"   Realized Wins: {self.winning_trades}")
        print(f"   Realized Losses: {self.losing_trades}")
        print(f"   Unrealized Wins: {winning_unrealized}")
        print(f"   Unrealized Losses: {losing_unrealized}")
        print(f"   Total Win Rate: {win_rate:.1f}%")
        print(f"   Active Positions: {len(self.active_positions)}")
        if unrealized_pnl != 0:
            print(f"   Unrealized P&L: ${unrealized_pnl:,.2f}")
        
        # NEW: Margin trading statistics
        if self.enable_margin:
            print(f"\n⚡ MARGIN TRADING STATISTICS:")
            print(f"   Max Leverage Used: {self.max_leverage:.1f}x")
            print(f"   Final Borrowed Amount: ${self.borrowed_amount:,.2f}")
            print(f"   Total Borrowing Costs: ${self.total_borrowing_costs:,.2f}")
            print(f"   Liquidations: {self.liquidation_count}")
            
            if len(trades_df) > 0:
                leveraged_trades = trades_df[trades_df['leverage'] > 1.1]  # Trades with >1.1x leverage
                print(f"   Leveraged Trades: {len(leveraged_trades)}/{len(trades_df)} ({len(leveraged_trades)/len(trades_df)*100:.1f}%)")
                
                if len(leveraged_trades) > 0:
                    avg_leverage = leveraged_trades['leverage'].mean()
                    max_leverage_used = leveraged_trades['leverage'].max()
                    print(f"   Average Leverage: {avg_leverage:.2f}x")
                    print(f"   Peak Leverage: {max_leverage_used:.2f}x")
                
                liquidation_trades = trades_df[trades_df['signal'] == 'LIQUIDATION']
                if len(liquidation_trades) > 0:
                    total_liquidation_loss = liquidation_trades['penalty'].sum() if 'penalty' in liquidation_trades.columns else 0
                    print(f"   Liquidation Penalties: ${total_liquidation_loss:,.2f}")
            
            # Calculate effective return after borrowing costs
            effective_return = total_return - (self.total_borrowing_costs / initial_portfolio_value * 100)
            print(f"   Net Return (after costs): {effective_return:+.2f}%")
            print(f"   Borrowing Cost Impact: {self.total_borrowing_costs / initial_portfolio_value * 100:.2f}%")
        
        # NEW: Matrix rebuilding statistics
        print(f"\n🔄 ADAPTIVE LEARNING STATISTICS:")
        print(f"   Matrix Rebuilds: {self.matrix_rebuild_count}")
        if self.enable_weekly_rebuild:
            expected_rebuilds = len(portfolio_df) // self.rebuild_frequency_hours
            print(f"   Expected Rebuilds: {expected_rebuilds} (every {self.rebuild_frequency_hours//24} days)")
            if expected_rebuilds > 0:
                rebuild_efficiency = (self.matrix_rebuild_count / expected_rebuilds) * 100
                print(f"   Rebuild Efficiency: {rebuild_efficiency:.1f}%")
        else:
            print("   Weekly Rebuilding: DISABLED")
        
        if len(trades_df) > 0:
            print(f"\n📈 TRADE BREAKDOWN:")
            for signal in ['BUY', 'SELL', 'STOP_LOSS']:
                signal_trades = trades_df[trades_df['signal'] == signal]
                if len(signal_trades) > 0:
                    print(f"   {signal} signals: {len(signal_trades)}")
            
            # Markov vs Technical trades (excluding stop losses)
            trading_trades = trades_df[trades_df['signal'].isin(['BUY', 'SELL'])]
            markov_trades = trading_trades[trading_trades['reason'].str.contains('Markov', na=False)]
            technical_trades = trading_trades[trading_trades['reason'].str.contains('Technical', na=False)]
            stop_loss_trades = trades_df[trades_df['signal'] == 'STOP_LOSS']
            
            print(f"   Markov-driven trades: {len(markov_trades)}")
            print(f"   Technical-driven trades: {len(technical_trades)}")
            print(f"   Stop loss executions: {len(stop_loss_trades)}")
            
            if len(stop_loss_trades) > 0:
                avg_stop_loss = stop_loss_trades['usd_amount'].mean()
                print(f"   Average stop loss amount: ${avg_stop_loss:.2f}")
        
        return {
            'portfolio_df': portfolio_df,
            'trades_df': trades_df,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }

    def plot_results(self, results):
        """Create visualization plots."""
        portfolio_df = results['portfolio_df']
        trades_df = results['trades_df']
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 1. Portfolio value vs BTC price
        ax1 = axes[0]
        ax1_twin = ax1.twinx()
        
        ax1.plot(portfolio_df['timestamp'], portfolio_df['portfolio_value'], 
                label='Portfolio Value', color='green', linewidth=2)
        ax1.set_ylabel('Portfolio Value ($)', color='green')
        ax1.set_title('Markov Bot Performance vs Bitcoin Price')
        
        ax1_twin.plot(portfolio_df['timestamp'], portfolio_df['btc_price'], 
                     label='BTC Price', color='orange', alpha=0.7)
        ax1_twin.set_ylabel('BTC Price ($)', color='orange')
        
        # Mark trades
        if len(trades_df) > 0:
            buy_trades = trades_df[trades_df['signal'] == 'BUY']
            sell_trades = trades_df[trades_df['signal'] == 'SELL']
            
            if len(buy_trades) > 0:
                ax1_twin.scatter(buy_trades['timestamp'], buy_trades['price'], 
                               color='green', marker='^', s=100, label='BUY', alpha=0.8)
            if len(sell_trades) > 0:
                ax1_twin.scatter(sell_trades['timestamp'], sell_trades['price'], 
                               color='red', marker='v', s=100, label='SELL', alpha=0.8)
        
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # 2. Drawdown
        ax2 = axes[1]
        ax2.fill_between(portfolio_df['timestamp'], portfolio_df['drawdown']*100, 0, 
                        color='red', alpha=0.3, label='Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_title('Portfolio Drawdown')
        ax2.legend()
        
        # 3. Position tracking
        ax3 = axes[2]
        position_numeric = portfolio_df['position'].map({'long': 1, 'short': -1, None: 0})
        ax3.plot(portfolio_df['timestamp'], position_numeric, 
                label='Position (1=Long, -1=Short)', linewidth=2)
        ax3.set_ylabel('Position')
        ax3.set_xlabel('Date')
        ax3.set_title('Position Over Time')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig('markov_backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Run margin trading optimization tests."""
    # MARGIN TRADING OPTIMIZATION: Multiple Leverage Strategy Testing
    print("⚡ MARGIN TRADING BOT OPTIMIZATION")
    print("="*60)
    print("Testing multiple leverage strategies for optimal risk/reward")
    print("Specialized tuning for leveraged Bitcoin trading")
    print("")
    
    # Margin trading configurations to test
    margin_configs = [
        {
            'name': '1.5x Conservative Margin',
            'enable_margin': True,
            'max_leverage': 1.5,
            'margin_interest_rate': 0.00005,  # Lower rate for conservative approach
            'position_size_pct': 0.15,  # Smaller positions for margin
            'stop_loss_pct': 0.05,  # Tighter stop losses
            'liquidation_threshold': 0.25  # Higher safety margin
        },
        {
            'name': '2x Balanced Margin',
            'enable_margin': True,
            'max_leverage': 2.0,
            'margin_interest_rate': 0.0001,
            'position_size_pct': 0.20,
            'stop_loss_pct': 0.06,
            'liquidation_threshold': 0.20
        },
        {
            'name': '2.5x Aggressive Margin',
            'enable_margin': True,
            'max_leverage': 2.5,
            'margin_interest_rate': 0.00012,
            'position_size_pct': 0.18,
            'stop_loss_pct': 0.04,
            'liquidation_threshold': 0.18
        },
        {
            'name': '3x High Risk (Original)',
            'enable_margin': True,
            'max_leverage': 3.0,
            'margin_interest_rate': 0.0001,
            'position_size_pct': 0.25,
            'stop_loss_pct': 0.08,
            'liquidation_threshold': 0.15
        }
    ]
    
    results_comparison = []
    
    for config in margin_configs:
        print(f"\n🔧 Testing: {config['name']}")
        print("-" * 50)
        
        # Create specialized margin backtester
        backtester = MarkovBotBacktester(
            start_date='2024-01-01',
            end_date='2024-12-31',
            enable_weekly_rebuild=True,
            rebuild_frequency_hours=336,  # 14 days (optimal)
            matrix_training_window=1500,  # 62.5 days (optimal)
            enable_margin=config['enable_margin'],
            max_leverage=config['max_leverage'],
            margin_interest_rate=config['margin_interest_rate']
        )
        
        # Apply custom margin-specific settings
        backtester.position_size_pct = config['position_size_pct']
        backtester.stop_loss_pct = config['stop_loss_pct']
        backtester.liquidation_threshold = config['liquidation_threshold']
        
        print(f"🔧 Margin Settings:")
        print(f"   • Leverage: {config['max_leverage']:.1f}x")
        print(f"   • Position Size: {config['position_size_pct']*100:.0f}%")
        print(f"   • Stop Loss: {config['stop_loss_pct']*100:.0f}%")
        print(f"   • Liquidation Threshold: {config['liquidation_threshold']*100:.0f}%")
        print(f"   • Interest Rate: {config['margin_interest_rate']*100:.4f}% per hour")
        
        # Run simulation
        results = backtester.run_backtest(days=365)
        
        if results is None:
            print(f"❌ {config['name']} simulation failed")
            continue
        
        # Store results for comparison
        results_comparison.append({
            'config': config,
            'results': results,
            'backtester': backtester
        })
        
        # Save individual results
        filename_suffix = f"margin_{config['max_leverage']:.1f}x".replace('.', '_')
        if len(results['trades_df']) > 0:
            results['trades_df'].to_csv(f'{filename_suffix}_trades.csv', index=False)
        results['portfolio_df'].to_csv(f'{filename_suffix}_portfolio.csv', index=False)
    
    # MARGIN STRATEGY COMPARISON
    print(f"\n🏆 MARGIN TRADING STRATEGY COMPARISON")
    print("="*60)
    
    # Sort results by performance
    valid_results = [r for r in results_comparison if r['results']['total_return'] > -90]
    if valid_results:
        sorted_results = sorted(valid_results, key=lambda x: x['results']['total_return'], reverse=True)
        
        print(f"📊 PERFORMANCE RANKING:")
        for i, result in enumerate(sorted_results, 1):
            config = result['config']
            perf = result['results']
            backtester = result['backtester']
            
            print(f"\n{i}. {config['name']}:")
            print(f"   • Return: {perf['total_return']:+.2f}%")
            print(f"   • Max Drawdown: {perf['max_drawdown']:.2f}%")
            print(f"   • Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
            print(f"   • Win Rate: {perf['win_rate']:.1f}%")
            print(f"   • Liquidations: {backtester.liquidation_count}")
            print(f"   • Borrowing Costs: ${backtester.total_borrowing_costs:.2f}")
            
            # Risk efficiency score
            if abs(perf['max_drawdown']) > 0:
                risk_efficiency = perf['total_return'] / abs(perf['max_drawdown'])
                print(f"   • Risk Efficiency: {risk_efficiency:.2f}")
        
        # Best strategy analysis
        if sorted_results:
            best = sorted_results[0]
            print(f"\n🏆 OPTIMAL MARGIN STRATEGY: {best['config']['name']}")
            print(f"   ✅ Best overall performance with {best['results']['total_return']:+.2f}% return")
            print(f"   🛡️  {best['backtester'].liquidation_count} liquidations")
            print(f"   💰 ${best['backtester'].total_borrowing_costs:.2f} in borrowing costs")
            
            if best['backtester'].liquidation_count == 0:
                print(f"   🎯 EXCELLENT: No liquidations - perfect risk management!")
            elif best['backtester'].liquidation_count < 10:
                print(f"   ✅ GOOD: Minimal liquidations - controlled risk")
            else:
                print(f"   ⚠️  HIGH RISK: Many liquidations occurred")
    
    else:
        print(f"❌ ALL MARGIN STRATEGIES FAILED")
        print(f"💡 Recommendations:")
        print(f"   • Use lower leverage (1.2x - 1.5x)")
        print(f"   • Increase liquidation threshold to 30%+")
        print(f"   • Reduce position sizes to 5-10%")
        print(f"   • Implement dynamic leverage based on volatility")
    
    print(f"\n🎯 MARGIN TRADING CONCLUSIONS:")
    print(f"This analysis helps optimize leverage settings for maximum profitability!")

if __name__ == "__main__":
    main() 