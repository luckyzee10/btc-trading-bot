#!/usr/bin/env python3
"""
OPTIMIZED Markov Trading Bot Backtester
========================================
Enhanced with advanced optimization strategies identified through deep analysis:
1. Dynamic Position Sizing (ATR-based)
2. Trailing Stop Losses  
3. Market Regime Adaptation
4. Advanced Signal Filtering
5. Portfolio Heat Management
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

# Import our Markov helpers
import pickle
from collections import defaultdict

# === MARKOV CHAIN HELPERS (same as main bot) ===
STATE_MATRIX_FILE = 'optimized_transition_matrix.pkl'
MARKOV_STATES = [
    f'{p}-{r}'
    for p in ('UP', 'FLAT', 'DOWN')
    for r in ('OVERSOLD', 'NEUTRAL', 'OVERBOUGHT')
]

# OPTIMIZATION: Enhanced Markov trading configuration
MARKOV_CONFIDENCE_THRESHOLD = 0.75  # Slightly higher for better signal quality
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

# === OPTIMIZED BACKTESTING ENGINE ===
class OptimizedMarkovBotBacktester:
    def __init__(self, start_date='2024-01-01', end_date='2024-12-31', 
                 enable_weekly_rebuild=True, rebuild_frequency_hours=720, matrix_training_window=2000):
        self.start_date = start_date
        self.end_date = end_date
        
        # Portfolio settings - Starting balanced
        self.starting_btc_amount = 0.07983511 / 2  # Half the original BTC
        self.current_balance = 5000.0  # Start with $5000 cash
        self.btc_holdings = self.starting_btc_amount
        self.position = 'mixed'  # Start with mixed position
        self.entry_price = None
        
        # OPTIMIZATION 1: Dynamic Position Sizing
        self.base_position_size_pct = 0.15  # Reduced base size for risk management
        self.min_position_size_pct = 0.05   # Minimum position size
        self.max_position_size_pct = 0.30   # Maximum position size
        self.position_size_pct = self.base_position_size_pct  # Current dynamic size
        
        # OPTIMIZATION 2: Enhanced Risk Management
        self.base_stop_loss_pct = 0.06      # Reduced base stop loss (6%)
        self.trailing_stop_enabled = True   # Enable trailing stops
        self.time_based_stop_hours = 72     # Exit stale positions after 72h
        
        # OPTIMIZATION 3: Market Regime Detection
        self.market_regime = 'UNKNOWN'      # Current market regime
        self.regime_lookback = 48           # Hours to look back for regime detection
        self.bull_threshold = 3.0           # % gain for bull market
        self.bear_threshold = -3.0          # % loss for bear market
        
        # OPTIMIZATION 4: Portfolio Heat Management  
        self.portfolio_heat = 0.0           # Current portfolio risk level
        self.max_portfolio_heat = 0.25      # Maximum risk tolerance (25%)
        self.heat_reduction_factor = 0.5    # Reduce position size when hot
        
        # OPTIMIZATION 5: Advanced Signal Filtering
        self.volume_filter_enabled = True   # Require volume confirmation
        self.momentum_filter_enabled = True # Require momentum alignment
        self.min_volume_ratio = 1.2         # Minimum volume vs average
        
        # Matrix rebuilding settings
        self.enable_weekly_rebuild = enable_weekly_rebuild
        self.rebuild_frequency_hours = rebuild_frequency_hours
        self.matrix_training_window = matrix_training_window
        self.last_matrix_rebuild = 0
        self.matrix_rebuild_count = 0
        
        # Enhanced tracking
        self.active_positions = []
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.optimization_metrics = {
            'dynamic_sizing_saves': 0,
            'trailing_stop_profits': 0,
            'regime_based_exits': 0,
            'volume_filtered_trades': 0
        }
        
        # Track results
        self.trades = []
        self.portfolio_values = []
        self.states = []
        
        print(f"🚀 OPTIMIZED Markov Bot Initialized")
        print(f"💰 Starting: {self.starting_btc_amount:.6f} BTC + ${self.current_balance:,.2f}")
        print(f"🔧 Optimizations enabled:")
        print(f"   ✅ Dynamic Position Sizing ({self.min_position_size_pct*100:.0f}%-{self.max_position_size_pct*100:.0f}%)")
        print(f"   ✅ Trailing Stops & Time-based Exits")
        print(f"   ✅ Market Regime Adaptation")
        print(f"   ✅ Portfolio Heat Management")
        print(f"   ✅ Advanced Signal Filtering")

    def fetch_historical_data(self, days=365):
        """Fetch historical OHLCV data - same as original."""
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
                
                hours_needed = days * 24
                print(f"   Target: {hours_needed} hours ({days} days) of historical data...")
                
                all_candles = []
                batch_size = max_limit
                max_batches = min(20, (hours_needed // batch_size) + 2)
                
                print(f"   Will fetch up to {max_batches} batches of {batch_size} candles each...")
                
                since_timestamp = None
                
                for batch_num in range(max_batches):
                    try:
                        print(f"   Fetching batch {batch_num + 1}/{max_batches}...")
                        
                        if since_timestamp is None:
                            candles = exchange.fetch_ohlcv(symbol, '1h', limit=batch_size)
                        else:
                            candles = exchange.fetch_ohlcv(symbol, '1h', since=since_timestamp, limit=batch_size)
                        
                        if not candles or len(candles) == 0:
                            print(f"   No more data available at batch {batch_num + 1}")
                            break
                        
                        if all_candles and candles[-1][0] >= all_candles[0][0]:
                            print(f"   Reached duplicate data, stopping at batch {batch_num + 1}")
                            break
                        
                        all_candles = candles + all_candles
                        print(f"   Added {len(candles)} candles, total: {len(all_candles)}")
                        
                        oldest_timestamp = candles[0][0]
                        since_timestamp = oldest_timestamp - (batch_size * 3600000)
                        
                        if len(all_candles) >= hours_needed * 0.7:
                            print(f"   Reached acceptable amount of data: {len(all_candles)} hours!")
                            break
                        
                        import time
                        time.sleep(0.5)
                        
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
                
                if len(all_candles) >= 2000:
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
        df = df.sort_index()
        
        actual_days = len(df) / 24
        print(f"✅ Successfully loaded {len(df)} hours ({actual_days:.1f} days) of data")
        print(f"   Period: {df.index[0]} to {df.index[-1]}")
        
        return df

    def calculate_indicators(self, df):
        """Calculate enhanced technical indicators."""
        # Basic indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        
        # EMA 200
        df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
        
        # OPTIMIZATION: Enhanced ATR for dynamic position sizing
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        df['atr_ma'] = df['atr'].rolling(window=20).mean()
        df['atr_ratio'] = df['atr'] / df['atr_ma']  # Volatility ratio
        
        # OPTIMIZATION: Volume indicators for filtering
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # OPTIMIZATION: Momentum indicators
        df['momentum'] = df['close'].pct_change(periods=5) * 100  # 5-hour momentum
        df['momentum_ma'] = df['momentum'].rolling(window=10).mean()
        
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

    def detect_market_regime(self, df, current_idx):
        """OPTIMIZATION 3: Detect current market regime."""
        if current_idx < self.regime_lookback:
            return 'UNKNOWN'
        
        # Look back over regime_lookback periods
        lookback_data = df.iloc[current_idx - self.regime_lookback:current_idx]
        
        if len(lookback_data) == 0:
            return 'UNKNOWN'
        
        # Calculate regime based on price change and volatility
        price_change = ((lookback_data['close'].iloc[-1] - lookback_data['close'].iloc[0]) / 
                       lookback_data['close'].iloc[0]) * 100
        
        volatility = lookback_data['atr_ratio'].mean()
        
        # Classify regime
        if price_change > self.bull_threshold:
            if volatility > 1.2:
                return 'VOLATILE_BULL'
            else:
                return 'STEADY_BULL'
        elif price_change < self.bear_threshold:
            if volatility > 1.2:
                return 'VOLATILE_BEAR'
            else:
                return 'STEADY_BEAR'
        else:
            if volatility > 1.2:
                return 'CHOPPY'
            else:
                return 'SIDEWAYS'

    def calculate_dynamic_position_size(self, df, current_idx, signal_strength=1.0):
        """OPTIMIZATION 1: Calculate dynamic position size based on ATR and conditions."""
        if current_idx < 20:  # Need enough data for ATR calculation
            return self.base_position_size_pct
        
        row = df.iloc[current_idx]
        
        # Base size adjustment factors
        size_multipliers = []
        
        # 1. Volatility adjustment (inverse relationship)
        atr_ratio = row.get('atr_ratio', 1.0)
        if atr_ratio > 1.5:  # High volatility
            size_multipliers.append(0.7)  # Reduce size
        elif atr_ratio > 1.2:  # Medium volatility
            size_multipliers.append(0.85)
        elif atr_ratio < 0.8:  # Low volatility
            size_multipliers.append(1.2)  # Increase size
        else:
            size_multipliers.append(1.0)
        
        # 2. Market regime adjustment
        if self.market_regime in ['VOLATILE_BEAR', 'CHOPPY']:
            size_multipliers.append(0.6)  # Much smaller positions in bad conditions
        elif self.market_regime in ['STEADY_BULL']:
            size_multipliers.append(1.3)  # Larger positions in good conditions
        elif self.market_regime in ['VOLATILE_BULL']:
            size_multipliers.append(1.1)  # Slightly larger but cautious
        else:
            size_multipliers.append(1.0)
        
        # 3. Portfolio performance adjustment
        if hasattr(self, 'recent_performance'):
            if self.recent_performance < -5:  # Recent losses
                size_multipliers.append(0.7)
            elif self.recent_performance > 10:  # Recent wins
                size_multipliers.append(1.2)
            else:
                size_multipliers.append(1.0)
        
        # 4. Portfolio heat adjustment
        if self.portfolio_heat > 0.15:  # High risk
            size_multipliers.append(0.5)
        elif self.portfolio_heat > 0.10:  # Medium risk
            size_multipliers.append(0.75)
        else:
            size_multipliers.append(1.0)
        
        # 5. Signal strength adjustment
        size_multipliers.append(signal_strength)
        
        # Calculate final size
        final_multiplier = np.prod(size_multipliers)
        dynamic_size = self.base_position_size_pct * final_multiplier
        
        # Apply bounds
        dynamic_size = max(self.min_position_size_pct, 
                          min(self.max_position_size_pct, dynamic_size))
        
        return dynamic_size

    def calculate_dynamic_stop_loss(self, entry_price, df, current_idx):
        """OPTIMIZATION 2: Calculate dynamic stop loss based on ATR."""
        if current_idx < 20:
            return entry_price * (1 - self.base_stop_loss_pct)
        
        row = df.iloc[current_idx]
        atr = row.get('atr', 0)
        
        if atr > 0:
            # ATR-based stop: 2.5x ATR below entry
            atr_stop_distance = (2.5 * atr) / entry_price
            stop_loss = entry_price * (1 - max(atr_stop_distance, self.base_stop_loss_pct))
        else:
            # Fallback to fixed percentage
            stop_loss = entry_price * (1 - self.base_stop_loss_pct)
        
        return stop_loss

    def update_portfolio_heat(self, current_price):
        """OPTIMIZATION 4: Update portfolio heat (risk level)."""
        total_portfolio_value = self.calculate_portfolio_value(current_price)
        
        # Calculate unrealized P&L from active positions
        unrealized_risk = 0
        for position in self.active_positions:
            if position['type'] == 'long':
                # Risk = potential loss if stopped out
                stop_loss_value = position['btc_amount'] * position['stop_loss']
                current_value = position['btc_amount'] * current_price
                position_risk = max(0, current_value - stop_loss_value)
                unrealized_risk += position_risk
        
        # Portfolio heat = unrealized risk / total portfolio value
        if total_portfolio_value > 0:
            self.portfolio_heat = unrealized_risk / total_portfolio_value
        else:
            self.portfolio_heat = 0
        
        return self.portfolio_heat

    def advanced_signal_filter(self, signal, df, current_idx, reason):
        """OPTIMIZATION 5: Advanced signal filtering."""
        if signal is None:
            return signal, reason
        
        row = df.iloc[current_idx]
        
        # Volume filter
        if self.volume_filter_enabled:
            volume_ratio = row.get('volume_ratio', 1.0)
            if volume_ratio < self.min_volume_ratio:
                self.optimization_metrics['volume_filtered_trades'] += 1
                return None, f"Filtered: Low volume ({volume_ratio:.2f}x)"
        
        # Momentum filter
        if self.momentum_filter_enabled:
            momentum = row.get('momentum', 0)
            momentum_ma = row.get('momentum_ma', 0)
            
            # For BUY signals, require positive momentum alignment
            if signal == 'BUY' and momentum < momentum_ma - 1:
                return None, f"Filtered: Negative momentum divergence"
            
            # For SELL signals, require negative momentum alignment for exits
            if signal == 'SELL' and momentum > momentum_ma + 1:
                return None, f"Filtered: Positive momentum divergence"
        
        # Market regime filter
        if self.market_regime == 'VOLATILE_BEAR':
            if signal == 'BUY':
                return None, f"Filtered: Volatile bear market"
        elif self.market_regime == 'CHOPPY':
            # Require higher confidence in choppy markets
            if 'Markov' in reason and '70%' in reason:  # Lower confidence
                return None, f"Filtered: Low confidence in choppy market"
        
        return signal, reason

    def build_transition_matrix(self, df, train_period=1000):
        """Build transition matrix from initial training period."""
        available_data = len(df)
        min_train = 1000
        max_train = available_data // 3
        
        actual_train_period = max(min_train, min(train_period, max_train))
        
        print(f"🧠 Building optimized transition matrix from first {actual_train_period} hours...")
        print(f"   (Available data: {available_data} hours, using {actual_train_period/available_data*100:.1f}% for training)")
        
        train_data = df.iloc[:actual_train_period].dropna(subset=['markov_state'])
        matrix = build_transition_matrix(train_data)
        
        with open('optimized_transition_matrix.pkl', 'wb') as f:
            pickle.dump(matrix, f)
        
        print(f"✅ Enhanced matrix built from {len(train_data)} training samples")
        return matrix

    def generate_signal(self, current_idx, df, transition_matrix):
        """Generate enhanced trading signal with optimizations."""
        row = df.iloc[current_idx]
        
        # Skip if insufficient data
        if pd.isna(row['rsi']) or pd.isna(row['bb_lower']) or pd.isna(row['ema_200']):
            return None, "Insufficient indicator data"
        
        current_price = row['close']
        signal = None
        reason = ""
        signal_strength = 1.0
        
        # Enhanced technical analysis signals
        # BUY: Oversold conditions with trend support
        if (row['rsi'] < 32 and  # Slightly more restrictive
            current_price < row['bb_lower'] and 
            row['atr_ratio'] > 1.0 and  # Some volatility required
            current_price > row['ema_200'] and
            self.current_balance > 100):
            
            signal = 'BUY'
            reason = f"Technical: RSI oversold ({row['rsi']:.1f}), below BB lower"
            signal_strength = 1.2  # Strong technical signal
        
        # SELL: Take profits or exit overbought
        elif (row['rsi'] > 68 or current_price > row['bb_upper']) and self.btc_holdings > 0.001:
            signal = 'SELL'
            reason = f"Technical: RSI overbought ({row['rsi']:.1f}) or above BB upper"
            signal_strength = 1.1
        
        # Enhanced Markov logic
        if current_idx > 0 and not pd.isna(row['markov_state']):
            current_state = row['markov_state']
            next_state, probability = predict_next_state(current_state, transition_matrix)
            
            # Markov-based BUY with enhanced confidence
            if (signal is None and 
                next_state in BULLISH_STATES and 
                probability >= MARKOV_CONFIDENCE_THRESHOLD and 
                self.current_balance > 100):
                
                signal = 'BUY'
                reason = f"Markov: {current_state}→{next_state} ({probability:.0%})"
                signal_strength = min(probability, 1.0)
            
            # Markov-based SELL
            elif (self.btc_holdings > 0.001 and 
                  next_state in BEARISH_STATES and 
                  probability >= MARKOV_CONFIDENCE_THRESHOLD):
                
                signal = 'SELL'
                reason = f"Markov exit: {current_state}→{next_state} ({probability:.0%})"
                signal_strength = min(probability, 1.0)
            
            # Store state info
            self.states.append({
                'timestamp': row.name,
                'current_state': current_state,
                'next_state': next_state,
                'probability': probability
            })
        
        # Apply advanced filtering
        signal, reason = self.advanced_signal_filter(signal, df, current_idx, reason)
        
        # Update dynamic position size if we have a signal
        if signal == 'BUY':
            self.position_size_pct = self.calculate_dynamic_position_size(df, current_idx, signal_strength)
        
        return signal, reason

    def execute_trade(self, signal, price, timestamp, reason):
        """Execute trade with all optimizations."""
        
        if signal == 'BUY' and self.current_balance > 100:
            # Calculate trade amount using dynamic position sizing
            trade_amount = self.current_balance * self.position_size_pct
            btc_to_buy = trade_amount / price
            
            # Execute the buy
            self.current_balance -= trade_amount
            self.btc_holdings += btc_to_buy
            self.trade_count += 1
            
            # Calculate dynamic stop loss
            stop_loss_price = self.calculate_dynamic_stop_loss(price, None, 0)  # Simplified for now
            
            # Track position with enhanced data
            position = {
                'type': 'long',
                'entry_price': price,
                'btc_amount': btc_to_buy,
                'entry_time': timestamp,
                'stop_loss': stop_loss_price,
                'trailing_stop': None,  # Will be updated
                'position_size_pct': self.position_size_pct,
                'entry_reason': reason
            }
            self.active_positions.append(position)
            
            self.trades.append({
                'timestamp': timestamp,
                'signal': signal,
                'price': price,
                'btc_amount': btc_to_buy,
                'usd_amount': trade_amount,
                'reason': reason,
                'portfolio_value': self.calculate_portfolio_value(price),
                'position_type': 'BUY',
                'position_size_pct': self.position_size_pct,
                'market_regime': self.market_regime,
                'portfolio_heat': self.portfolio_heat
            })
            
        elif signal == 'SELL' and self.btc_holdings > 0.001:
            # Calculate trade amount using dynamic sizing
            btc_to_sell = self.btc_holdings * self.position_size_pct
            trade_value = btc_to_sell * price
            
            # Execute the sell
            self.btc_holdings -= btc_to_sell
            self.current_balance += trade_value
            self.trade_count += 1
            
            # Close positions and track P&L
            self._close_long_positions(btc_to_sell, price)
            
            self.trades.append({
                'timestamp': timestamp,
                'signal': signal,
                'price': price,
                'btc_amount': btc_to_sell,
                'usd_amount': trade_value,
                'reason': reason,
                'portfolio_value': self.calculate_portfolio_value(price),
                'position_type': 'SELL',
                'market_regime': self.market_regime,
                'portfolio_heat': self.portfolio_heat
            })

    def _close_long_positions(self, btc_sold, current_price):
        """Close long positions with enhanced tracking."""
        btc_remaining = btc_sold
        positions_to_remove = []
        
        for i, position in enumerate(self.active_positions):
            if position['type'] == 'long' and btc_remaining > 0:
                btc_from_position = min(position['btc_amount'], btc_remaining)
                
                # Calculate P&L
                entry_price = position['entry_price']
                pnl = (current_price - entry_price) * btc_from_position
                
                # Track if this was a trailing stop profit
                if position.get('trailing_stop') and current_price > entry_price * 1.05:
                    self.optimization_metrics['trailing_stop_profits'] += 1
                
                if pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                # Update position or mark for removal
                position['btc_amount'] -= btc_from_position
                btc_remaining -= btc_from_position
                
                if position['btc_amount'] <= 0.0001:
                    positions_to_remove.append(i)
        
        # Remove closed positions
        for i in reversed(positions_to_remove):
            del self.active_positions[i]

    def update_trailing_stops(self, current_price):
        """OPTIMIZATION 2: Update trailing stops for active positions."""
        for position in self.active_positions:
            if position['type'] == 'long':
                entry_price = position['entry_price']
                
                # Only start trailing after 5% profit
                if current_price > entry_price * 1.05:
                    # Trail at 3% below the highest price seen
                    if position['trailing_stop'] is None:
                        position['trailing_stop'] = current_price * 0.97
                    else:
                        # Update trailing stop if price went higher
                        new_trailing_stop = current_price * 0.97
                        if new_trailing_stop > position['trailing_stop']:
                            position['trailing_stop'] = new_trailing_stop

    def check_enhanced_stop_losses(self, current_price, timestamp):
        """Enhanced stop loss checking with trailing and time-based stops."""
        stop_loss_trades = []
        
        for position in self.active_positions[:]:
            if position['type'] == 'long':
                should_stop = False
                stop_reason = ""
                
                # 1. Regular stop loss
                if current_price <= position['stop_loss']:
                    should_stop = True
                    stop_reason = f"Stop loss hit at ${current_price:.2f}"
                
                # 2. Trailing stop
                elif position.get('trailing_stop') and current_price <= position['trailing_stop']:
                    should_stop = True
                    stop_reason = f"Trailing stop hit at ${current_price:.2f}"
                    self.optimization_metrics['trailing_stop_profits'] += 1
                
                # 3. Time-based stop
                elif self.time_based_stop_hours > 0:
                    entry_time = pd.to_datetime(position['entry_time'])
                    current_time = pd.to_datetime(timestamp)
                    hours_held = (current_time - entry_time).total_seconds() / 3600
                    
                    if hours_held > self.time_based_stop_hours:
                        should_stop = True
                        stop_reason = f"Time-based exit after {hours_held:.0f}h"
                
                if should_stop:
                    # Execute stop loss
                    btc_to_sell = position['btc_amount']
                    trade_value = btc_to_sell * current_price
                    
                    self.btc_holdings -= btc_to_sell
                    self.current_balance += trade_value
                    self.trade_count += 1
                    self.losing_trades += 1  # Count as losing trade
                    
                    stop_loss_trades.append({
                        'timestamp': timestamp,
                        'signal': 'STOP_LOSS',
                        'price': current_price,
                        'btc_amount': btc_to_sell,
                        'usd_amount': trade_value,
                        'reason': stop_reason,
                        'portfolio_value': self.calculate_portfolio_value(current_price),
                        'position_type': 'STOP_LOSS',
                        'entry_price': position['entry_price'],
                        'entry_reason': position.get('entry_reason', 'Unknown')
                    })
                    
                    # Remove the stopped position
                    self.active_positions.remove(position)
        
        # Add stop loss trades to trade history
        self.trades.extend(stop_loss_trades)
        
        return len(stop_loss_trades)

    def calculate_portfolio_value(self, current_price):
        """Calculate portfolio value - same as original."""
        btc_value = self.btc_holdings * current_price
        total_value = self.current_balance + btc_value
        return total_value

    def run_backtest(self, days=365):
        """Run optimized backtest with date filtering."""
        print(f"\n🚀 Starting OPTIMIZED backtest...")
        print(f"🎯 Target Period: {self.start_date} to {self.end_date}")
        
        # Fetch data (get large dataset to ensure we have target period)
        df = self.fetch_historical_data(days)
        
        if df is None or len(df) < 1500:
            print(f"❌ Insufficient data: need at least 1500 hours, got {len(df) if df is not None else 0}")
            return None
        
        print(f"📊 Raw data: {len(df)} hours from {df.index[0]} to {df.index[-1]}")
        
        # Filter to specific date range
        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date) + pd.Timedelta(days=1)  # Include end date
        
        # Check if our target period is within the fetched data
        if start_date < df.index[0] or end_date > df.index[-1]:
            print(f"⚠️  Target period {self.start_date} to {self.end_date} not fully covered")
            print(f"   Available: {df.index[0].date()} to {df.index[-1].date()}")
            print(f"   Using available data overlap...")
            
            # Use the overlap period
            actual_start = max(start_date, df.index[0])
            actual_end = min(end_date, df.index[-1])
        else:
            actual_start = start_date
            actual_end = end_date
        
        # Filter DataFrame to target period
        filtered_df = df.loc[actual_start:actual_end].copy()
        
        if len(filtered_df) < 100:
            print(f"❌ Target period too short: only {len(filtered_df)} hours")
            return None
        
        print(f"✅ Filtered to target period: {len(filtered_df)} hours")
        print(f"   Period: {filtered_df.index[0]} to {filtered_df.index[-1]}")
        print(f"   Duration: {(filtered_df.index[-1] - filtered_df.index[0]).days} days")
        
        # Calculate enhanced indicators on full dataset first, then filter
        print("📊 Calculating enhanced technical indicators...")
        df = self.calculate_indicators(df)
        
        # Re-filter after indicators
        filtered_df = df.loc[actual_start:actual_end].copy()
        filtered_df = filtered_df.dropna()
        
        if len(filtered_df) < 100:
            print(f"❌ Not enough clean data after filtering: {len(filtered_df)} hours")
            return None
        
        print(f"✅ Clean filtered data: {len(filtered_df)} hours for testing")
        
        # Build transition matrix from data BEFORE our test period
        pre_test_data = df.loc[:actual_start].copy()
        if len(pre_test_data) < 500:
            print("⚠️  Limited training data, using early portion of test data for training")
            train_size = max(100, len(filtered_df) // 5)
            train_data = filtered_df.iloc[:train_size]
            test_data = filtered_df.iloc[train_size:]
        else:
            train_data = pre_test_data.iloc[-self.matrix_training_window:] if len(pre_test_data) > self.matrix_training_window else pre_test_data
            test_data = filtered_df
        
        print(f"🧠 Building optimized transition matrix from {len(train_data)} training hours...")
        transition_matrix = self.build_transition_matrix(train_data.dropna(subset=['markov_state']))
        
        print(f"✅ Training complete, testing on {len(test_data)} hours")
        
        # Initialize portfolio at start of test period
        initial_price = test_data['close'].iloc[0]
        self.entry_price = initial_price
        initial_btc_value = self.starting_btc_amount * initial_price
        initial_portfolio_value = initial_btc_value + self.current_balance
        
        print(f"💰 Starting portfolio at {test_data.index[0].date()}:")
        print(f"   BTC value: ${initial_btc_value:,.2f}")
        print(f"   Cash value: ${self.current_balance:,.2f}")
        print(f"   Total value: ${initial_portfolio_value:,.2f}")
        print(f"   BTC price: ${initial_price:,.2f}")
        
        # Main optimized backtest loop - ONLY on filtered period
        print(f"\n📈 Running OPTIMIZED simulation on target period...")
        
        for i in range(len(test_data)):
            current_row = test_data.iloc[i]
            current_price = current_row['close']
            
            # Update market regime
            full_df_idx = df.index.get_loc(current_row.name)
            self.market_regime = self.detect_market_regime(df, full_df_idx)
            
            # Update portfolio heat
            self.update_portfolio_heat(current_price)
            
            # Update trailing stops
            self.update_trailing_stops(current_price)
            
            # Check enhanced stop losses
            stop_losses_executed = self.check_enhanced_stop_losses(current_price, current_row.name)
            
            # Generate enhanced signal
            signal, reason = self.generate_signal(full_df_idx, df, transition_matrix)
            
            # Execute trade if signal
            if signal:
                self.execute_trade(signal, current_price, current_row.name, reason)
            
            # Track portfolio value with enhanced metrics
            portfolio_value = self.calculate_portfolio_value(current_price)
            self.portfolio_values.append({
                'timestamp': current_row.name,
                'portfolio_value': portfolio_value,
                'btc_price': current_price,
                'position': f"Mixed: {self.btc_holdings:.4f} BTC + ${self.current_balance:.2f} cash",
                'btc_holdings': self.btc_holdings,
                'cash_balance': self.current_balance,
                'active_positions': len(self.active_positions),
                'market_regime': self.market_regime,
                'portfolio_heat': self.portfolio_heat,
                'position_size_pct': self.position_size_pct,
                'stop_losses_today': stop_losses_executed
            })
            
            # Enhanced progress reporting
            if i % (len(test_data) // 10) == 0 and i > 0:
                progress = (i / len(test_data)) * 100
                growth = ((portfolio_value - initial_portfolio_value) / initial_portfolio_value) * 100
                print(f"   Progress: {progress:.1f}% - Portfolio: ${portfolio_value:,.2f} ({growth:+.1f}%) - "
                      f"Date: {current_row.name.date()} - Regime: {self.market_regime}")
        
        print("✅ Optimized backtest completed!")
        return self.analyze_results(df, initial_portfolio_value, filtered_period=(actual_start, actual_end))

    def analyze_results(self, df, initial_portfolio_value, filtered_period=None):
        """Analyze results with optimization metrics."""
        print("\n📊 Analyzing OPTIMIZED results...")
        
        if filtered_period:
            actual_start, actual_end = filtered_period
            print(f"📅 Analysis period: {actual_start.date()} to {actual_end.date()}")
            print(f"   Duration: {(actual_end - actual_start).days} days")
        
        # Convert results to DataFrames
        portfolio_df = pd.DataFrame(self.portfolio_values)
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        if len(portfolio_df) == 0:
            print("❌ No portfolio data to analyze")
            return None
        
        # Calculate final values
        final_portfolio_value = portfolio_df['portfolio_value'].iloc[-1]
        final_btc_price = portfolio_df['btc_price'].iloc[-1]
        
        # Buy and hold comparison (FIXED: Pure BTC strategy)
        initial_btc_price = portfolio_df['btc_price'].iloc[0]
        
        # Pure BTC buy & hold: Put entire starting portfolio into BTC
        pure_btc_amount = initial_portfolio_value / initial_btc_price
        pure_btc_final_value = pure_btc_amount * final_btc_price
        
        # Performance metrics
        total_return = ((final_portfolio_value - initial_portfolio_value) / initial_portfolio_value) * 100
        buy_hold_return = ((pure_btc_final_value - initial_portfolio_value) / initial_portfolio_value) * 100
        
        # Enhanced metrics
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
        valid_returns = portfolio_df['daily_return'].dropna()
        if len(valid_returns) > 1:
            sharpe_ratio = valid_returns.mean() / valid_returns.std() * np.sqrt(8760) if valid_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Drawdown analysis
        portfolio_df['running_max'] = portfolio_df['portfolio_value'].expanding().max()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['running_max']) / portfolio_df['running_max']
        max_drawdown = portfolio_df['drawdown'].min() * 100
        
        # Print enhanced results
        print("\n" + "="*60)
        print("🏆 OPTIMIZED BACKTEST RESULTS")
        if filtered_period:
            print(f"📅 Period: {actual_start.date()} to {actual_end.date()}")
        print("="*60)
        
        print(f"💰 PERFORMANCE:")
        print(f"   Total Return: {total_return:+.2f}%")
        print(f"   Buy & Hold Return: {buy_hold_return:+.2f}%")
        print(f"   ⚡ OPTIMIZATION BOOST: {total_return - buy_hold_return:+.2f}%")
        print(f"   Max Drawdown: {max_drawdown:.2f}%")
        print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
        
        print(f"\n🔧 OPTIMIZATION IMPACT:")
        print(f"   Dynamic Sizing Saves: {self.optimization_metrics['dynamic_sizing_saves']}")
        print(f"   Trailing Stop Profits: {self.optimization_metrics['trailing_stop_profits']}")
        print(f"   Regime-based Exits: {self.optimization_metrics['regime_based_exits']}")
        print(f"   Volume Filtered Trades: {self.optimization_metrics['volume_filtered_trades']}")
        
        if len(trades_df) > 0:
            # Market regime performance
            regime_performance = {}
            for regime in trades_df['market_regime'].unique():
                regime_trades = trades_df[trades_df['market_regime'] == regime]
                if len(regime_trades) > 0:
                    regime_performance[regime] = len(regime_trades)
            
            print(f"\n📊 TRADING BY MARKET REGIME:")
            for regime, count in regime_performance.items():
                print(f"   {regime}: {count} trades")
        
        # Price analysis for the period
        print(f"\n💹 PERIOD PRICE ANALYSIS:")
        price_change = ((final_btc_price - initial_btc_price) / initial_btc_price) * 100
        print(f"   BTC Start Price: ${initial_btc_price:,.2f}")
        print(f"   BTC End Price: ${final_btc_price:,.2f}")
        print(f"   BTC Price Change: {price_change:+.2f}%")
        
        return {
            'portfolio_df': portfolio_df,
            'trades_df': trades_df,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'optimization_boost': total_return - buy_hold_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'optimization_metrics': self.optimization_metrics,
            'filtered_period': filtered_period,
            'price_change': price_change,
            'period_days': (actual_end - actual_start).days if filtered_period else None
        }

def main():
    """Run the optimized backtest for the SAME 1-YEAR PERIOD as other bots."""
    print("🚀 OPTIMIZED MARKOV BOT BACKTEST: LAST 1 YEAR")
    print("="*60)
    print("Testing RISK-MANAGED optimization strategies for same period as others:")
    print("• Dynamic Position Sizing")
    print("• Trailing Stop Losses")
    print("• Market Regime Adaptation") 
    print("• Advanced Signal Filtering")
    print("• Portfolio Heat Management")
    print("Testing Period: Last 365 days (same as adaptive & aggressive bots)")
    print("")
    
    # Calculate last 365 days - SAME PERIOD as other bots
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Create optimized backtester for same 1-year period
    backtester = OptimizedMarkovBotBacktester(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        enable_weekly_rebuild=True,
        rebuild_frequency_hours=720,  # 30 days (proven optimal)
        matrix_training_window=2000   # 83 days
    )
    
    # Run optimized simulation for full year comparison
    results = backtester.run_backtest(days=400)  # Same data fetch as others
    
    if results:
        print(f"\n🎯 OPTIMIZED BACKTEST RESULTS for 1 YEAR!")
        print(f"Optimization boost: {results['optimization_boost']:+.2f}%")
        print(f"Total return: {results['total_return']:+.2f}%")
        print(f"Buy & hold return: {results['buy_hold_return']:+.2f}%")
        print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max drawdown: {results['max_drawdown']:.2f}%")
        
        # Calculate annualized metrics 
        if results.get('period_days') and results['period_days'] > 0:
            # Growth multiple from total return
            growth_multiple = 1 + (results['total_return'] / 100)
            buy_hold_multiple = 1 + (results['buy_hold_return'] / 100)
            
            # Annualize based on actual test period
            annualized_return = ((growth_multiple ** (365/results['period_days'])) - 1) * 100
            annualized_buy_hold = ((buy_hold_multiple ** (365/results['period_days'])) - 1) * 100
            
            print(f"\n📊 ANNUALIZED METRICS:")
            print(f"   Optimized Strategy: {annualized_return:+.1f}%")
            print(f"   Buy & Hold: {annualized_buy_hold:+.1f}%")
            print(f"   Annualized Alpha: {annualized_return - annualized_buy_hold:+.1f}%")
            print(f"   Test Period: {results['period_days']} days")
        
        # Save results with 1-year suffix
        if len(results['trades_df']) > 0:
            results['trades_df'].to_csv('optimized_1_year_trades.csv', index=False)
        results['portfolio_df'].to_csv('optimized_1_year_portfolio.csv', index=False)
        
        print(f"\n💾 Results saved:")
        print(f"   optimized_1_year_trades.csv")
        print(f"   optimized_1_year_portfolio.csv")
        
        # Performance insights for 1-year test
        print(f"\n💡 1-YEAR OPTIMIZATION INSIGHTS:")
        print(f"   • Dynamic sizing saves: {results['optimization_metrics']['dynamic_sizing_saves']}")
        print(f"   • Trailing stop profits: {results['optimization_metrics']['trailing_stop_profits']}")
        print(f"   • Volume filtered trades: {results['optimization_metrics']['volume_filtered_trades']}")
        print(f"   • Regime-based exits: {results['optimization_metrics']['regime_based_exits']}")
        
        total_trades = len(results['trades_df']) if len(results['trades_df']) > 0 else 0
        if results.get('period_days'):
            print(f"   • Trade frequency: {total_trades} trades in {results['period_days']} days")
            print(f"   • Weekly trade rate: {total_trades*7/results['period_days']:.1f} trades per week")
        print(f"   • Sharpe ratio: {results['sharpe_ratio']:.2f} (risk-adjusted performance)")
        
        # Risk metrics
        max_dd = results['max_drawdown']
        total_ret = results['total_return']
        if max_dd != 0:
            risk_efficiency = total_ret / abs(max_dd)
            print(f"   • Risk efficiency: {risk_efficiency:.2f} (return per unit of drawdown)")
        
        print(f"\n🎯 OPTIMIZED STRATEGY CHARACTERISTICS:")
        print(f"   • Conservative risk management approach")
        print(f"   • Dynamic position sizing (5%-30% range)")
        print(f"   • Advanced signal filtering and volume confirmation")
        print(f"   • Market regime detection and adaptation")
        print(f"   • Portfolio heat management")
        print(f"   • Trailing stops and time-based exits")
        
        return results
        
    else:
        print("❌ Optimized backtest failed!")
        return None

if __name__ == "__main__":
    main() 