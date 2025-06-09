#!/usr/bin/env python3
"""
AGGRESSIVE GROWTH Markov Trading Bot
====================================
Designed for SMALL ACCOUNTS that need FAST GROWTH
Focus: Maximum gains through intelligent risk-taking, not risk minimization

Key Strategies:
1. High Position Sizing (50-90% of capital)
2. Momentum Pyramiding 
3. Trend Following with Aggressive Entries
4. Compound Growth Optimization
5. Kelly Criterion Position Sizing
6. Reduced Conservative Filters
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

# === MARKOV CHAIN HELPERS ===
STATE_MATRIX_FILE = 'aggressive_transition_matrix.pkl'
MARKOV_STATES = [
    f'{p}-{r}'
    for p in ('UP', 'FLAT', 'DOWN')
    for r in ('OVERSOLD', 'NEUTRAL', 'OVERBOUGHT')
]

# AGGRESSIVE: Lower confidence threshold for more trades
MARKOV_CONFIDENCE_THRESHOLD = 0.65  # Lowered from 0.75 for more opportunities
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

# === AGGRESSIVE GROWTH BACKTESTING ENGINE ===
class AggressiveGrowthBot:
    def __init__(self, start_date='2024-01-01', end_date='2024-12-31'):
        self.start_date = start_date
        self.end_date = end_date
        
        # AGGRESSIVE: Start with larger balanced portfolio: $8K cash + $8K BTC
        self.starting_btc_amount = 0.08  # Approximately $8K worth of BTC
        self.current_balance = 8000.0    # $8K cash for aggressive trading
        self.btc_holdings = self.starting_btc_amount
        self.position = 'mixed'
        self.entry_price = None
        
        # AGGRESSIVE POSITION SIZING: 50-90% of capital per trade
        self.base_position_size_pct = 0.60   # Base 60% position size
        self.min_position_size_pct = 0.30    # Minimum 30%
        self.max_position_size_pct = 0.90    # Maximum 90% - GO BIG!
        self.position_size_pct = self.base_position_size_pct
        
        # AGGRESSIVE RISK MANAGEMENT: Looser stops, longer holds
        self.base_stop_loss_pct = 0.12       # 12% stop loss (wider)
        self.trailing_stop_enabled = True    # But still use trailing stops
        self.time_based_stop_hours = 168     # Hold positions longer (1 week)
        
        # AGGRESSIVE MOMENTUM: Focus on trending markets
        self.momentum_threshold = 2.0        # Higher momentum requirement
        self.trend_strength_required = True  # Require strong trends
        self.pyramid_enabled = True          # Add to winning positions
        self.max_pyramid_levels = 3          # Up to 3x pyramid
        
        # AGGRESSIVE COMPOUNDING: Reinvest all gains
        self.compound_gains = True           # Reinvest everything
        self.kelly_criterion = True          # Use Kelly for optimal sizing
        self.kelly_lookback = 100            # Recent performance for Kelly
        
        # REDUCED FILTERING: Less conservative, more trades
        self.volume_filter_enabled = False   # Remove volume filter
        self.momentum_filter_threshold = 0.5 # Lower threshold
        self.confidence_filter = False       # Take more marginal trades
        
        # Tracking
        self.active_positions = []
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.pyramid_count = 0
        self.kelly_multiplier = 1.0
        
        # Performance tracking
        self.trades = []
        self.portfolio_values = []
        self.growth_metrics = {
            'max_position_size_used': 0,
            'pyramid_trades': 0,
            'kelly_adjustments': 0,
            'momentum_captures': 0
        }
        
        print(f"🚀 AGGRESSIVE GROWTH BOT INITIALIZED")
        print(f"💰 Starting: {self.starting_btc_amount:.6f} BTC + ${self.current_balance:,.2f}")
        print(f"⚡ AGGRESSIVE SETTINGS:")
        print(f"   🎯 Position Sizing: {self.min_position_size_pct*100:.0f}%-{self.max_position_size_pct*100:.0f}% (Base: {self.base_position_size_pct*100:.0f}%)")
        print(f"   🛡️  Stop Loss: {self.base_stop_loss_pct*100:.0f}% (Wider for big moves)")
        print(f"   📈 Pyramiding: Enabled (Up to {self.max_pyramid_levels}x)")
        print(f"   💎 Kelly Criterion: Enabled")
        print(f"   🔥 Compound Growth: All gains reinvested")
        print(f"   ⚡ Reduced Filters: Maximum opportunity capture")
        print("")
        print("🎯 STRATEGY: Grow fast first, manage risk later!")

    def fetch_historical_data(self, days=365):
        """Fetch historical data - same as before."""
        print("📈 Fetching historical data for aggressive growth testing...")
        
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
                all_candles = []
                batch_size = max_limit
                max_batches = min(20, (hours_needed // batch_size) + 2)
                
                since_timestamp = None
                
                for batch_num in range(max_batches):
                    try:
                        if since_timestamp is None:
                            candles = exchange.fetch_ohlcv(symbol, '1h', limit=batch_size)
                        else:
                            candles = exchange.fetch_ohlcv(symbol, '1h', since=since_timestamp, limit=batch_size)
                        
                        if not candles or len(candles) == 0:
                            break
                        
                        if all_candles and candles[-1][0] >= all_candles[0][0]:
                            break
                        
                        all_candles = candles + all_candles
                        
                        oldest_timestamp = candles[0][0]
                        since_timestamp = oldest_timestamp - (batch_size * 3600000)
                        
                        if len(all_candles) >= hours_needed * 0.7:
                            break
                        
                        import time
                        time.sleep(0.5)
                        
                    except Exception as e:
                        if "rate limit" in str(e).lower():
                            import time
                            time.sleep(30)
                            continue
                        else:
                            break
                
                if len(all_candles) >= 2000:
                    print(f"✅ Got {len(all_candles)} candles from {exchange_name.upper()}")
                    break
                    
            except Exception as e:
                continue
        
        if len(all_candles) == 0:
            print("❌ Could not fetch data!")
            return None
        
        # Process data
        seen_timestamps = set()
        unique_candles = []
        
        for candle in sorted(all_candles, key=lambda x: x[0]):
            if candle[0] not in seen_timestamps:
                unique_candles.append(candle)
                seen_timestamps.add(candle[0])
        
        df = pd.DataFrame(unique_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()
        
        actual_days = len(df) / 24
        print(f"✅ Loaded {len(df)} hours ({actual_days:.1f} days)")
        print(f"   Period: {df.index[0]} to {df.index[-1]}")
        
        return df

    def calculate_indicators(self, df):
        """Calculate indicators with aggressive momentum focus."""
        # Basic indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        
        # EMAs for trend detection
        df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
        
        # AGGRESSIVE: Enhanced momentum indicators
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        df['momentum_5'] = df['close'].pct_change(periods=5) * 100
        df['momentum_10'] = df['close'].pct_change(periods=10) * 100  
        df['momentum_20'] = df['close'].pct_change(periods=20) * 100
        
        # Trend strength indicators
        df['trend_strength'] = np.where(
            (df['ema_12'] > df['ema_26']) & (df['ema_26'] > df['ema_50']) & (df['ema_50'] > df['ema_200']),
            2,  # Strong uptrend
            np.where(
                (df['ema_12'] < df['ema_26']) & (df['ema_26'] < df['ema_50']) & (df['ema_50'] < df['ema_200']),
                -2,  # Strong downtrend
                np.where(df['ema_12'] > df['ema_26'], 1, -1)  # Weak trend
            )
        )
        
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

    def calculate_kelly_position_size(self, df, current_idx):
        """Calculate optimal position size using Kelly Criterion."""
        if current_idx < self.kelly_lookback:
            return self.base_position_size_pct
        
        # Look at recent trades performance
        recent_trades = []
        if len(self.trades) >= 10:  # Need minimum trade history
            recent_trades = self.trades[-self.kelly_lookback:]
            
            wins = [t for t in recent_trades if 'BUY' in t.get('signal', '') and t.get('portfolio_value', 0) > 0]
            losses = [t for t in recent_trades if 'STOP_LOSS' in t.get('signal', '')]
            
            if len(wins) > 0 and len(losses) > 0:
                win_rate = len(wins) / (len(wins) + len(losses))
                avg_win = np.mean([w.get('usd_amount', 0) for w in wins])
                avg_loss = np.mean([l.get('usd_amount', 0) for l in losses])
                
                if avg_loss > 0:
                    # Kelly formula: f = (bp - q) / b
                    # where b = odds received, p = win probability, q = loss probability
                    b = avg_win / avg_loss  # Odds
                    p = win_rate
                    q = 1 - win_rate
                    
                    kelly_fraction = (b * p - q) / b
                    kelly_fraction = max(0.1, min(0.9, kelly_fraction))  # Bound between 10-90%
                    
                    self.kelly_multiplier = kelly_fraction / self.base_position_size_pct
                    self.growth_metrics['kelly_adjustments'] += 1
                    
                    return kelly_fraction
        
        return self.base_position_size_pct

    def calculate_aggressive_position_size(self, df, current_idx, signal_strength=1.0):
        """Calculate aggressive position size for maximum growth."""
        if current_idx < 20:
            return self.base_position_size_pct
        
        row = df.iloc[current_idx]
        
        # Start with Kelly Criterion
        kelly_size = self.calculate_kelly_position_size(df, current_idx)
        
        # Momentum multipliers - bigger positions in strong trends
        momentum_multipliers = []
        
        # 1. Trend strength multiplier
        trend_strength = row.get('trend_strength', 0)
        if abs(trend_strength) == 2:  # Strong trend
            momentum_multipliers.append(1.3)
        elif abs(trend_strength) == 1:  # Weak trend
            momentum_multipliers.append(1.1)
        else:
            momentum_multipliers.append(0.9)
        
        # 2. Momentum multiplier
        momentum_20 = row.get('momentum_20', 0)
        if abs(momentum_20) > 10:  # Strong momentum
            momentum_multipliers.append(1.2)
        elif abs(momentum_20) > 5:  # Medium momentum
            momentum_multipliers.append(1.1)
        else:
            momentum_multipliers.append(1.0)
        
        # 3. Signal strength multiplier
        momentum_multipliers.append(signal_strength)
        
        # 4. Portfolio growth multiplier - bigger positions as we grow
        current_portfolio_value = self.calculate_portfolio_value(row['close'])
        initial_value = self.starting_btc_amount * row['close'] + 6000
        growth_ratio = current_portfolio_value / initial_value
        
        if growth_ratio > 1.5:  # 50%+ gains
            momentum_multipliers.append(1.2)
        elif growth_ratio > 1.2:  # 20%+ gains
            momentum_multipliers.append(1.1)
        else:
            momentum_multipliers.append(1.0)
        
        # Calculate final aggressive size
        final_multiplier = np.prod(momentum_multipliers)
        aggressive_size = kelly_size * final_multiplier
        
        # Apply bounds but allow very large positions
        aggressive_size = max(self.min_position_size_pct, 
                            min(self.max_position_size_pct, aggressive_size))
        
        # Track maximum position size used
        if aggressive_size > self.growth_metrics['max_position_size_used']:
            self.growth_metrics['max_position_size_used'] = aggressive_size
        
        return aggressive_size

    def should_pyramid(self, current_price):
        """Check if we should add to winning positions (pyramiding)."""
        if not self.pyramid_enabled:
            return False
        
        profitable_positions = []
        for position in self.active_positions:
            if position['type'] == 'long':
                unrealized_pnl = (current_price - position['entry_price']) / position['entry_price']
                if unrealized_pnl > 0.05:  # 5%+ profit
                    profitable_positions.append(position)
        
        # Only pyramid if we have profitable positions and haven't maxed out
        if profitable_positions and len(self.active_positions) < self.max_pyramid_levels:
            return True
        
        return False

    def generate_aggressive_signal(self, current_idx, df, transition_matrix):
        """Generate aggressive trading signals focused on momentum and trends."""
        row = df.iloc[current_idx]
        
        if pd.isna(row['rsi']) or pd.isna(row['bb_lower']) or pd.isna(row['ema_200']):
            return None, "Insufficient data"
        
        current_price = row['close']
        signal = None
        reason = ""
        signal_strength = 1.0
        
        # AGGRESSIVE BUY CONDITIONS
        buy_conditions = []
        
        # 1. Strong momentum breakout
        if (row['momentum_10'] > self.momentum_threshold and 
            row['trend_strength'] >= 1 and
            current_price > row['ema_12'] and
            self.current_balance > 100):
            buy_conditions.append(("Momentum Breakout", 1.3))
        
        # 2. Oversold in uptrend (aggressive dip buying)
        if (row['rsi'] < 40 and  # Less oversold requirement
            row['trend_strength'] >= 1 and
            current_price < row['bb_lower'] and
            current_price > row['ema_50'] and
            self.current_balance > 100):
            buy_conditions.append(("Aggressive Dip Buy", 1.2))
        
        # 3. Pyramid into winning positions
        if (self.should_pyramid(current_price) and
            row['momentum_5'] > 1 and
            row['trend_strength'] >= 1 and
            self.current_balance > 100):
            buy_conditions.append(("Pyramid Add", 1.4))
        
        # AGGRESSIVE SELL CONDITIONS  
        sell_conditions = []
        
        # 1. Take profits on momentum exhaustion
        if (row['rsi'] > 75 and  # More overbought
            row['momentum_5'] < -1 and
            self.btc_holdings > 0.001):
            sell_conditions.append(("Momentum Exhaustion", 1.2))
        
        # 2. Trend reversal
        if (row['trend_strength'] <= -1 and
            current_price < row['ema_12'] and
            self.btc_holdings > 0.001):
            sell_conditions.append(("Trend Reversal", 1.3))
        
        # Execute strongest condition
        if buy_conditions:
            strongest_buy = max(buy_conditions, key=lambda x: x[1])
            signal = 'BUY'
            reason = strongest_buy[0]
            signal_strength = strongest_buy[1]
            
            if "Pyramid" in reason:
                self.pyramid_count += 1
                self.growth_metrics['pyramid_trades'] += 1
        
        elif sell_conditions:
            strongest_sell = max(sell_conditions, key=lambda x: x[1])
            signal = 'SELL'
            reason = strongest_sell[0]
            signal_strength = strongest_sell[1]
        
        # MARKOV GATING (Reduced threshold for more trades)
        if current_idx > 0 and not pd.isna(row['markov_state']):
            current_state = row['markov_state']
            next_state, probability = predict_next_state(current_state, transition_matrix)
            
            # Aggressive Markov BUY
            if (signal is None and 
                next_state in BULLISH_STATES and 
                probability >= MARKOV_CONFIDENCE_THRESHOLD and 
                self.current_balance > 100):
                
                signal = 'BUY'
                reason = f"Markov Aggressive: {current_state}→{next_state} ({probability:.0%})"
                signal_strength = probability
            
            # Aggressive Markov SELL
            elif (self.btc_holdings > 0.001 and 
                  next_state in BEARISH_STATES and 
                  probability >= MARKOV_CONFIDENCE_THRESHOLD):
                
                signal = 'SELL'
                reason = f"Markov Exit: {current_state}→{next_state} ({probability:.0%})"
                signal_strength = probability
        
        # Update position size for this signal
        if signal == 'BUY':
            self.position_size_pct = self.calculate_aggressive_position_size(df, current_idx, signal_strength)
        
        return signal, reason

    def execute_aggressive_trade(self, signal, price, timestamp, reason):
        """Execute trades with aggressive position sizing."""
        
        if signal == 'BUY' and self.current_balance > 100:
            # AGGRESSIVE: Use large position sizes
            trade_amount = self.current_balance * self.position_size_pct
            btc_to_buy = trade_amount / price
            
            self.current_balance -= trade_amount
            self.btc_holdings += btc_to_buy
            self.trade_count += 1
            
            # Wider stop loss for bigger moves
            stop_loss_price = price * (1 - self.base_stop_loss_pct)
            
            position = {
                'type': 'long',
                'entry_price': price,
                'btc_amount': btc_to_buy,
                'entry_time': timestamp,
                'stop_loss': stop_loss_price,
                'trailing_stop': None,
                'position_size_pct': self.position_size_pct,
                'entry_reason': reason,
                'pyramid_level': self.pyramid_count if "Pyramid" in reason else 0
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
                'kelly_multiplier': self.kelly_multiplier
            })
            
        elif signal == 'SELL' and self.btc_holdings > 0.001:
            # AGGRESSIVE: Sell larger portions
            btc_to_sell = self.btc_holdings * self.position_size_pct
            trade_value = btc_to_sell * price
            
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
                'kelly_multiplier': self.kelly_multiplier
            })

    def _close_long_positions(self, btc_sold, current_price):
        """Close positions with P&L tracking."""
        btc_remaining = btc_sold
        positions_to_remove = []
        
        for i, position in enumerate(self.active_positions):
            if position['type'] == 'long' and btc_remaining > 0:
                btc_from_position = min(position['btc_amount'], btc_remaining)
                
                # Calculate P&L
                entry_price = position['entry_price']
                pnl = (current_price - entry_price) * btc_from_position
                
                if pnl > 0:
                    self.winning_trades += 1
                    if position.get('pyramid_level', 0) > 0:
                        self.growth_metrics['pyramid_trades'] += 1
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

    def update_trailing_stops_aggressive(self, current_price):
        """Update trailing stops but allow for bigger moves."""
        for position in self.active_positions:
            if position['type'] == 'long':
                entry_price = position['entry_price']
                
                # Start trailing after bigger profit (10% vs 5%)
                if current_price > entry_price * 1.10:
                    # Trail at 8% below high (vs 3%)
                    if position['trailing_stop'] is None:
                        position['trailing_stop'] = current_price * 0.92
                    else:
                        new_trailing_stop = current_price * 0.92
                        if new_trailing_stop > position['trailing_stop']:
                            position['trailing_stop'] = new_trailing_stop

    def check_aggressive_stop_losses(self, current_price, timestamp):
        """Check stop losses with wider tolerance."""
        stop_loss_trades = []
        
        for position in self.active_positions[:]:
            if position['type'] == 'long':
                should_stop = False
                stop_reason = ""
                
                # 1. Regular stop loss (wider at 12%)
                if current_price <= position['stop_loss']:
                    should_stop = True
                    stop_reason = f"Stop loss hit at ${current_price:.2f}"
                
                # 2. Trailing stop (wider trail)
                elif position.get('trailing_stop') and current_price <= position['trailing_stop']:
                    should_stop = True
                    stop_reason = f"Trailing stop hit at ${current_price:.2f}"
                
                # 3. Time-based stop (hold longer - 1 week)
                elif self.time_based_stop_hours > 0:
                    entry_time = pd.to_datetime(position['entry_time'])
                    current_time = pd.to_datetime(timestamp)
                    hours_held = (current_time - entry_time).total_seconds() / 3600
                    
                    if hours_held > self.time_based_stop_hours:
                        should_stop = True
                        stop_reason = f"Time-based exit after {hours_held:.0f}h"
                
                if should_stop:
                    btc_to_sell = position['btc_amount']
                    trade_value = btc_to_sell * current_price
                    
                    self.btc_holdings -= btc_to_sell
                    self.current_balance += trade_value
                    self.trade_count += 1
                    self.losing_trades += 1
                    
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
                        'pyramid_level': position.get('pyramid_level', 0)
                    })
                    
                    self.active_positions.remove(position)
        
        self.trades.extend(stop_loss_trades)
        return len(stop_loss_trades)

    def calculate_portfolio_value(self, current_price):
        """Calculate portfolio value."""
        btc_value = self.btc_holdings * current_price
        total_value = self.current_balance + btc_value
        return total_value

    def build_transition_matrix(self, df, train_period=1000):
        """Build transition matrix."""
        available_data = len(df)
        actual_train_period = max(1000, min(train_period, available_data // 3))
        
        print(f"🧠 Building aggressive transition matrix from first {actual_train_period} hours...")
        
        train_data = df.iloc[:actual_train_period].dropna(subset=['markov_state'])
        matrix = build_transition_matrix(train_data)
        
        with open('aggressive_transition_matrix.pkl', 'wb') as f:
            pickle.dump(matrix, f)
        
        print(f"✅ Aggressive matrix built from {len(train_data)} training samples")
        return matrix

    def run_aggressive_backtest(self, days=365):
        """Run aggressive growth backtest."""
        print(f"\n🚀 Starting AGGRESSIVE GROWTH backtest...")
        print(f"🎯 Target Period: {self.start_date} to {self.end_date}")
        
        # Fetch data (get large dataset to ensure we have target period)
        df = self.fetch_historical_data(days)
        
        if df is None or len(df) < 1500:
            print(f"❌ Insufficient data")
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
        
        # Calculate indicators on full dataset first, then filter
        print("📊 Calculating aggressive momentum indicators...")
        df = self.calculate_indicators(df)
        
        # Re-filter after indicators (some may be NaN at the start)
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
            # Use first 20% of filtered data for training if no pre-data
            train_size = max(100, len(filtered_df) // 5)
            train_data = filtered_df.iloc[:train_size]
            test_data = filtered_df.iloc[train_size:]
        else:
            # Use pre-test data for training
            train_data = pre_test_data.iloc[-1000:] if len(pre_test_data) > 1000 else pre_test_data
            test_data = filtered_df
        
        print(f"🧠 Building transition matrix from {len(train_data)} training hours...")
        transition_matrix = build_transition_matrix(train_data.dropna(subset=['markov_state']))
        
        with open('aggressive_transition_matrix.pkl', 'wb') as f:
            pickle.dump(transition_matrix, f)
        
        print(f"✅ Training complete, testing on {len(test_data)} hours")
        
        # Initialize portfolio at start of test period
        initial_price = test_data['close'].iloc[0]
        initial_btc_value = self.starting_btc_amount * initial_price
        initial_portfolio_value = initial_btc_value + self.current_balance
        
        print(f"💰 Starting portfolio at {test_data.index[0].date()}:")
        print(f"   BTC value: ${initial_btc_value:,.2f}")
        print(f"   Cash value: ${self.current_balance:,.2f}")
        print(f"   Total value: ${initial_portfolio_value:,.2f}")
        print(f"   BTC price: ${initial_price:,.2f}")
        
        # Main aggressive backtest loop - ONLY on filtered period
        print(f"\n🔥 Running AGGRESSIVE GROWTH simulation on target period...")
        
        for i in range(len(test_data)):
            current_row = test_data.iloc[i]
            current_price = current_row['close']
            
            # Update trailing stops
            self.update_trailing_stops_aggressive(current_price)
            
            # Check stop losses
            stop_losses_executed = self.check_aggressive_stop_losses(current_price, current_row.name)
            
            # Generate aggressive signal (need full df index for context)
            full_df_idx = df.index.get_loc(current_row.name)
            signal, reason = self.generate_aggressive_signal(full_df_idx, df, transition_matrix)
            
            # Execute trade
            if signal:
                self.execute_aggressive_trade(signal, current_price, current_row.name, reason)
            
            # Track portfolio value
            portfolio_value = self.calculate_portfolio_value(current_price)
            self.portfolio_values.append({
                'timestamp': current_row.name,
                'portfolio_value': portfolio_value,
                'btc_price': current_price,
                'btc_holdings': self.btc_holdings,
                'cash_balance': self.current_balance,
                'active_positions': len(self.active_positions),
                'position_size_pct': self.position_size_pct,
                'kelly_multiplier': self.kelly_multiplier,
                'stop_losses_today': stop_losses_executed
            })
            
            # Progress update
            if i % (len(test_data) // 10) == 0 and i > 0:
                progress = (i / len(test_data)) * 100
                growth = ((portfolio_value - initial_portfolio_value) / initial_portfolio_value) * 100
                print(f"   Progress: {progress:.1f}% - Portfolio: ${portfolio_value:,.2f} ({growth:+.1f}%) - "
                      f"Date: {current_row.name.date()} - "
                      f"Max Pos: {self.growth_metrics['max_position_size_used']*100:.0f}%")
        
        print("✅ Aggressive growth backtest completed!")
        return self.analyze_aggressive_results(df, initial_portfolio_value, filtered_period=(actual_start, actual_end))

    def analyze_aggressive_results(self, df, initial_portfolio_value, filtered_period=None):
        """Analyze aggressive growth results."""
        print("\n📊 Analyzing AGGRESSIVE GROWTH results...")
        
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
        
        # Print aggressive results
        print("\n" + "="*60)
        print("🔥 AGGRESSIVE GROWTH BACKTEST RESULTS")
        if filtered_period:
            print(f"📅 Period: {actual_start.date()} to {actual_end.date()}")
        print("="*60)
        
        print(f"💰 PERFORMANCE:")
        print(f"   Total Return: {total_return:+.2f}%")
        print(f"   Buy & Hold Return: {buy_hold_return:+.2f}%")
        print(f"   🚀 AGGRESSIVE BOOST: {total_return - buy_hold_return:+.2f}%")
        print(f"   Max Drawdown: {max_drawdown:.2f}%")
        print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
        
        print(f"\n🔥 AGGRESSIVE GROWTH METRICS:")
        print(f"   Max Position Size Used: {self.growth_metrics['max_position_size_used']*100:.1f}%")
        print(f"   Pyramid Trades: {self.growth_metrics['pyramid_trades']}")
        print(f"   Kelly Adjustments: {self.growth_metrics['kelly_adjustments']}")
        print(f"   Total Trades: {len(trades_df)}")
        if self.winning_trades + self.losing_trades > 0:
            print(f"   Win Rate: {(self.winning_trades/(self.winning_trades+self.losing_trades)*100):.1f}%")
        
        # Capital growth analysis
        print(f"\n📈 CAPITAL GROWTH ANALYSIS:")
        growth_multiple = final_portfolio_value / initial_portfolio_value
        print(f"   Growth Multiple: {growth_multiple:.2f}x")
        print(f"   Initial Capital: ${initial_portfolio_value:,.2f}")
        print(f"   Final Capital: ${final_portfolio_value:,.2f}")
        print(f"   Capital Added: ${final_portfolio_value - initial_portfolio_value:,.2f}")
        
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
            'aggressive_boost': total_return - buy_hold_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'growth_multiple': growth_multiple,
            'growth_metrics': self.growth_metrics,
            'filtered_period': filtered_period,
            'price_change': price_change,
            'period_days': (actual_end - actual_start).days if filtered_period else None
        }

def main():
    """Run aggressive growth backtest for the last 1 YEAR for comparison with adaptive bot."""
    print("🔥 AGGRESSIVE GROWTH BOT BACKTEST: LAST 1 YEAR")
    print("="*60)
    print("Strategy: MAXIMUM GAINS for small accounts")
    print("Focus: Fast growth first, risk management later")
    print("Testing Period: Last 365 days (same as adaptive bot)")
    print("")
    
    # Calculate last 365 days - SAME PERIOD as adaptive bot
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Create aggressive growth bot
    bot = AggressiveGrowthBot(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    # Run aggressive simulation for full year
    results = bot.run_aggressive_backtest(days=400)  # Same data fetch as adaptive
    
    if results:
        print(f"\n🔥 AGGRESSIVE GROWTH RESULTS for 1 YEAR!")
        print(f"Aggressive boost: {results['aggressive_boost']:+.2f}%")
        print(f"Total return: {results['total_return']:+.2f}%")
        print(f"Buy & hold return: {results['buy_hold_return']:+.2f}%")
        print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max drawdown: {results['max_drawdown']:.2f}%")
        print(f"Growth multiple: {results['growth_multiple']:.2f}x")
        
        # Calculate annualized metrics  
        if results.get('period_days') and results['period_days'] > 0:
            # Growth multiple from total return
            growth_multiple = 1 + (results['total_return'] / 100)
            buy_hold_multiple = 1 + (results['buy_hold_return'] / 100)
            
            # Annualize based on actual test period
            annualized_return = ((growth_multiple ** (365/results['period_days'])) - 1) * 100
            annualized_buy_hold = ((buy_hold_multiple ** (365/results['period_days'])) - 1) * 100
            
            print(f"\n📊 ANNUALIZED METRICS:")
            print(f"   Aggressive Strategy: {annualized_return:+.1f}%")
            print(f"   Buy & Hold: {annualized_buy_hold:+.1f}%")
            print(f"   Annualized Alpha: {annualized_return - annualized_buy_hold:+.1f}%")
            print(f"   Test Period: {results['period_days']} days")
        
        # Save results with 1-year suffix
        if len(results['trades_df']) > 0:
            results['trades_df'].to_csv('aggressive_1_year_trades.csv', index=False)
        results['portfolio_df'].to_csv('aggressive_1_year_portfolio.csv', index=False)
        
        print(f"\n💾 Results saved:")
        print(f"   aggressive_1_year_trades.csv")
        print(f"   aggressive_1_year_portfolio.csv")
        
        # Performance insights for 1-year test
        print(f"\n💡 1-YEAR AGGRESSIVE INSIGHTS:")
        print(f"   • Maximum position sizing: up to {results['growth_metrics']['max_position_size_used']*100:.0f}% of capital")
        print(f"   • Trade frequency: {len(results['trades_df'])} trades in {results.get('period_days', 365)} days")
        total_trades = len(results['trades_df']) if len(results['trades_df']) > 0 else 0
        if results.get('period_days'):
            print(f"   • Weekly trade rate: {total_trades*7/results['period_days']:.1f} trades per week")
        print(f"   • Pyramid trades: {results['growth_metrics']['pyramid_trades']}")
        print(f"   • Kelly adjustments: {results['growth_metrics']['kelly_adjustments']}")
        print(f"   • Momentum captures: {results['growth_metrics']['momentum_captures']}")
        
        # Risk metrics
        max_dd = results['max_drawdown']
        total_ret = results['total_return']
        if max_dd != 0:
            risk_efficiency = total_ret / abs(max_dd)
            print(f"   • Risk efficiency: {risk_efficiency:.2f} (return per unit of drawdown)")
        
        print(f"\n🎯 AGGRESSIVE STRATEGY CHARACTERISTICS:")
        print(f"   • High-risk, high-reward approach")
        print(f"   • Larger position sizes for maximum growth")
        print(f"   • Wider stop losses for trend riding")
        print(f"   • Momentum-focused entry/exit signals")
        print(f"   • Compound growth optimization")
        
        return results
        
    else:
        print("❌ Aggressive backtest failed!")
        return None

if __name__ == "__main__":
    main() 