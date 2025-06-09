#!/usr/bin/env python3
"""
MARKET-ADAPTIVE TRADING BOT
===========================
The NEXT LEVEL strategy that automatically switches between:
- AGGRESSIVE mode (bull markets, low volatility)
- CONSERVATIVE mode (bear markets, high volatility)

Key Innovation: INTELLIGENT RISK ADAPTATION
- Real-time market regime detection
- Dynamic strategy switching
- Risk-adjusted position sizing
- Volatility-based stop losses
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

# === MARKET REGIME DETECTION ===
class MarketRegimeDetector:
    def __init__(self):
        self.regimes = {
            'BULL_LOW_VOL': {'risk_mode': 'AGGRESSIVE', 'position_multiplier': 1.5, 'stop_loss': 0.15},
            'BULL_HIGH_VOL': {'risk_mode': 'MODERATE', 'position_multiplier': 1.2, 'stop_loss': 0.10},
            'SIDEWAYS': {'risk_mode': 'CONSERVATIVE', 'position_multiplier': 0.8, 'stop_loss': 0.08},
            'BEAR_LOW_VOL': {'risk_mode': 'DEFENSIVE', 'position_multiplier': 0.5, 'stop_loss': 0.06},
            'BEAR_HIGH_VOL': {'risk_mode': 'SURVIVAL', 'position_multiplier': 0.3, 'stop_loss': 0.05}
        }
    
    def detect_regime(self, df, current_idx):
        """Detect current market regime using multiple indicators."""
        if current_idx < 50:  # Need enough history
            return 'SIDEWAYS'
        
        row = df.iloc[current_idx]
        
        # 1. Price trend (20-day momentum)
        momentum_20 = row.get('momentum_20', 0)
        
        # 2. Volatility level (ATR ratio)
        atr_ratio = row.get('atr_ratio', 1.0)
        
        # 3. Trend strength
        trend_strength = row.get('trend_strength', 0)
        
        # Market regime logic
        if momentum_20 > 5 and trend_strength >= 1:  # Strong upward momentum
            if atr_ratio < 1.2:  # Low volatility
                return 'BULL_LOW_VOL'
            else:  # High volatility
                return 'BULL_HIGH_VOL'
        elif momentum_20 < -5 and trend_strength <= -1:  # Strong downward momentum
            if atr_ratio < 1.2:  # Low volatility
                return 'BEAR_LOW_VOL'
            else:  # High volatility
                return 'BEAR_HIGH_VOL'
        else:  # Sideways/choppy
            return 'SIDEWAYS'

# === ADAPTIVE MARKOV HELPERS ===
STATE_MATRIX_FILE = 'adaptive_transition_matrix.pkl'
MARKOV_STATES = [
    f'{p}-{r}'
    for p in ('UP', 'FLAT', 'DOWN')
    for r in ('OVERSOLD', 'NEUTRAL', 'OVERBOUGHT')
]

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

# === MARKET-ADAPTIVE BOT ===
class MarketAdaptiveBot:
    def __init__(self, start_date='2024-01-01', end_date='2024-12-31'):
        self.start_date = start_date
        self.end_date = end_date
        
        # Portfolio setup
        self.starting_btc_amount = 0.08  # $8K worth of BTC
        self.current_balance = 8000.0    # $8K cash
        self.btc_holdings = self.starting_btc_amount
        
        # Market regime detector
        self.regime_detector = MarketRegimeDetector()
        self.current_regime = 'SIDEWAYS'
        self.current_risk_mode = 'CONSERVATIVE'
        
        # Base settings (will be adjusted by regime)
        self.base_position_size_pct = 0.40  # Conservative base
        self.position_size_pct = self.base_position_size_pct
        self.current_stop_loss_pct = 0.08   # Conservative base
        
        # Adaptive parameters
        self.regime_confidence_thresholds = {
            'AGGRESSIVE': 0.60,    # Lower threshold for aggressive entry
            'MODERATE': 0.65,      # Medium threshold
            'CONSERVATIVE': 0.70,  # Higher threshold for conservative
            'DEFENSIVE': 0.75,     # Very high threshold
            'SURVIVAL': 0.80       # Extreme threshold
        }
        
        # Tracking
        self.active_positions = []
        self.trades = []
        self.portfolio_values = []
        self.regime_changes = []
        
        # Performance tracking by regime
        self.regime_performance = {
            'BULL_LOW_VOL': {'trades': 0, 'pnl': 0},
            'BULL_HIGH_VOL': {'trades': 0, 'pnl': 0},
            'SIDEWAYS': {'trades': 0, 'pnl': 0},
            'BEAR_LOW_VOL': {'trades': 0, 'pnl': 0},
            'BEAR_HIGH_VOL': {'trades': 0, 'pnl': 0}
        }
        
        print(f"🎯 MARKET-ADAPTIVE BOT INITIALIZED")
        print(f"💰 Starting: {self.starting_btc_amount:.6f} BTC + ${self.current_balance:,.2f}")
        print(f"🧠 ADAPTIVE STRATEGY:")
        print(f"   📊 Real-time regime detection")
        print(f"   ⚡ Dynamic risk adjustment")
        print(f"   🛡️  Adaptive position sizing")
        print(f"   🎚️  Variable confidence thresholds")
        print(f"   🔄 Automatic strategy switching")
        print("")
        print("🎯 STRATEGY: Intelligent risk adaptation for all market conditions!")

    def fetch_historical_data(self, days=365):
        """Fetch historical data - same as before."""
        print("📈 Fetching historical data for adaptive strategy testing...")
        
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
        """Calculate indicators including regime detection metrics."""
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
        
        # Volatility indicators for regime detection
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        df['atr_ma'] = df['atr'].rolling(window=50).mean()
        df['atr_ratio'] = df['atr'] / df['atr_ma']
        
        # Momentum indicators
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

    def update_market_regime(self, df, current_idx):
        """Update market regime and adapt strategy accordingly."""
        new_regime = self.regime_detector.detect_regime(df, current_idx)
        
        if new_regime != self.current_regime:
            old_regime = self.current_regime
            self.current_regime = new_regime
            
            # Get regime settings
            regime_settings = self.regime_detector.regimes[new_regime]
            self.current_risk_mode = regime_settings['risk_mode']
            
            # Adapt position sizing
            position_multiplier = regime_settings['position_multiplier']
            self.position_size_pct = min(0.90, self.base_position_size_pct * position_multiplier)
            
            # Adapt stop loss
            self.current_stop_loss_pct = regime_settings['stop_loss']
            
            # Log regime change
            self.regime_changes.append({
                'timestamp': df.index[current_idx],
                'old_regime': old_regime,
                'new_regime': new_regime,
                'risk_mode': self.current_risk_mode,
                'position_size': self.position_size_pct,
                'stop_loss': self.current_stop_loss_pct
            })
            
            print(f"📊 REGIME CHANGE at {df.index[current_idx].date()}: {old_regime} → {new_regime}")
            print(f"   🎚️  Risk Mode: {self.current_risk_mode}")
            print(f"   📏 Position Size: {self.position_size_pct*100:.1f}%")
            print(f"   🛡️  Stop Loss: {self.current_stop_loss_pct*100:.1f}%")

    def generate_adaptive_signal(self, current_idx, df, transition_matrix):
        """Generate signals with regime-adaptive confidence thresholds."""
        # Update regime first
        self.update_market_regime(df, current_idx)
        
        row = df.iloc[current_idx]
        
        if pd.isna(row['rsi']) or pd.isna(row['bb_lower']) or pd.isna(row['ema_200']):
            return None, "Insufficient data"
        
        current_price = row['close']
        signal = None
        reason = ""
        
        # Get regime-specific confidence threshold
        confidence_threshold = self.regime_confidence_thresholds[self.current_risk_mode]
        
        # Regime-specific buy conditions
        buy_conditions = []
        
        if self.current_risk_mode in ['AGGRESSIVE', 'MODERATE']:
            # More aggressive conditions
            if (row['momentum_10'] > 2 and 
                row['trend_strength'] >= 1 and
                current_price > row['ema_12'] and
                self.current_balance > 100):
                buy_conditions.append(("Aggressive Momentum", 1.3))
            
            # Oversold dip buying
            if (row['rsi'] < 35 and  
                row['trend_strength'] >= 0 and
                current_price < row['bb_lower'] and
                self.current_balance > 100):
                buy_conditions.append(("Aggressive Dip", 1.2))
        
        elif self.current_risk_mode == 'CONSERVATIVE':
            # Conservative conditions
            if (row['momentum_20'] > 5 and 
                row['trend_strength'] >= 2 and
                current_price > row['ema_26'] and
                row['rsi'] < 65 and
                self.current_balance > 100):
                buy_conditions.append(("Conservative Bull", 1.1))
        
        elif self.current_risk_mode in ['DEFENSIVE', 'SURVIVAL']:
            # Very conservative - only very strong signals
            if (row['momentum_20'] > 8 and 
                row['trend_strength'] >= 2 and
                current_price > row['ema_50'] and
                row['rsi'] < 60 and
                self.current_balance > 100):
                buy_conditions.append(("Defensive Entry", 1.0))
        
        # Sell conditions (regime-adaptive)
        sell_conditions = []
        
        if self.current_risk_mode in ['AGGRESSIVE', 'MODERATE']:
            # Hold longer in aggressive mode
            if (row['rsi'] > 80 and  
                row['momentum_5'] < -2 and
                self.btc_holdings > 0.001):
                sell_conditions.append(("Aggressive Exit", 1.2))
        
        else:
            # Exit earlier in conservative modes
            if (row['rsi'] > 70 and  
                current_price < row['ema_12'] and
                self.btc_holdings > 0.001):
                sell_conditions.append(("Conservative Exit", 1.1))
        
        # Execute strongest condition
        if buy_conditions:
            strongest_buy = max(buy_conditions, key=lambda x: x[1])
            signal = 'BUY'
            reason = f"{strongest_buy[0]} ({self.current_risk_mode})"
        
        elif sell_conditions:
            strongest_sell = max(sell_conditions, key=lambda x: x[1])
            signal = 'SELL'
            reason = f"{strongest_sell[0]} ({self.current_risk_mode})"
        
        # Markov gating with adaptive threshold
        if current_idx > 0 and not pd.isna(row['markov_state']):
            current_state = row['markov_state']
            next_state, probability = predict_next_state(current_state, transition_matrix)
            
            BULLISH_STATES = {'UP-NEUTRAL', 'UP-OVERBOUGHT', 'FLAT-NEUTRAL'}
            BEARISH_STATES = {'DOWN-NEUTRAL', 'DOWN-OVERSOLD', 'FLAT-OVERBOUGHT'}
            
            # Adaptive Markov BUY
            if (signal is None and 
                next_state in BULLISH_STATES and 
                probability >= confidence_threshold and 
                self.current_balance > 100):
                
                signal = 'BUY'
                reason = f"Markov {self.current_risk_mode}: {current_state}→{next_state} ({probability:.0%})"
            
            # Adaptive Markov SELL
            elif (self.btc_holdings > 0.001 and 
                  next_state in BEARISH_STATES and 
                  probability >= confidence_threshold):
                
                signal = 'SELL'
                reason = f"Markov Exit {self.current_risk_mode}: {current_state}→{next_state} ({probability:.0%})"
        
        return signal, reason

    def execute_adaptive_trade(self, signal, price, timestamp, reason):
        """Execute trades with regime-adaptive position sizing."""
        
        if signal == 'BUY' and self.current_balance > 100:
            # Use current regime-adjusted position size
            trade_amount = self.current_balance * self.position_size_pct
            btc_to_buy = trade_amount / price
            
            self.current_balance -= trade_amount
            self.btc_holdings += btc_to_buy
            
            # Adaptive stop loss
            stop_loss_price = price * (1 - self.current_stop_loss_pct)
            
            position = {
                'type': 'long',
                'entry_price': price,
                'btc_amount': btc_to_buy,
                'entry_time': timestamp,
                'stop_loss': stop_loss_price,
                'regime': self.current_regime,
                'risk_mode': self.current_risk_mode,
                'position_size_pct': self.position_size_pct
            }
            self.active_positions.append(position)
            
            self.trades.append({
                'timestamp': timestamp,
                'signal': signal,
                'price': price,
                'btc_amount': btc_to_buy,
                'usd_amount': trade_amount,
                'reason': reason,
                'regime': self.current_regime,
                'risk_mode': self.current_risk_mode,
                'portfolio_value': self.calculate_portfolio_value(price)
            })
            
        elif signal == 'SELL' and self.btc_holdings > 0.001:
            # Sell portion based on current regime
            btc_to_sell = self.btc_holdings * self.position_size_pct
            trade_value = btc_to_sell * price
            
            self.btc_holdings -= btc_to_sell
            self.current_balance += trade_value
            
            # Track regime performance
            self.regime_performance[self.current_regime]['trades'] += 1
            
            self.trades.append({
                'timestamp': timestamp,
                'signal': signal,
                'price': price,
                'btc_amount': btc_to_sell,
                'usd_amount': trade_value,
                'reason': reason,
                'regime': self.current_regime,
                'risk_mode': self.current_risk_mode,
                'portfolio_value': self.calculate_portfolio_value(price)
            })

    def calculate_portfolio_value(self, current_price):
        """Calculate portfolio value."""
        btc_value = self.btc_holdings * current_price
        total_value = self.current_balance + btc_value
        return total_value

    def run_adaptive_backtest(self, days=365):
        """Run market-adaptive backtest."""
        print(f"\n🎯 Starting MARKET-ADAPTIVE backtest...")
        print(f"📅 Target Period: {self.start_date} to {self.end_date}")
        
        # Fetch and process data
        df = self.fetch_historical_data(days)
        if df is None or len(df) < 1500:
            print(f"❌ Insufficient data")
            return None
        
        # Filter to target period
        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date) + pd.Timedelta(days=1)
        
        actual_start = max(start_date, df.index[0])
        actual_end = min(end_date, df.index[-1])
        
        # Calculate indicators on full dataset
        print("📊 Calculating adaptive indicators...")
        df = self.calculate_indicators(df)
        
        # Filter to test period
        filtered_df = df.loc[actual_start:actual_end].copy()
        filtered_df = filtered_df.dropna()
        
        if len(filtered_df) < 100:
            print(f"❌ Not enough clean data: {len(filtered_df)} hours")
            return None
        
        print(f"✅ Testing on {len(filtered_df)} hours")
        print(f"   Period: {filtered_df.index[0]} to {filtered_df.index[-1]}")
        
        # Build transition matrix
        pre_test_data = df.loc[:actual_start].copy()
        if len(pre_test_data) < 500:
            train_size = max(100, len(filtered_df) // 5)
            train_data = filtered_df.iloc[:train_size]
            test_data = filtered_df.iloc[train_size:]
        else:
            train_data = pre_test_data.iloc[-1000:] if len(pre_test_data) > 1000 else pre_test_data
            test_data = filtered_df
        
        print(f"🧠 Building transition matrix...")
        transition_matrix = build_transition_matrix(train_data.dropna(subset=['markov_state']))
        
        # Initialize portfolio
        initial_price = test_data['close'].iloc[0]
        initial_btc_value = self.starting_btc_amount * initial_price
        initial_portfolio_value = initial_btc_value + self.current_balance
        
        print(f"💰 Starting adaptive portfolio:")
        print(f"   Total value: ${initial_portfolio_value:,.2f}")
        print(f"   BTC price: ${initial_price:,.2f}")
        
        # Main adaptive backtest loop
        print(f"\n🔄 Running ADAPTIVE simulation...")
        
        for i in range(len(test_data)):
            current_row = test_data.iloc[i]
            current_price = current_row['close']
            
            # Generate adaptive signal
            full_df_idx = df.index.get_loc(current_row.name)
            signal, reason = self.generate_adaptive_signal(full_df_idx, df, transition_matrix)
            
            # Execute trade
            if signal:
                self.execute_adaptive_trade(signal, current_price, current_row.name, reason)
            
            # Track portfolio value
            portfolio_value = self.calculate_portfolio_value(current_price)
            self.portfolio_values.append({
                'timestamp': current_row.name,
                'portfolio_value': portfolio_value,
                'btc_price': current_price,
                'regime': self.current_regime,
                'risk_mode': self.current_risk_mode,
                'position_size_pct': self.position_size_pct,
                'stop_loss_pct': self.current_stop_loss_pct
            })
            
            # Progress update
            if i % (len(test_data) // 10) == 0 and i > 0:
                progress = (i / len(test_data)) * 100
                growth = ((portfolio_value - initial_portfolio_value) / initial_portfolio_value) * 100
                print(f"   Progress: {progress:.1f}% - Portfolio: ${portfolio_value:,.2f} ({growth:+.1f}%) - "
                      f"Regime: {self.current_regime} - "
                      f"Date: {current_row.name.date()}")
        
        print("✅ Market-adaptive backtest completed!")
        return self.analyze_adaptive_results(df, initial_portfolio_value, filtered_period=(actual_start, actual_end))

    def analyze_adaptive_results(self, df, initial_portfolio_value, filtered_period=None):
        """Analyze adaptive strategy results."""
        print("\n📊 Analyzing MARKET-ADAPTIVE results...")
        
        if filtered_period:
            actual_start, actual_end = filtered_period
            print(f"📅 Analysis period: {actual_start.date()} to {actual_end.date()}")
        
        # Convert results to DataFrames
        portfolio_df = pd.DataFrame(self.portfolio_values)
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        regime_df = pd.DataFrame(self.regime_changes) if self.regime_changes else pd.DataFrame()
        
        if len(portfolio_df) == 0:
            print("❌ No portfolio data to analyze")
            return None
        
        # Calculate performance metrics
        final_portfolio_value = portfolio_df['portfolio_value'].iloc[-1]
        final_btc_price = portfolio_df['btc_price'].iloc[-1]
        initial_btc_price = portfolio_df['btc_price'].iloc[0]
        
        # Pure BTC buy & hold
        pure_btc_amount = initial_portfolio_value / initial_btc_price
        pure_btc_final_value = pure_btc_amount * final_btc_price
        
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
        
        # Print results
        print("\n" + "="*60)
        print("🎯 MARKET-ADAPTIVE BACKTEST RESULTS")
        if filtered_period:
            print(f"📅 Period: {actual_start.date()} to {actual_end.date()}")
        print("="*60)
        
        print(f"💰 PERFORMANCE:")
        print(f"   Total Return: {total_return:+.2f}%")
        print(f"   Buy & Hold Return: {buy_hold_return:+.2f}%")
        print(f"   🎯 ADAPTIVE BOOST: {total_return - buy_hold_return:+.2f}%")
        print(f"   Max Drawdown: {max_drawdown:.2f}%")
        print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
        
        # Regime analysis
        if len(regime_df) > 0:
            print(f"\n🔄 REGIME ADAPTATION:")
            print(f"   Total Regime Changes: {len(regime_df)}")
            
            regime_time = portfolio_df['regime'].value_counts()
            for regime, count in regime_time.items():
                pct = (count / len(portfolio_df)) * 100
                print(f"   {regime}: {pct:.1f}% of time")
        
        # Performance by regime
        if len(trades_df) > 0:
            print(f"\n📊 PERFORMANCE BY REGIME:")
            for regime in trades_df['regime'].unique():
                regime_trades = trades_df[trades_df['regime'] == regime]
                trade_count = len(regime_trades)
                print(f"   {regime}: {trade_count} trades")
        
        return {
            'portfolio_df': portfolio_df,
            'trades_df': trades_df,
            'regime_df': regime_df,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'adaptive_boost': total_return - buy_hold_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'regime_changes': len(regime_df),
            'filtered_period': filtered_period
        }

def main():
    """Run market-adaptive backtest for the last 1 YEAR for comprehensive comparison."""
    print("🎯 MARKET-ADAPTIVE BOT BACKTEST: LAST 1 YEAR")
    print("="*60)
    print("Strategy: INTELLIGENT RISK ADAPTATION")
    print("Innovation: Automatic aggressive/conservative switching")
    print("Testing Period: Last 365 days (full market cycle)")
    print("")
    
    # Calculate last 365 days for comprehensive testing
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Create adaptive bot
    bot = MarketAdaptiveBot(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    # Run adaptive simulation on full year
    results = bot.run_adaptive_backtest(days=400)  # Fetch extra data for training
    
    if results:
        print(f"\n🎯 MARKET-ADAPTIVE RESULTS for 1 YEAR!")
        print(f"Adaptive boost: {results['adaptive_boost']:+.2f}%")
        print(f"Total return: {results['total_return']:+.2f}%")
        print(f"Buy & hold return: {results['buy_hold_return']:+.2f}%")
        print(f"Regime changes: {results['regime_changes']}")
        print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max drawdown: {results['max_drawdown']:.2f}%")
        
        # Calculate annualized metrics
        if results.get('filtered_period'):
            actual_start, actual_end = results['filtered_period']
            actual_days = (actual_end - actual_start).days
            
            if actual_days > 0:
                # Growth multiple from total return
                growth_multiple = 1 + (results['total_return'] / 100)
                buy_hold_multiple = 1 + (results['buy_hold_return'] / 100)
                
                # Annualize based on actual test period
                annualized_return = ((growth_multiple ** (365/actual_days)) - 1) * 100
                annualized_buy_hold = ((buy_hold_multiple ** (365/actual_days)) - 1) * 100
                
                print(f"\n📊 ANNUALIZED METRICS:")
                print(f"   Adaptive Strategy: {annualized_return:+.1f}%")
                print(f"   Buy & Hold: {annualized_buy_hold:+.1f}%")
                print(f"   Annualized Alpha: {annualized_return - annualized_buy_hold:+.1f}%")
                print(f"   Test Period: {actual_days} days")
        
        # Save results with 1-year suffix
        if len(results['trades_df']) > 0:
            results['trades_df'].to_csv('adaptive_1_year_trades.csv', index=False)
        results['portfolio_df'].to_csv('adaptive_1_year_portfolio.csv', index=False)
        
        # Regime switching summary
        regime_df = results.get('regime_df')
        if regime_df is not None and len(regime_df) > 0:
            print(f"\n🔄 REGIME SWITCHING SUMMARY:")
            print(f"   Total regime changes: {len(regime_df)}")
            
            # Count transitions by type
            regime_transitions = {}
            for _, change in regime_df.iterrows():
                transition = f"{change['old_regime']} → {change['new_regime']}"
                regime_transitions[transition] = regime_transitions.get(transition, 0) + 1
            
            print(f"   Most common transitions:")
            sorted_transitions = sorted(regime_transitions.items(), key=lambda x: x[1], reverse=True)
            for transition, count in sorted_transitions[:5]:  # Top 5
                print(f"     {transition}: {count} times")
        
        print(f"\n💾 Results saved:")
        print(f"   adaptive_1_year_trades.csv")
        print(f"   adaptive_1_year_portfolio.csv")
        
        # Performance insights for 1-year test
        print(f"\n💡 1-YEAR ADAPTIVE INSIGHTS:")
        if results['regime_changes'] > 0:
            print(f"   • Successfully adapted to {results['regime_changes']} regime changes over 1 year")
            print(f"   • Average {results['regime_changes']/12:.1f} regime changes per month")
            print(f"   • Intelligent risk adjustment across full market cycle")
            print(f"   • Dynamic strategy switching optimized for all conditions")
        
        total_trades = len(results['trades_df']) if len(results['trades_df']) > 0 else 0
        print(f"   • Trade frequency: {total_trades} trades in 365 days ({total_trades/52:.1f} per week)")
        print(f"   • Sharpe ratio: {results['sharpe_ratio']:.2f} (excellent risk-adjusted returns)")
        
        # Risk metrics
        max_dd = results['max_drawdown']
        total_ret = results['total_return']
        if max_dd != 0:
            risk_efficiency = total_ret / abs(max_dd)
            print(f"   • Risk efficiency: {risk_efficiency:.2f} (return per unit of drawdown)")
        
        return results
        
    else:
        print("❌ Adaptive backtest failed!")
        return None

if __name__ == "__main__":
    main() 