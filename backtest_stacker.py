#!/usr/bin/env python3
"""
BTC STACKER STRATEGY BACKTESTER
==============================
Backtesting framework for the BTC accumulation strategy:
- 📊 Historical performance analysis
- 💎 BTC accumulation metrics
- 📈 Portfolio growth simulation
- 🔄 Transaction fee simulation
"""

import pandas as pd
import numpy as np
import ta
import ccxt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from typing import Dict, List, Tuple
import logging
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BTCStackerBacktest:
    def __init__(self, 
                 initial_usdt: float = 10000.0,
                 initial_btc: float = 0.0,
                 start_date: str = "2023-01-01",
                 end_date: str = "2024-01-01"):
        """Initialize backtester with starting conditions."""
        
        self.initial_usdt = initial_usdt
        self.initial_btc = initial_btc
        self.start_date = start_date
        self.end_date = end_date
        
        # Strategy Parameters
        self.target_btc_allocation = 0.75  # Target 75% portfolio in BTC
        self.rebalance_threshold = 0.1     # Rebalance at 10% deviation
        self.position_size_btc = 0.01      # Base position size
        self.max_position_btc = 0.1        # Maximum position size
        self.dca_enabled = True
        self.dca_interval_hours = 24
        
        # Trading Parameters
        self.trading_fee = 0.001  # 0.1% trading fee
        self.slippage = 0.001    # 0.1% slippage
        self.min_trade_usdt = 10
        
        # Performance Tracking
        self.trades = []
        self.portfolio_history = []
        self.btc_accumulated = 0
        self.total_fees_btc = 0
        self.total_fees_usdt = 0
        
        # For advanced metrics
        self.active_positions = deque()
        self.winning_trades = 0
        self.losing_trades = 0
        
        logger.info("🔄 Initializing BTC Stacker Backtest")
        logger.info(f"📅 Period: {start_date} to {end_date}")
        logger.info(f"💰 Initial USDT: ${initial_usdt}")
        logger.info(f"₿ Initial BTC: {initial_btc}")

    def fetch_historical_data(self, days: int = 365) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for backtesting.
        Tries multiple exchanges for resilience and fetches backwards in time.
        """
        logger.info(f"📊 Fetching historical data for approximately {days} days...")
        
        exchanges_to_try = [
            ('binance', 'BTC/USDT', 1000),
            ('coinbase', 'BTC/USDT', 300),
            ('kraken', 'BTC/USD', 720),
            ('gate', 'BTC/USDT', 1000)
        ]
        
        all_candles = []
        for exchange_name, symbol, limit in exchanges_to_try:
            try:
                logger.info(f"   Attempting to fetch from {exchange_name.upper()}...")
                exchange = getattr(ccxt, exchange_name)({'enableRateLimit': True})
                
                hours_needed = days * 24
                # Start fetching from now backwards
                from_ts = exchange.milliseconds()
                
                while len(all_candles) < hours_needed:
                    logger.info(f"   Fetching chunk of {limit} candles from {exchange_name.upper()}...")
                    candles = exchange.fetch_ohlcv(symbol, '1h', since=from_ts - limit * 60 * 60 * 1000, limit=limit)
                    
                    if not candles:
                        logger.warning(f"   No more data available from {exchange_name.upper()}.")
                        break
                    
                    # Prepend to maintain chronological order
                    all_candles = candles + all_candles
                    from_ts = candles[0][0]
                    
                    if len(all_candles) >= hours_needed:
                        break # We have enough data
                
                if len(all_candles) >= hours_needed * 0.9: # If we got most of what we need, success
                    logger.info(f"✅ Successfully fetched {len(all_candles)} candles from {exchange_name.upper()}.")
                    break
                else:
                    logger.warning(f"   Could not fetch sufficient data from {exchange_name.upper()}. Trying next exchange...")
                    all_candles = [] # Reset for next attempt
            
            except Exception as e:
                logger.error(f"   Failed to fetch from {exchange_name.upper()}: {e}")
                all_candles = [] # Reset for next attempt
                continue

        if not all_candles:
            logger.error("❌ Could not fetch any historical data from any exchange.")
            return pd.DataFrame()
            
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')] # Remove duplicates
        df = df.sort_index()

        logger.info(f"✅ Loaded {len(df)} unique hours of data from {df.index.min()} to {df.index.max()}")
        return df

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for trading signals."""
        try:
            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            
            # EMA 200
            df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
            
            # ATR
            df['atr'] = ta.volatility.AverageTrueRange(
                df['high'], df['low'], df['close'], window=14
            ).average_true_range()
            
            # Percentage changes
            df['pct_change'] = df['close'].pct_change()
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Indicator calculation failed: {e}")
            raise

    def close_positions_fifo(self, btc_to_sell: float, price: float) -> float:
        """Closes positions on a FIFO basis and calculates P&L."""
        btc_remaining_to_close = btc_to_sell
        total_pnl = 0

        while btc_remaining_to_close > 0 and self.active_positions:
            position = self.active_positions[0]
            amount_to_close_from_position = min(position['btc_amount'], btc_remaining_to_close)

            if amount_to_close_from_position > 0:
                pnl = (price - position['entry_price']) * amount_to_close_from_position
                total_pnl += pnl

                if pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1

                position['btc_amount'] -= amount_to_close_from_position
                btc_remaining_to_close -= amount_to_close_from_position

                if position['btc_amount'] < 1e-8: # Epsilon for float comparison
                    self.active_positions.popleft()
            else:
                # This should not happen if logic is correct
                break
        
        return total_pnl

    def generate_signal(self, row: pd.Series, portfolio: Dict) -> Tuple[str, str]:
        """Generate trading signal based on indicators and portfolio state."""
        signal = None
        reason = ""
        
        # Check portfolio balance
        btc_value_usdt = portfolio['btc'] * row['close']
        total_value_usdt = btc_value_usdt + portfolio['usdt']
        btc_allocation = btc_value_usdt / total_value_usdt if total_value_usdt > 0 else 0
        
        # Portfolio rebalancing check
        deviation = btc_allocation - self.target_btc_allocation
        if abs(deviation) > self.rebalance_threshold:
            if deviation < 0 and portfolio['usdt'] >= self.min_trade_usdt:
                return 'BUY', 'Portfolio rebalancing'
            elif deviation > 0 and portfolio['btc'] >= self.position_size_btc:
                return 'SELL', 'Portfolio rebalancing'
        
        # Technical analysis signals
        if (row['rsi'] < 30 and 
            row['close'] < row['bb_lower'] and
            row['close'] > row['ema_200'] * 0.95 and
            portfolio['usdt'] >= self.min_trade_usdt):
            
            signal = 'BUY'
            reason = 'Strong oversold condition'
            
        elif (row['rsi'] > 75 and
              row['close'] > row['bb_upper'] * 1.02 and
              portfolio['btc'] >= self.position_size_btc):
            
            signal = 'SELL'
            reason = 'Strong overbought condition'
        
        return signal, reason

    def execute_trade(self, 
                     signal: str, 
                     price: float, 
                     portfolio: Dict,
                     timestamp: pd.Timestamp,
                     reason: str) -> Dict:
        """Simulate trade execution with fees and slippage."""
        pnl = 0
        btc_amount = 0
        usdt_spent = 0
        usdt_received = 0
        btc_to_sell = 0
        
        if signal == 'BUY':
            # Calculate buy amount
            max_btc_purchase = portfolio['usdt'] / price
            btc_amount = min(
                max_btc_purchase * 0.95,  # Use 95% of available USDT
                self.max_position_btc
            )
            
            if btc_amount * price < self.min_trade_usdt:
                return portfolio
            
            # Apply fees and slippage
            execution_price = price * (1 + self.slippage)
            fee_btc = btc_amount * self.trading_fee
            
            # Update portfolio
            usdt_spent = btc_amount * execution_price
            portfolio['usdt'] -= usdt_spent
            btc_bought = btc_amount - fee_btc
            portfolio['btc'] += btc_bought
            
            # Add to active positions for P&L tracking
            self.active_positions.append({
                'entry_price': execution_price,
                'btc_amount': btc_bought,
                'timestamp': timestamp
            })
            
            # Track metrics
            self.btc_accumulated += btc_bought
            self.total_fees_btc += fee_btc
            self.total_fees_usdt += usdt_spent * self.trading_fee
            
        elif signal == 'SELL':
            # Calculate sell amount
            btc_to_sell = min(
                portfolio['btc'] * 0.25,  # Sell up to 25% of holdings
                self.position_size_btc
            )
            
            if btc_to_sell * price < self.min_trade_usdt:
                return portfolio
            
            # Apply fees and slippage
            execution_price = price * (1 - self.slippage)
            fee_usdt = btc_to_sell * execution_price * self.trading_fee
            
            # Update portfolio
            usdt_received = (btc_to_sell * execution_price) - fee_usdt
            portfolio['btc'] -= btc_to_sell
            portfolio['usdt'] += usdt_received
            
            # Close positions and calculate P&L (FIFO)
            pnl = self.close_positions_fifo(btc_to_sell, execution_price)
            
            # Track metrics (subtract sold BTC from accumulated total)
            self.btc_accumulated -= btc_to_sell
            self.total_fees_usdt += fee_usdt
        
        # Log trade
        self.trades.append({
            'timestamp': timestamp,
            'type': signal,
            'price': price,
            'btc_amount': btc_amount if signal == 'BUY' else btc_to_sell,
            'usdt_value': usdt_spent if signal == 'BUY' else usdt_received,
            'reason': reason,
            'pnl': pnl,
            'portfolio_btc': portfolio['btc'],
            'portfolio_usdt': portfolio['usdt']
        })
        
        return portfolio

    def run_backtest(self) -> Dict:
        """Execute backtest simulation."""
        try:
            # Fetch a large chunk of historical data with a buffer for indicators
            start_dt = pd.to_datetime(self.start_date)
            end_dt = pd.to_datetime(self.end_date)
            days_to_fetch = (datetime.now() - start_dt).days + 2 # Fetch up to today
            
            full_df = self.fetch_historical_data(days=days_to_fetch)
            if full_df.empty:
                raise ValueError("Failed to fetch historical data.")

            # Calculate indicators on the full dataset to ensure accuracy
            full_df = self.calculate_indicators(full_df)
            
            # Filter the dataframe to the precise backtest period
            df = full_df.loc[start_dt:end_dt].copy()
            if df.empty:
                logger.error(f"No data available for the specified date range: {self.start_date} to {self.end_date}")
                raise ValueError("Date range not covered by fetched data.")

            # Initialize portfolio
            portfolio = {
                'btc': self.initial_btc,
                'usdt': self.initial_usdt
            }
            
            # Initialize metrics
            last_dca_time = None
            trades_today = 0
            last_trade_time = None
            current_date = None
            
            logger.info("🚀 Starting backtest simulation...")
            
            # Simulation loop
            for timestamp, row in df.iterrows():
                # Reset daily counters
                if current_date != timestamp.date():
                    trades_today = 0
                    current_date = timestamp.date()
                
                # Skip if not enough indicator data
                if pd.isna(row['rsi']) or pd.isna(row['bb_lower']) or pd.isna(row['ema_200']):
                    continue
                
                # Check DCA
                if self.dca_enabled:
                    if (last_dca_time is None or 
                        (timestamp - last_dca_time).total_seconds() >= self.dca_interval_hours * 3600):
                        
                        if portfolio['usdt'] >= self.min_trade_usdt:
                            portfolio = self.execute_trade(
                                'BUY', row['close'], portfolio, timestamp, 'DCA Buy'
                            )
                            last_dca_time = timestamp
                
                # Check trading signals
                if trades_today < 10:  # Max 10 trades per day
                    if (last_trade_time is None or 
                        (timestamp - last_trade_time).total_seconds() >= 14400):  # 4-hour cooldown
                        
                        signal, reason = self.generate_signal(row, portfolio)
                        
                        if signal:
                            portfolio = self.execute_trade(
                                signal, row['close'], portfolio, timestamp, reason
                            )
                            trades_today += 1
                            last_trade_time = timestamp
                
                # Track portfolio value
                total_value_usdt = portfolio['usdt'] + (portfolio['btc'] * row['close'])
                total_value_btc = portfolio['btc'] + (portfolio['usdt'] / row['close'])
                
                self.portfolio_history.append({
                    'timestamp': timestamp,
                    'price': row['close'],
                    'portfolio_usdt': total_value_usdt,
                    'portfolio_btc': total_value_btc,
                    'btc_holdings': portfolio['btc'],
                    'usdt_holdings': portfolio['usdt']
                })
            
            # Calculate performance metrics
            results = self.calculate_performance()
            self.plot_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Backtest failed: {e}")
            raise

    def calculate_performance(self) -> Dict:
        """Calculate backtest performance metrics."""
        if not self.portfolio_history:
            return {}
            
        df = pd.DataFrame(self.portfolio_history)
        df.set_index('timestamp', inplace=True)
        trades_df = pd.DataFrame(self.trades)
        
        # --- Core Returns ---
        initial_value_usdt = self.initial_usdt + (self.initial_btc * df['price'].iloc[0])
        final_value_usdt = df['portfolio_usdt'].iloc[-1]
        usdt_return_pct = ((final_value_usdt - initial_value_usdt) / initial_value_usdt) * 100

        initial_value_btc = self.initial_btc + (self.initial_usdt / df['price'].iloc[0])
        final_value_btc = df['portfolio_btc'].iloc[-1]
        btc_return_pct = ((final_value_btc - initial_value_btc) / initial_value_btc) * 100 if initial_value_btc > 0 else 0

        # --- Buy & Hold Calculation ---
        buy_hold_btc = initial_value_usdt / df['price'].iloc[0]
        buy_hold_final_value = buy_hold_btc * df['price'].iloc[-1]
        buy_hold_return_pct = ((buy_hold_final_value - initial_value_usdt) / initial_value_usdt) * 100 if initial_value_usdt > 0 else 0

        # --- Advanced Metrics ---
        df['daily_return'] = df['portfolio_usdt'].pct_change()
        # Annualized for hourly data
        sharpe_ratio = (df['daily_return'].mean() / df['daily_return'].std()) * np.sqrt(24 * 365) if df['daily_return'].std() != 0 else 0

        df['running_max'] = df['portfolio_usdt'].expanding().max()
        df['drawdown'] = (df['portfolio_usdt'] - df['running_max']) / df['running_max']
        max_drawdown = df['drawdown'].min() * 100

        total_closed_trades = self.winning_trades + self.losing_trades
        win_rate = (self.winning_trades / total_closed_trades) * 100 if total_closed_trades > 0 else 0
        
        total_days = (df.index.max() - df.index.min()).days
        annualized_return = ((1 + (usdt_return_pct / 100)) ** (365 / total_days) - 1) * 100 if total_days > 0 else usdt_return_pct

        return {
            'start_date': df.index.min(),
            'end_date': df.index.max(),
            'total_trades': len(self.trades),
            'initial_portfolio_usdt': initial_value_usdt,
            'final_portfolio_usdt': final_value_usdt,
            'usdt_return_pct': usdt_return_pct,
            'buy_and_hold_return_pct': buy_hold_return_pct,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'win_rate_pct': win_rate,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'annualized_return_pct': annualized_return,
            'initial_portfolio_btc': initial_value_btc,
            'final_portfolio_btc': final_value_btc,
            'btc_return_pct': btc_return_pct,
            'btc_accumulated': self.btc_accumulated,
            'total_fees_usdt': self.total_fees_usdt,
            'buy_trades': len(trades_df[trades_df['type'] == 'BUY']),
            'sell_trades': len(trades_df[trades_df['type'] == 'SELL']),
            'portfolio_df': df # Pass dataframe for plotting
        }

    def plot_results(self, results: Dict):
        """Generate performance visualization plots."""
        df = results['portfolio_df']
        
        # Set style
        plt.style.use('default')
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 20), sharex=True)
        
        # Plot 1: Portfolio Value in USDT
        ax1.plot(df.index, df['portfolio_usdt'], label='Portfolio Value', color='blue')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(df.index, df['price'], label='BTC Price', color='grey', alpha=0.5)
        ax1.set_title('Total Portfolio Value (USDT)')
        ax1.set_ylabel('USDT Value')
        ax1_twin.set_ylabel('BTC Price')
        ax1.grid(True)
        
        # Plot 2: BTC Holdings
        ax2.plot(df.index, df['btc_holdings'], label='BTC Holdings', color='orange')
        ax2.set_title('BTC Holdings Over Time')
        ax2.set_ylabel('BTC')
        ax2.grid(True)
        
        # Plot 3: BTC Price with Trades
        ax3.plot(df.index, df['price'], label='BTC Price', color='green', alpha=0.7)
        ax3.set_title('BTC Price with Trades')
        ax3.set_ylabel('USDT')
        ax3.grid(True)
        
        # Add trades to plots
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            buys = trades_df[trades_df['type'] == 'BUY']
            sells = trades_df[trades_df['type'] == 'SELL']
            
            ax3.scatter(buys['timestamp'], buys['price'], 
                       color='green', marker='^', s=50, label='Buy', alpha=0.8)
            ax3.scatter(sells['timestamp'], sells['price'], 
                       color='red', marker='v', s=50, label='Sell', alpha=0.8)
            ax3.legend()
        
        # Plot 4: Drawdown
        ax4.fill_between(df.index, df['drawdown']*100, 0,
                        color='red', alpha=0.3, label='Drawdown')
        ax4.set_title('Portfolio Drawdown')
        ax4.set_ylabel('Drawdown (%)')
        ax4.set_xlabel('Date')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png')
        logger.info("📊 Performance plots saved as 'backtest_results.png'")

def main():
    """Run backtest with default parameters."""
    print("🔄 BTC STACKER STRATEGY BACKTEST")
    print("=" * 50)
    
    # Initialize and run backtest
    backtest = BTCStackerBacktest(
        initial_usdt=10000.0,
        initial_btc=0.0,
        start_date="2024-06-01",
        end_date="2025-06-01"
    )
    
    try:
        results = backtest.run_backtest()
        
        # Print results
        print("\n📊 BACKTEST RESULTS")
        print("=" * 50)
        print(f"Period: {results['start_date'].date()} to {results['end_date'].date()}")
        print(f"Total Trades: {results['total_trades']} ({results['buy_trades']} buys, {results['sell_trades']} sells)")

        print("\n--- PERFORMANCE ---")
        print(f"Final Portfolio Value: ${results['final_portfolio_usdt']:.2f}")
        print(f"Strategy Return: {results['usdt_return_pct']:.2f}%")
        print(f"Buy & Hold Return: {results['buy_and_hold_return_pct']:.2f}%")
        print(f"Annualized Return: {results['annualized_return_pct']:.2f}%")

        print("\n--- RISK & RATIOS ---")
        print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio (annualized): {results['sharpe_ratio']:.2f}")
        print(f"Win Rate: {results['win_rate_pct']:.2f}% ({results['winning_trades']} wins, {results['losing_trades']} losses)")

        print("\n--- BTC METRICS ---")
        print(f"Final BTC Holdings: {results['final_portfolio_btc']:.8f} BTC")
        print(f"Net BTC Accumulated: {results['btc_accumulated']:.8f} BTC")
        print(f"Total Fees Paid (USDT): ${results['total_fees_usdt']:.2f}")
        
    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        raise

if __name__ == "__main__":
    main() 