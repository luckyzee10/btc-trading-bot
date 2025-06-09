#!/usr/bin/env python3
"""
ADAPTIVE BOT VISUALIZATION SUITE
================================
Comprehensive charts and graphs to visualize the Market-Adaptive Bot's performance:
- Portfolio performance vs Buy & Hold
- Regime switching visualization
- Drawdown analysis
- Trading activity by regime
- Dynamic position sizing
- Risk metrics over time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_adaptive_data():
    """Load adaptive bot data for visualization."""
    try:
        portfolio_df = pd.read_csv('adaptive_1_year_portfolio.csv')
        trades_df = pd.read_csv('adaptive_1_year_trades.csv')
        
        # Convert timestamps
        portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        print("✅ Successfully loaded adaptive bot data")
        print(f"   Portfolio records: {len(portfolio_df)}")
        print(f"   Trades: {len(trades_df)}")
        print(f"   Period: {portfolio_df['timestamp'].min().date()} to {portfolio_df['timestamp'].max().date()}")
        
        return portfolio_df, trades_df
        
    except FileNotFoundError:
        print("❌ Could not find adaptive bot results files!")
        print("   Please run the adaptive bot first with: python3 market_adaptive_bot.py")
        return None, None

def create_performance_chart(portfolio_df):
    """Create portfolio performance vs buy & hold chart."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Calculate buy & hold performance
    initial_portfolio_value = portfolio_df['portfolio_value'].iloc[0]
    initial_btc_price = portfolio_df['btc_price'].iloc[0]
    pure_btc_amount = initial_portfolio_value / initial_btc_price
    portfolio_df['buy_hold_value'] = pure_btc_amount * portfolio_df['btc_price']
    
    # Calculate returns
    portfolio_df['adaptive_return'] = (portfolio_df['portfolio_value'] / initial_portfolio_value - 1) * 100
    portfolio_df['buy_hold_return'] = (portfolio_df['buy_hold_value'] / initial_portfolio_value - 1) * 100
    
    # Main performance chart
    ax1.plot(portfolio_df['timestamp'], portfolio_df['adaptive_return'], 
             label='🔄 Adaptive Strategy', linewidth=2, color='#2E86AB')
    ax1.plot(portfolio_df['timestamp'], portfolio_df['buy_hold_return'], 
             label='📊 Buy & Hold', linewidth=2, color='#A23B72', linestyle='--')
    
    ax1.set_title('🏆 Market-Adaptive Bot vs Buy & Hold Performance', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Return (%)', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add final performance text
    final_adaptive = portfolio_df['adaptive_return'].iloc[-1]
    final_buy_hold = portfolio_df['buy_hold_return'].iloc[-1]
    alpha = final_adaptive - final_buy_hold
    
    ax1.text(0.02, 0.98, f'Final Returns:\n🔄 Adaptive: {final_adaptive:+.1f}%\n📊 Buy & Hold: {final_buy_hold:+.1f}%\n⚡ Alpha: {alpha:+.1f}%', 
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Portfolio value chart
    ax2.plot(portfolio_df['timestamp'], portfolio_df['portfolio_value'], 
             label='🔄 Adaptive Portfolio Value', linewidth=2, color='#F18F01')
    ax2.plot(portfolio_df['timestamp'], portfolio_df['buy_hold_value'], 
             label='📊 Buy & Hold Value', linewidth=2, color='#C73E1D', linestyle='--')
    
    ax2.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig('adaptive_performance_chart.png', dpi=300, bbox_inches='tight')
    print("📊 Saved: adaptive_performance_chart.png")
    return fig

def create_regime_chart(portfolio_df):
    """Create regime switching visualization."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 14))
    
    # Regime colors
    regime_colors = {
        'SIDEWAYS': '#95A5A6',
        'BULL_LOW_VOL': '#27AE60', 
        'BULL_HIGH_VOL': '#F39C12',
        'BEAR_LOW_VOL': '#E74C3C',
        'BEAR_HIGH_VOL': '#8E44AD',
        'UNKNOWN': '#BDC3C7'
    }
    
    # Top: BTC Price with regime background
    ax1.plot(portfolio_df['timestamp'], portfolio_df['btc_price'], 
             linewidth=2, color='#F39C12', label='BTC Price')
    
    # Color background by regime
    current_regime = None
    start_idx = 0
    
    for i, row in portfolio_df.iterrows():
        regime = row.get('regime', 'UNKNOWN')
        if regime != current_regime:
            if current_regime is not None:
                # Fill previous regime
                end_timestamp = portfolio_df['timestamp'].iloc[i-1] if i > 0 else row['timestamp']
                ax1.axvspan(portfolio_df['timestamp'].iloc[start_idx], end_timestamp, 
                           alpha=0.2, color=regime_colors.get(current_regime, '#BDC3C7'))
            current_regime = regime
            start_idx = i
    
    # Fill last regime
    if current_regime is not None:
        ax1.axvspan(portfolio_df['timestamp'].iloc[start_idx], portfolio_df['timestamp'].iloc[-1], 
                   alpha=0.2, color=regime_colors.get(current_regime, '#BDC3C7'))
    
    ax1.set_title('🔄 Market Regime Detection & BTC Price', fontsize=16, fontweight='bold')
    ax1.set_ylabel('BTC Price ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Create regime legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.3, label=regime) 
                      for regime, color in regime_colors.items() if regime in portfolio_df.get('regime', []).unique()]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Middle: Position sizing over time
    if 'position_size_pct' in portfolio_df.columns:
        ax2.fill_between(portfolio_df['timestamp'], 0, portfolio_df['position_size_pct'] * 100,
                        alpha=0.6, color='#3498DB', label='Position Size %')
        ax2.set_title('⚖️ Dynamic Position Sizing', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Position Size (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Add position size statistics
        if not portfolio_df['position_size_pct'].isna().all():
            avg_pos_size = portfolio_df['position_size_pct'].mean() * 100
            max_pos_size = portfolio_df['position_size_pct'].max() * 100
            min_pos_size = portfolio_df['position_size_pct'].min() * 100
            
            ax2.text(0.02, 0.98, f'Position Size Stats:\nAvg: {avg_pos_size:.1f}%\nMax: {max_pos_size:.1f}%\nMin: {min_pos_size:.1f}%', 
                    transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Bottom: Portfolio heat/risk over time
    if 'portfolio_heat' in portfolio_df.columns:
        ax3.fill_between(portfolio_df['timestamp'], 0, portfolio_df['portfolio_heat'] * 100,
                        alpha=0.6, color='#E74C3C', label='Portfolio Heat %')
        ax3.axhline(y=25, color='red', linestyle='--', alpha=0.7, label='Max Risk Threshold')
        ax3.set_title('🌡️ Portfolio Heat (Risk Level)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_ylabel('Portfolio Heat (%)', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 30)
    
    plt.tight_layout()
    plt.savefig('adaptive_regime_chart.png', dpi=300, bbox_inches='tight')
    print("📊 Saved: adaptive_regime_chart.png")
    return fig

def create_drawdown_chart(portfolio_df):
    """Create drawdown analysis chart."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Calculate drawdowns
    portfolio_df['running_max'] = portfolio_df['portfolio_value'].expanding().max()
    portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['running_max']) / portfolio_df['running_max'] * 100
    
    # Top: Portfolio value with drawdown periods
    ax1.plot(portfolio_df['timestamp'], portfolio_df['portfolio_value'], 
             linewidth=2, color='#2E86AB', label='Portfolio Value')
    ax1.plot(portfolio_df['timestamp'], portfolio_df['running_max'], 
             linewidth=1, color='#27AE60', linestyle='--', alpha=0.7, label='Running Maximum')
    
    # Highlight drawdown periods
    ax1.fill_between(portfolio_df['timestamp'], portfolio_df['portfolio_value'], portfolio_df['running_max'],
                    where=(portfolio_df['drawdown'] < 0), alpha=0.3, color='red', label='Drawdown Periods')
    
    ax1.set_title('🛡️ Drawdown Analysis - Portfolio Protection', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Bottom: Drawdown percentage over time
    ax2.fill_between(portfolio_df['timestamp'], 0, portfolio_df['drawdown'],
                    alpha=0.6, color='#E74C3C')
    ax2.plot(portfolio_df['timestamp'], portfolio_df['drawdown'], 
             linewidth=1, color='#C0392B')
    
    max_drawdown = portfolio_df['drawdown'].min()
    ax2.axhline(y=max_drawdown, color='red', linestyle='--', alpha=0.8, 
               label=f'Max Drawdown: {max_drawdown:.1f}%')
    
    ax2.set_title('Drawdown Percentage Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(max_drawdown * 1.1, 5)  # Add some padding
    
    # Add drawdown statistics
    avg_drawdown = portfolio_df[portfolio_df['drawdown'] < 0]['drawdown'].mean()
    drawdown_periods = (portfolio_df['drawdown'] < -1).sum()  # Periods with >1% drawdown
    
    ax2.text(0.02, 0.02, f'Drawdown Stats:\nMax: {max_drawdown:.1f}%\nAvg (when in DD): {avg_drawdown:.1f}%\nPeriods >1% DD: {drawdown_periods}', 
            transform=ax2.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('adaptive_drawdown_chart.png', dpi=300, bbox_inches='tight')
    print("📊 Saved: adaptive_drawdown_chart.png")
    return fig

def create_trading_analysis_chart(trades_df):
    """Create trading activity analysis charts."""
    if len(trades_df) == 0:
        print("⚠️ No trades data available for analysis")
        return None
        
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Trades by regime (if available)
    if 'regime' in trades_df.columns:
        regime_counts = trades_df['regime'].value_counts()
        colors = ['#3498DB', '#E74C3C', '#F39C12', '#27AE60', '#9B59B6']
        
        wedges, texts, autotexts = ax1.pie(regime_counts.values, labels=regime_counts.index, 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('🔄 Trades by Market Regime', fontsize=14, fontweight='bold')
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    # 2. Trade signal distribution
    if 'signal' in trades_df.columns:
        signal_counts = trades_df['signal'].value_counts()
        ax2.bar(signal_counts.index, signal_counts.values, 
               color=['#27AE60', '#E74C3C', '#F39C12'])
        ax2.set_title('📊 Trade Signal Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Trades', fontsize=12)
        
        # Add count labels on bars
        for i, v in enumerate(signal_counts.values):
            ax2.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
    
    # 3. Portfolio value over time with trade markers
    if 'portfolio_value' in trades_df.columns:
        # Plot all trades as scatter
        buy_trades = trades_df[trades_df['signal'] == 'BUY']
        sell_trades = trades_df[trades_df['signal'] == 'SELL']
        
        if len(buy_trades) > 0:
            ax3.scatter(buy_trades['timestamp'], buy_trades['portfolio_value'], 
                       color='green', alpha=0.7, s=30, label='BUY Trades', marker='^')
        if len(sell_trades) > 0:
            ax3.scatter(sell_trades['timestamp'], sell_trades['portfolio_value'], 
                       color='red', alpha=0.7, s=30, label='SELL Trades', marker='v')
        
        ax3.set_title('💹 Trade Execution Points', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_ylabel('Portfolio Value at Trade ($)', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # 4. Monthly trade distribution
    trades_df['month'] = trades_df['timestamp'].dt.to_period('M')
    monthly_trades = trades_df.groupby('month').size()
    
    ax4.bar(range(len(monthly_trades)), monthly_trades.values, color='#3498DB', alpha=0.7)
    ax4.set_title('📅 Monthly Trade Distribution', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Month', fontsize=12)
    ax4.set_ylabel('Number of Trades', fontsize=12)
    ax4.set_xticks(range(len(monthly_trades)))
    ax4.set_xticklabels([str(m) for m in monthly_trades.index], rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add average line
    avg_trades = monthly_trades.mean()
    ax4.axhline(y=avg_trades, color='red', linestyle='--', alpha=0.7, 
               label=f'Average: {avg_trades:.1f} trades/month')
    ax4.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('adaptive_trading_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Saved: adaptive_trading_analysis.png")
    return fig

def create_performance_metrics_chart(portfolio_df, trades_df):
    """Create performance metrics visualization."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Daily returns distribution
    portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change() * 100
    daily_returns = portfolio_df['daily_return'].dropna()
    
    ax1.hist(daily_returns, bins=50, alpha=0.7, color='#3498DB', edgecolor='black')
    ax1.axvline(daily_returns.mean(), color='red', linestyle='--', 
               label=f'Mean: {daily_returns.mean():.3f}%')
    ax1.axvline(daily_returns.median(), color='green', linestyle='--', 
               label=f'Median: {daily_returns.median():.3f}%')
    ax1.set_title('📊 Daily Returns Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Daily Return (%)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Rolling Sharpe ratio
    window = 30  # 30-period rolling window
    if len(daily_returns) > window:
        rolling_sharpe = daily_returns.rolling(window=window).apply(
            lambda x: x.mean() / x.std() * np.sqrt(24*365) if x.std() > 0 else 0
        )
        
        # Ensure arrays have same length by using valid rolling_sharpe indices
        valid_rolling_sharpe = rolling_sharpe.dropna()
        valid_timestamps = portfolio_df['timestamp'].iloc[valid_rolling_sharpe.index]
        
        ax2.plot(valid_timestamps, valid_rolling_sharpe, 
                linewidth=2, color='#E74C3C')
        ax2.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Good (>1.0)')
        ax2.axhline(y=2, color='blue', linestyle='--', alpha=0.7, label='Excellent (>2.0)')
        ax2.set_title(f'📈 Rolling Sharpe Ratio ({window}-period)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Sharpe Ratio', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    # 3. Portfolio composition over time
    if 'btc_holdings' in portfolio_df.columns and 'cash_balance' in portfolio_df.columns:
        portfolio_df['btc_value'] = portfolio_df['btc_holdings'] * portfolio_df['btc_price']
        portfolio_df['cash_pct'] = portfolio_df['cash_balance'] / portfolio_df['portfolio_value'] * 100
        portfolio_df['btc_pct'] = portfolio_df['btc_value'] / portfolio_df['portfolio_value'] * 100
        
        ax3.fill_between(portfolio_df['timestamp'], 0, portfolio_df['cash_pct'], 
                        alpha=0.7, color='#F39C12', label='Cash %')
        ax3.fill_between(portfolio_df['timestamp'], portfolio_df['cash_pct'], 100,
                        alpha=0.7, color='#E74C3C', label='BTC %')
        
        ax3.set_title('💰 Portfolio Composition Over Time', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_ylabel('Allocation (%)', fontsize=12)
        ax3.legend(fontsize=12)
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)
    
    # 4. Risk-Return scatter (if we have regime data)
    if 'regime' in portfolio_df.columns and len(daily_returns) > 10:
        # Group by regime and calculate risk/return metrics
        regime_stats = []
        for regime in portfolio_df['regime'].unique():
            if pd.notna(regime):
                regime_mask = portfolio_df['regime'] == regime
                regime_returns = daily_returns[regime_mask[1:]]  # Align with returns
                
                if len(regime_returns) > 5:  # Need minimum data points
                    avg_return = regime_returns.mean()
                    volatility = regime_returns.std()
                    regime_stats.append((regime, avg_return, volatility))
        
        if regime_stats:
            regimes, returns, vols = zip(*regime_stats)
            colors = ['#3498DB', '#E74C3C', '#F39C12', '#27AE60', '#9B59B6']
            
            for i, (regime, ret, vol) in enumerate(regime_stats):
                ax4.scatter(vol, ret, s=100, alpha=0.7, 
                           color=colors[i % len(colors)], label=regime)
            
            ax4.set_title('⚖️ Risk-Return by Regime', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Volatility (Daily Return Std)', fontsize=12)
            ax4.set_ylabel('Average Daily Return (%)', fontsize=12)
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adaptive_performance_metrics.png', dpi=300, bbox_inches='tight')
    print("📊 Saved: adaptive_performance_metrics.png")
    return fig

def main():
    """Generate all adaptive bot visualizations."""
    print("🎨 ADAPTIVE BOT VISUALIZATION SUITE")
    print("="*50)
    print("Creating comprehensive charts and graphs...")
    print("")
    
    # Load data
    portfolio_df, trades_df = load_adaptive_data()
    
    if portfolio_df is None:
        return
    
    # Create all visualizations
    print("\n📊 Creating visualizations...")
    
    try:
        # 1. Performance charts
        print("1. Portfolio Performance vs Buy & Hold...")
        create_performance_chart(portfolio_df)
        
        # 2. Regime switching
        print("2. Market Regime Analysis...")
        create_regime_chart(portfolio_df)
        
        # 3. Drawdown analysis
        print("3. Drawdown Protection Analysis...")
        create_drawdown_chart(portfolio_df)
        
        # 4. Trading analysis
        print("4. Trading Activity Analysis...")
        create_trading_analysis_chart(trades_df)
        
        # 5. Performance metrics
        print("5. Advanced Performance Metrics...")
        create_performance_metrics_chart(portfolio_df, trades_df)
        
        print("\n✅ All visualizations created successfully!")
        print("\n📂 Generated files:")
        print("   • adaptive_performance_chart.png - Portfolio vs Buy & Hold")
        print("   • adaptive_regime_chart.png - Regime switching & position sizing")
        print("   • adaptive_drawdown_chart.png - Risk management analysis")
        print("   • adaptive_trading_analysis.png - Trading patterns & signals")
        print("   • adaptive_performance_metrics.png - Advanced performance stats")
        
        # Summary statistics
        print(f"\n📊 ADAPTIVE BOT SUMMARY STATISTICS:")
        initial_value = portfolio_df['portfolio_value'].iloc[0]
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value / initial_value - 1) * 100
        
        print(f"   💰 Total Return: {total_return:+.2f}%")
        print(f"   🎯 Number of Trades: {len(trades_df)}")
        print(f"   📅 Test Period: {(portfolio_df['timestamp'].max() - portfolio_df['timestamp'].min()).days} days")
        
        if 'regime' in portfolio_df.columns:
            regime_changes = (portfolio_df['regime'] != portfolio_df['regime'].shift()).sum()
            print(f"   🔄 Regime Changes: {regime_changes}")
        
        # Calculate Sharpe ratio
        daily_returns = portfolio_df['portfolio_value'].pct_change().dropna()
        if len(daily_returns) > 1:
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(24*365)
            print(f"   📈 Sharpe Ratio: {sharpe:.2f}")
        
        print(f"\n🎨 Open the PNG files to view detailed visualizations!")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 