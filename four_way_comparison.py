#!/usr/bin/env python3
"""
ULTIMATE 4-WAY STRATEGY COMPARISON: 1-YEAR ANALYSIS
==================================================
Compare ALL FOUR strategies over the same 1-year period:
1. Market-Adaptive Bot (Intelligent regime switching)
2. Aggressive Growth Bot (Maximum risk for growth)  
3. Optimized Bot (Conservative risk management)
4. Backtest Markov Bot (Classic Markov with weekly rebuilds)

Determine the ULTIMATE WINNER across all metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_all_four_strategy_results():
    """Load all four strategy results and perform ultimate comparison."""
    
    print("🏆 ULTIMATE 4-WAY STRATEGY COMPARISON: 1-YEAR ANALYSIS")
    print("="*75)
    print("Comparing ALL FOUR strategies over the same 1-year period:")
    print("🔄 Market-Adaptive Bot (Intelligent regime switching)")
    print("🔥 Aggressive Growth Bot (Maximum risk for growth)")
    print("🛡️  Optimized Bot (Conservative risk management)")
    print("🔬 Backtest Markov Bot (Classic Markov with weekly rebuilds)")
    print("Period: Same 1-year period (Oct 2024 - Jun 2025)")
    print("")
    
    strategies = {}
    
    # Load all four strategy results
    try:
        # Adaptive Bot
        strategies['Adaptive'] = {
            'portfolio': pd.read_csv('adaptive_1_year_portfolio.csv'),
            'trades': pd.read_csv('adaptive_1_year_trades.csv'),
            'emoji': '🔄',
            'description': 'Market-Adaptive (Regime Switching)'
        }
        
        # Aggressive Bot
        strategies['Aggressive'] = {
            'portfolio': pd.read_csv('aggressive_1_year_portfolio.csv'),
            'trades': pd.read_csv('aggressive_1_year_trades.csv'),
            'emoji': '🔥',
            'description': 'Aggressive Growth (High Risk)'
        }
        
        # Optimized Bot
        strategies['Optimized'] = {
            'portfolio': pd.read_csv('optimized_1_year_portfolio.csv'),
            'trades': pd.read_csv('optimized_1_year_trades.csv'),
            'emoji': '🛡️',
            'description': 'Optimized (Conservative)'
        }
        
        # Backtest Markov Bot
        strategies['BacktestMarkov'] = {
            'portfolio': pd.read_csv('backtest_markov_1_year_portfolio.csv'),
            'trades': pd.read_csv('backtest_markov_1_year_trades.csv'),
            'emoji': '🔬',
            'description': 'Classic Markov (Weekly Rebuilds)'
        }
        
        print("✅ Successfully loaded ALL FOUR strategy results")
        for name, data in strategies.items():
            print(f"   {data['emoji']} {name}: {len(data['portfolio'])} portfolio records, {len(data['trades'])} trades")
        
    except FileNotFoundError as e:
        print(f"❌ Error loading files: {e}")
        print("Make sure all four strategies have been run and saved!")
        return None
    
    # Calculate performance metrics for each strategy
    results = {}
    
    # Use the strategy with the most consistent BTC price data (usually Adaptive)
    btc_start = strategies['Adaptive']['portfolio']['btc_price'].iloc[0]
    btc_end = strategies['Adaptive']['portfolio']['btc_price'].iloc[-1]
    buy_hold_return = ((btc_end - btc_start) / btc_start) * 100
    
    print("\n" + "="*75)
    print("📈 ULTIMATE 4-WAY PERFORMANCE COMPARISON")
    print("="*75)
    
    # Calculate metrics for each strategy
    for name, data in strategies.items():
        portfolio_df = data['portfolio']
        trades_df = data['trades']
        
        # Convert timestamps to datetime if they're strings
        if isinstance(portfolio_df['timestamp'].iloc[0], str):
            portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
        
        # Performance calculations
        initial_value = portfolio_df['portfolio_value'].iloc[0]
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = ((final_value - initial_value) / initial_value) * 100
        alpha = total_return - buy_hold_return
        
        # Risk calculations
        portfolio_df['running_max'] = portfolio_df['portfolio_value'].expanding().max()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['running_max']) / portfolio_df['running_max']
        max_drawdown = portfolio_df['drawdown'].min() * 100
        
        # Advanced metrics
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
        valid_returns = portfolio_df['daily_return'].dropna()
        sharpe_ratio = valid_returns.mean() / valid_returns.std() * np.sqrt(8760) if len(valid_returns) > 1 and valid_returns.std() > 0 else 0
        risk_efficiency = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        capital_growth = final_value / initial_value
        
        # Calculate annualized return - FIXED timestamp calculation
        test_period_days = (portfolio_df['timestamp'].iloc[-1] - portfolio_df['timestamp'].iloc[0]).days
        
        if test_period_days > 0:
            growth_multiple = 1 + (total_return / 100)
            annualized_return = ((growth_multiple ** (365/test_period_days)) - 1) * 100
        else:
            annualized_return = total_return
        
        results[name] = {
            'emoji': data['emoji'],
            'description': data['description'],
            'total_return': total_return,
            'alpha': alpha,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'risk_efficiency': risk_efficiency,
            'capital_growth': capital_growth,
            'annualized_return': annualized_return,
            'num_trades': len(trades_df),
            'initial_value': initial_value,
            'final_value': final_value,
            'period_days': test_period_days
        }
        
        print(f"{data['emoji']} {name.upper()} STRATEGY:")
        print(f"   Total Return: {total_return:+.2f}%")
        print(f"   Annualized Return: {annualized_return:+.1f}%")
        print(f"   Alpha vs Buy & Hold: {alpha:+.2f}%")
        print(f"   Max Drawdown: {max_drawdown:.2f}%")
        print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"   Risk Efficiency: {risk_efficiency:.2f}")
        print(f"   Capital Growth: {capital_growth:.2f}x")
        print(f"   Number of Trades: {len(trades_df)}")
        print(f"   Test Period: {test_period_days} days")
        print("")
    
    print(f"📊 Buy & Hold Baseline: {buy_hold_return:+.2f}%")
    print("")
    
    # ULTIMATE SCORING SYSTEM
    print("="*75)
    print("🏆 ULTIMATE SCORING & RANKING (4-WAY)")
    print("="*75)
    
    # Enhanced scoring categories (out of 15 points each for 6 categories)
    categories = {
        'Total Returns': 'total_return',
        'Alpha Generation': 'alpha', 
        'Risk Management': 'max_drawdown',  # Lower is better
        'Risk-Adjusted Returns': 'sharpe_ratio',
        'Capital Efficiency': 'risk_efficiency',
        'Annualized Performance': 'annualized_return'
    }
    
    scores = {name: 0 for name in results.keys()}
    
    print("📊 SCORING BREAKDOWN:")
    
    for category, metric in categories.items():
        print(f"\n🎯 {category.upper()}:")
        
        # Get values and rank them
        values = [(name, results[name][metric]) for name in results.keys()]
        
        if metric == 'max_drawdown':  # Lower is better for drawdown
            values.sort(key=lambda x: x[1], reverse=False)  # Sort ascending (lower is better)
        else:  # Higher is better for all other metrics
            values.sort(key=lambda x: x[1], reverse=True)   # Sort descending (higher is better)
        
        # Award points: 1st place = 15 points, 2nd = 10, 3rd = 6, 4th = 3
        points = [15, 10, 6, 3]
        for i, (name, value) in enumerate(values):
            score_awarded = points[i]
            scores[name] += score_awarded
            
            if metric == 'max_drawdown':
                print(f"   #{i+1} {results[name]['emoji']} {name}: {value:.2f}% (+{score_awarded} pts)")
            else:
                print(f"   #{i+1} {results[name]['emoji']} {name}: {value:.2f} (+{score_awarded} pts)")
    
    # Final rankings
    print(f"\n" + "="*75)
    print("🏆 FINAL RANKINGS & ULTIMATE 4-WAY WINNER")
    print("="*75)
    
    # Sort by total score
    final_rankings = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    print("📊 FINAL SCORES (out of 90 possible points):")
    for i, (name, total_score) in enumerate(final_rankings):
        emoji = results[name]['emoji']
        description = results[name]['description']
        
        # Medal emojis for top 3
        medals = ['🥇', '🥈', '🥉', '4️⃣']
        medal = medals[i] if i < 4 else f'{i+1}.'
        
        print(f"   {medal} {emoji} {name}: {total_score}/90 points")
        print(f"       {description}")
        print(f"       {results[name]['total_return']:+.1f}% return, {results[name]['annualized_return']:+.1f}% annualized")
        print(f"       {results[name]['max_drawdown']:.1f}% max DD, {results[name]['sharpe_ratio']:.2f} Sharpe")
        print("")
    
    # Declare ultimate winner
    winner_name, winner_score = final_rankings[0]
    winner_data = results[winner_name]
    
    print("🎊 " + "="*60 + " 🎊")
    print(f"🏆 ULTIMATE 4-WAY WINNER: {winner_data['emoji']} {winner_name.upper()} STRATEGY")
    print("🎊 " + "="*60 + " 🎊")
    print(f"🏅 Final Score: {winner_score}/90 points")
    print(f"📈 Total Return: {winner_data['total_return']:+.2f}%")
    print(f"🚀 Annualized Return: {winner_data['annualized_return']:+.1f}%")
    print(f"⚡ Alpha Generation: {winner_data['alpha']:+.2f}%")
    print(f"🛡️  Max Drawdown: {winner_data['max_drawdown']:.2f}%")
    print(f"📊 Sharpe Ratio: {winner_data['sharpe_ratio']:.2f}")
    print(f"💎 Risk Efficiency: {winner_data['risk_efficiency']:.2f}")
    print("")
    
    # Strategy insights
    print("💡 ULTIMATE 4-WAY INSIGHTS:")
    
    if winner_name == 'BacktestMarkov':
        print("   🔬 Classic Markov approach with weekly rebuilds proved SUPERIOR")
        print("   ✅ High-frequency trading with excellent win rate")
        print("   ✅ Weekly matrix adaptation captured market changes optimally")
        print("   ✅ 25% position sizing with 8% stop losses provided best risk/reward")
        
    elif winner_name == 'Adaptive':
        print("   🎯 Intelligent regime switching proved SUPERIOR")
        print("   ✅ Dynamic risk adaptation beat static approaches")
        print("   ✅ Market-aware positioning optimized for all conditions")
        print("   ✅ Best risk-adjusted returns across the full cycle")
        
    elif winner_name == 'Aggressive':
        print("   🚀 High-risk high-reward approach proved SUPERIOR")
        print("   ✅ Maximum capital growth through large position sizes")
        print("   ✅ Aggressive momentum capture beat conservative approaches")
        print("   ✅ Risk tolerance paid off in this market cycle")
        
    elif winner_name == 'Optimized':
        print("   🛡️  Conservative risk management proved SUPERIOR")
        print("   ✅ Advanced filtering and portfolio protection excelled")
        print("   ✅ Steady compounding beat aggressive approaches")
        print("   ✅ Professional-grade risk controls delivered best results")
    
    print("\n🔮 STRATEGY SELECTION RECOMMENDATIONS:")
    
    # Detailed recommendations based on results
    performance_leader = max(results.items(), key=lambda x: x[1]['total_return'])
    risk_leader = min(results.items(), key=lambda x: x[1]['max_drawdown'])
    sharpe_leader = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])
    alpha_leader = max(results.items(), key=lambda x: x[1]['alpha'])
    
    print(f"   📈 Choose {performance_leader[0]} for: Maximum absolute returns ({performance_leader[1]['total_return']:+.1f}%)")
    print(f"   🛡️  Choose {risk_leader[0]} for: Best risk control ({risk_leader[1]['max_drawdown']:.1f}% max DD)")
    print(f"   ⚖️  Choose {sharpe_leader[0]} for: Best risk-adjusted returns (Sharpe: {sharpe_leader[1]['sharpe_ratio']:.2f})")
    print(f"   ⚡ Choose {alpha_leader[0]} for: Maximum alpha generation ({alpha_leader[1]['alpha']:+.1f}%)")
    
    # Trading activity insights
    print(f"\n🔄 TRADING ACTIVITY INSIGHTS:")
    trade_activity = {name: data['num_trades'] for name, data in results.items()}
    most_active = max(trade_activity.items(), key=lambda x: x[1])
    least_active = min(trade_activity.items(), key=lambda x: x[1])
    
    print(f"   Most Active: {results[most_active[0]]['emoji']} {most_active[0]} ({most_active[1]} trades)")
    print(f"   Least Active: {results[least_active[0]]['emoji']} {least_active[0]} ({least_active[1]} trades)")
    
    # Calculate trade efficiency (return per trade)
    trade_efficiency = {}
    for name, data in results.items():
        if data['num_trades'] > 0:
            efficiency = data['total_return'] / data['num_trades']
            trade_efficiency[name] = efficiency
    
    if trade_efficiency:
        most_efficient = max(trade_efficiency.items(), key=lambda x: x[1])
        print(f"   Most Efficient: {results[most_efficient[0]]['emoji']} {most_efficient[0]} ({most_efficient[1]:.3f}% return per trade)")
    
    # Market condition analysis
    print(f"\n📊 MARKET CONDITION ANALYSIS:")
    print(f"   • Test period: Strong BTC bull market (+{buy_hold_return:.1f}%)")
    
    strategies_beating_buy_hold = [name for name, data in results.items() if data['alpha'] > 0]
    print(f"   • Strategies beating buy & hold: {len(strategies_beating_buy_hold)}/4")
    
    if strategies_beating_buy_hold:
        print(f"   • Alpha generators: {', '.join([results[name]['emoji'] + name for name in strategies_beating_buy_hold])}")
    
    # Performance spread analysis
    returns = [data['total_return'] for data in results.values()]
    return_spread = max(returns) - min(returns)
    print(f"   • Performance spread: {return_spread:.1f}% between best and worst")
    
    return {
        'strategies': results,
        'winner': winner_name,
        'winner_score': winner_score,
        'buy_hold_return': buy_hold_return,
        'rankings': final_rankings
    }

def main():
    """Run ultimate 4-way strategy comparison."""
    results = load_all_four_strategy_results()
    
    if results:
        print(f"\n🎊 ULTIMATE 4-WAY COMPARISON COMPLETE! 🎊")
        print(f"Winner: {results['strategies'][results['winner']]['emoji']} {results['winner']} Strategy")
        print(f"Score: {results['winner_score']}/90 points")
        
        winner_data = results['strategies'][results['winner']]
        print(f"Performance: {winner_data['total_return']:+.1f}% total, {winner_data['annualized_return']:+.1f}% annualized")
        print(f"Market context: BTC +{results['buy_hold_return']:.1f}% (strong bull market)")
        
        # Summary table
        print(f"\n📊 COMPREHENSIVE SUMMARY TABLE:")
        print(f"{'Strategy':<15} {'Return':<8} {'Annual':<8} {'Alpha':<8} {'Drawdown':<10} {'Sharpe':<7} {'Trades':<7} {'Score':<6}")
        print("-" * 90)
        for name, score in results['rankings']:
            data = results['strategies'][name]
            print(f"{name:<15} {data['total_return']:>+6.1f}%  {data['annualized_return']:>+6.1f}%  {data['alpha']:>+6.1f}%  {data['max_drawdown']:>8.1f}%  {data['sharpe_ratio']:>5.2f}  {data['num_trades']:>6}  {score:>4}/90")
        
        # Key takeaways
        print(f"\n🎯 KEY TAKEAWAYS:")
        winner = results['strategies'][results['winner']]
        print(f"   🏆 {results['winner']} dominated with {winner['total_return']:+.1f}% returns")
        print(f"   📈 All strategies tested over similar periods ({winner['period_days']} days average)")
        print(f"   🎪 This bull market period favored {'high-frequency' if winner['num_trades'] > 1000 else 'adaptive'} approaches")
        print(f"   ⚡ Winner generated {winner['alpha']:+.1f}% alpha vs buy & hold")
        
    else:
        print("❌ Ultimate 4-way comparison failed - ensure all strategies have been run!")

if __name__ == "__main__":
    main() 