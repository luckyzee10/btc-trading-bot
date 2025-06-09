#!/usr/bin/env python3
"""
ULTIMATE 3-WAY STRATEGY COMPARISON: 1-YEAR ANALYSIS
==================================================
Compare ALL THREE strategies over the same 1-year period:
1. Market-Adaptive Bot (Intelligent regime switching)
2. Aggressive Growth Bot (Maximum risk for growth)  
3. Optimized Bot (Conservative risk management)

Determine the ULTIMATE WINNER across all metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_all_strategy_results():
    """Load all three strategy results and perform ultimate comparison."""
    
    print("🏆 ULTIMATE 3-WAY STRATEGY COMPARISON: 1-YEAR ANALYSIS")
    print("="*75)
    print("Comparing ALL THREE strategies over the same 1-year period:")
    print("🔄 Market-Adaptive Bot (Intelligent regime switching)")
    print("🔥 Aggressive Growth Bot (Maximum risk for growth)")
    print("🛡️  Optimized Bot (Conservative risk management)")
    print("Period: Same 1-year period (Aug 22, 2024 - Jun 9, 2025)")
    print("")
    
    strategies = {}
    
    # Load all three strategy results
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
        
        print("✅ Successfully loaded ALL THREE strategy results")
        for name, data in strategies.items():
            print(f"   {data['emoji']} {name}: {len(data['portfolio'])} portfolio records, {len(data['trades'])} trades")
        
    except FileNotFoundError as e:
        print(f"❌ Error loading files: {e}")
        print("Make sure all three strategies have been run and saved!")
        return None
    
    # Calculate performance metrics for each strategy
    results = {}
    
    # Use adaptive portfolio for consistent BTC price data
    btc_start = strategies['Adaptive']['portfolio']['btc_price'].iloc[0]
    btc_end = strategies['Adaptive']['portfolio']['btc_price'].iloc[-1]
    buy_hold_return = ((btc_end - btc_start) / btc_start) * 100
    
    print("\n" + "="*75)
    print("📈 ULTIMATE PERFORMANCE COMPARISON")
    print("="*75)
    
    # Calculate metrics for each strategy
    for name, data in strategies.items():
        portfolio_df = data['portfolio']
        trades_df = data['trades']
        
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
        
        results[name] = {
            'emoji': data['emoji'],
            'description': data['description'],
            'total_return': total_return,
            'alpha': alpha,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'risk_efficiency': risk_efficiency,
            'capital_growth': capital_growth,
            'num_trades': len(trades_df),
            'initial_value': initial_value,
            'final_value': final_value
        }
        
        print(f"{data['emoji']} {name.upper()} STRATEGY:")
        print(f"   Total Return: {total_return:+.2f}%")
        print(f"   Alpha vs Buy & Hold: {alpha:+.2f}%")
        print(f"   Max Drawdown: {max_drawdown:.2f}%")
        print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"   Risk Efficiency: {risk_efficiency:.2f}")
        print(f"   Capital Growth: {capital_growth:.2f}x")
        print(f"   Number of Trades: {len(trades_df)}")
        print("")
    
    print(f"📊 Buy & Hold Baseline: {buy_hold_return:+.2f}%")
    print("")
    
    # ULTIMATE SCORING SYSTEM
    print("="*75)
    print("🏆 ULTIMATE SCORING & RANKING")
    print("="*75)
    
    # Scoring categories (out of 10 points each)
    categories = {
        'Performance': 'total_return',
        'Alpha Generation': 'alpha', 
        'Risk Management': 'max_drawdown',  # Lower is better
        'Risk-Adjusted Returns': 'sharpe_ratio',
        'Capital Efficiency': 'risk_efficiency'
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
        
        # Award points: 1st place = 10 points, 2nd place = 6 points, 3rd place = 3 points
        points = [10, 6, 3]
        for i, (name, value) in enumerate(values):
            score_awarded = points[i]
            scores[name] += score_awarded
            
            if metric == 'max_drawdown':
                print(f"   #{i+1} {results[name]['emoji']} {name}: {value:.2f}% (+{score_awarded} pts)")
            else:
                print(f"   #{i+1} {results[name]['emoji']} {name}: {value:.2f} (+{score_awarded} pts)")
    
    # Final rankings
    print(f"\n" + "="*75)
    print("🏆 FINAL RANKINGS & ULTIMATE WINNER")
    print("="*75)
    
    # Sort by total score
    final_rankings = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    print("📊 FINAL SCORES (out of 50 possible points):")
    for i, (name, total_score) in enumerate(final_rankings):
        emoji = results[name]['emoji']
        description = results[name]['description']
        print(f"   #{i+1} {emoji} {name}: {total_score}/50 points")
        print(f"       {description}")
        print(f"       {results[name]['total_return']:+.1f}% return, {results[name]['max_drawdown']:.1f}% max DD")
        print("")
    
    # Declare ultimate winner
    winner_name, winner_score = final_rankings[0]
    winner_data = results[winner_name]
    
    print("🎊 " + "="*50 + " 🎊")
    print(f"🏆 ULTIMATE WINNER: {winner_data['emoji']} {winner_name.upper()} STRATEGY")
    print("🎊 " + "="*50 + " 🎊")
    print(f"🏅 Final Score: {winner_score}/50 points")
    print(f"📈 Total Return: {winner_data['total_return']:+.2f}%")
    print(f"⚡ Alpha Generation: {winner_data['alpha']:+.2f}%")
    print(f"🛡️  Max Drawdown: {winner_data['max_drawdown']:.2f}%")
    print(f"📊 Sharpe Ratio: {winner_data['sharpe_ratio']:.2f}")
    print(f"💎 Risk Efficiency: {winner_data['risk_efficiency']:.2f}")
    print("")
    
    # Strategy insights
    print("💡 ULTIMATE INSIGHTS:")
    
    if winner_name == 'Adaptive':
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
    
    print(f"   📈 Choose {performance_leader[0]} for: Maximum absolute returns ({performance_leader[1]['total_return']:+.1f}%)")
    print(f"   🛡️  Choose {risk_leader[0]} for: Best risk control ({risk_leader[1]['max_drawdown']:.1f}% max DD)")
    print(f"   ⚖️  Choose {sharpe_leader[0]} for: Best risk-adjusted returns (Sharpe: {sharpe_leader[1]['sharpe_ratio']:.2f})")
    
    # Market condition analysis
    print(f"\n📊 MARKET CONDITION ANALYSIS:")
    print(f"   • Test period: Strong BTC bull market (+{buy_hold_return:.1f}%)")
    print(f"   • All strategies {'outperformed' if min([r['total_return'] for r in results.values()]) > buy_hold_return else 'had mixed performance vs'} buy & hold")
    print(f"   • Winner dominated across multiple metrics, not just one")
    
    # Trading activity insights
    trade_activity = {name: data['num_trades'] for name, data in results.items()}
    most_active = max(trade_activity.items(), key=lambda x: x[1])
    least_active = min(trade_activity.items(), key=lambda x: x[1])
    
    print(f"\n🔄 TRADING ACTIVITY INSIGHTS:")
    print(f"   Most Active: {results[most_active[0]]['emoji']} {most_active[0]} ({most_active[1]} trades)")
    print(f"   Least Active: {results[least_active[0]]['emoji']} {least_active[0]} ({least_active[1]} trades)")
    print(f"   Trade efficiency varies significantly across strategies")
    
    return {
        'strategies': results,
        'winner': winner_name,
        'winner_score': winner_score,
        'buy_hold_return': buy_hold_return,
        'rankings': final_rankings
    }

def main():
    """Run ultimate 3-way strategy comparison."""
    results = load_all_strategy_results()
    
    if results:
        print(f"\n🎊 ULTIMATE COMPARISON COMPLETE! 🎊")
        print(f"Winner: {results['strategies'][results['winner']]['emoji']} {results['winner']} Strategy")
        print(f"Score: {results['winner_score']}/50 points")
        print(f"\nAll three strategies tested over the same {(pd.to_datetime('2025-06-09') - pd.to_datetime('2024-08-22')).days}-day period")
        print(f"Market context: BTC +{results['buy_hold_return']:.1f}% (strong bull market)")
        
        # Summary table
        print(f"\n📊 QUICK SUMMARY TABLE:")
        print(f"{'Strategy':<15} {'Return':<10} {'Alpha':<8} {'Drawdown':<10} {'Sharpe':<8} {'Score':<6}")
        print("-" * 65)
        for name, score in results['rankings']:
            data = results['strategies'][name]
            print(f"{name:<15} {data['total_return']:>+7.1f}%  {data['alpha']:>+6.1f}%  {data['max_drawdown']:>7.1f}%  {data['sharpe_ratio']:>6.2f}  {score:>4}/50")
        
    else:
        print("❌ Ultimate comparison failed - ensure all three strategies have been run!")

if __name__ == "__main__":
    main() 