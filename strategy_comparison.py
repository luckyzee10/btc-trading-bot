#!/usr/bin/env python3
"""
COMPREHENSIVE STRATEGY COMPARISON: 1-YEAR ANALYSIS
=================================================
Compare Market-Adaptive Bot vs Aggressive Growth Bot
Over the same 1-year period to determine optimal strategy.

Key Comparison Metrics:
- Total Returns & Alpha Generation
- Risk Management Efficiency  
- Drawdown Analysis
- Trade Frequency & Strategy Adaptation
- Market Condition Performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_and_analyze_results():
    """Load both strategy results and perform comprehensive comparison."""
    
    print("📊 COMPREHENSIVE STRATEGY COMPARISON: 1-YEAR ANALYSIS")
    print("="*70)
    print("Comparing: Market-Adaptive Bot vs Aggressive Growth Bot")
    print("Period: Same 1-year period (Aug 22, 2024 - Jun 9, 2025)")
    print("")
    
    try:
        # Load adaptive bot results
        adaptive_portfolio = pd.read_csv('adaptive_1_year_portfolio.csv')
        adaptive_trades = pd.read_csv('adaptive_1_year_trades.csv')
        
        # Load aggressive bot results  
        aggressive_portfolio = pd.read_csv('aggressive_1_year_portfolio.csv')
        aggressive_trades = pd.read_csv('aggressive_1_year_trades.csv')
        
        print("✅ Successfully loaded both strategy results")
        print(f"   Adaptive: {len(adaptive_portfolio)} portfolio records, {len(adaptive_trades)} trades")
        print(f"   Aggressive: {len(aggressive_portfolio)} portfolio records, {len(aggressive_trades)} trades")
        
    except FileNotFoundError as e:
        print(f"❌ Error loading files: {e}")
        return None
    
    # Performance Summary
    print("\n" + "="*70)
    print("📈 PERFORMANCE COMPARISON SUMMARY")
    print("="*70)
    
    # Adaptive Strategy Results
    adaptive_initial = adaptive_portfolio['portfolio_value'].iloc[0]
    adaptive_final = adaptive_portfolio['portfolio_value'].iloc[-1]
    adaptive_return = ((adaptive_final - adaptive_initial) / adaptive_initial) * 100
    
    # Aggressive Strategy Results
    aggressive_initial = aggressive_portfolio['portfolio_value'].iloc[0]
    aggressive_final = aggressive_portfolio['portfolio_value'].iloc[-1]
    aggressive_return = ((aggressive_final - aggressive_initial) / aggressive_initial) * 100
    
    # Buy & Hold calculation (using BTC price change)
    btc_start = adaptive_portfolio['btc_price'].iloc[0]
    btc_end = adaptive_portfolio['btc_price'].iloc[-1]
    buy_hold_return = ((btc_end - btc_start) / btc_start) * 100
    
    print(f"💰 RETURNS ANALYSIS:")
    print(f"   🔄 Market-Adaptive Strategy:")
    print(f"      Total Return: {adaptive_return:+.2f}%")
    print(f"      Alpha vs Buy & Hold: {adaptive_return - buy_hold_return:+.2f}%")
    print(f"")
    print(f"   🔥 Aggressive Growth Strategy:")
    print(f"      Total Return: {aggressive_return:+.2f}%")
    print(f"      Alpha vs Buy & Hold: {aggressive_return - buy_hold_return:+.2f}%")
    print(f"")
    print(f"   📊 Buy & Hold Baseline: {buy_hold_return:+.2f}%")
    print(f"")
    print(f"   🏆 WINNER: {'Adaptive' if adaptive_return > aggressive_return else 'Aggressive'}")
    print(f"      Performance Gap: {abs(adaptive_return - aggressive_return):.2f}%")
    
    # Risk Analysis
    print(f"\n📊 RISK ANALYSIS:")
    
    # Calculate drawdowns
    adaptive_portfolio['running_max'] = adaptive_portfolio['portfolio_value'].expanding().max()
    adaptive_portfolio['drawdown'] = (adaptive_portfolio['portfolio_value'] - adaptive_portfolio['running_max']) / adaptive_portfolio['running_max']
    adaptive_max_dd = adaptive_portfolio['drawdown'].min() * 100
    
    aggressive_portfolio['running_max'] = aggressive_portfolio['portfolio_value'].expanding().max()
    aggressive_portfolio['drawdown'] = (aggressive_portfolio['portfolio_value'] - aggressive_portfolio['running_max']) / aggressive_portfolio['running_max']
    aggressive_max_dd = aggressive_portfolio['drawdown'].min() * 100
    
    print(f"   🔄 Adaptive Max Drawdown: {adaptive_max_dd:.2f}%")
    print(f"   🔥 Aggressive Max Drawdown: {aggressive_max_dd:.2f}%")
    print(f"   🛡️  Better Risk Control: {'Adaptive' if adaptive_max_dd > aggressive_max_dd else 'Aggressive'}")
    
    # Risk-adjusted returns
    adaptive_risk_efficiency = adaptive_return / abs(adaptive_max_dd) if adaptive_max_dd != 0 else 0
    aggressive_risk_efficiency = aggressive_return / abs(aggressive_max_dd) if aggressive_max_dd != 0 else 0
    
    print(f"   📏 Risk Efficiency (Return/Drawdown):")
    print(f"      Adaptive: {adaptive_risk_efficiency:.2f}")
    print(f"      Aggressive: {aggressive_risk_efficiency:.2f}")
    print(f"      Better Risk-Adjusted: {'Adaptive' if adaptive_risk_efficiency > aggressive_risk_efficiency else 'Aggressive'}")
    
    # Trading Activity Analysis
    print(f"\n🔄 TRADING ACTIVITY ANALYSIS:")
    print(f"   📊 Trade Frequency:")
    print(f"      Adaptive: {len(adaptive_trades)} trades ({len(adaptive_trades)/52:.1f} per week)")
    print(f"      Aggressive: {len(aggressive_trades)} trades ({len(aggressive_trades)/52:.1f} per week)")
    print(f"      More Active: {'Adaptive' if len(adaptive_trades) > len(aggressive_trades) else 'Aggressive'}")
    
    # Strategy Characteristics
    print(f"\n🎯 STRATEGY CHARACTERISTICS:")
    
    # Adaptive strategy analysis
    if 'regime' in adaptive_trades.columns:
        regime_trades = adaptive_trades['regime'].value_counts()
        print(f"   🔄 Adaptive Strategy Regime Distribution:")
        for regime, count in regime_trades.items():
            pct = (count / len(adaptive_trades)) * 100
            print(f"      {regime}: {count} trades ({pct:.1f}%)")
    
    # Win rate analysis
    if 'portfolio_value' in adaptive_trades.columns and len(adaptive_trades) > 1:
        # Calculate trade returns for both strategies
        print(f"\n📈 TRADE PERFORMANCE:")
        # This would require more detailed trade-by-trade analysis
        print(f"   Note: Detailed trade performance analysis would require additional trade metadata")
    
    # Capital Efficiency
    print(f"\n💰 CAPITAL EFFICIENCY:")
    adaptive_capital_growth = (adaptive_final / adaptive_initial)
    aggressive_capital_growth = (aggressive_final / aggressive_initial)
    
    print(f"   Capital Growth Multiple:")
    print(f"      Adaptive: {adaptive_capital_growth:.2f}x")
    print(f"      Aggressive: {aggressive_capital_growth:.2f}x")
    print(f"      Better Capital Growth: {'Adaptive' if adaptive_capital_growth > aggressive_capital_growth else 'Aggressive'}")
    
    # Final Recommendation
    print(f"\n" + "="*70)
    print("🏆 FINAL STRATEGY RECOMMENDATION")
    print("="*70)
    
    # Score each strategy
    adaptive_score = 0
    aggressive_score = 0
    
    # Performance scoring
    if adaptive_return > aggressive_return:
        adaptive_score += 3
        performance_winner = "Adaptive"
    else:
        aggressive_score += 3
        performance_winner = "Aggressive"
    
    # Risk scoring (lower drawdown is better)
    if adaptive_max_dd > aggressive_max_dd:  # Note: more negative = worse
        aggressive_score += 2
        risk_winner = "Aggressive"
    else:
        adaptive_score += 2
        risk_winner = "Adaptive"
    
    # Risk-adjusted scoring
    if adaptive_risk_efficiency > aggressive_risk_efficiency:
        adaptive_score += 2
        risk_adj_winner = "Adaptive"
    else:
        aggressive_score += 2
        risk_adj_winner = "Aggressive"
    
    print(f"📊 SCORING BREAKDOWN:")
    print(f"   Performance Winner: {performance_winner} (+3 points)")
    print(f"   Risk Management Winner: {risk_winner} (+2 points)")
    print(f"   Risk-Adjusted Winner: {risk_adj_winner} (+2 points)")
    print(f"")
    print(f"   🔄 Adaptive Total Score: {adaptive_score}/7")
    print(f"   🔥 Aggressive Total Score: {aggressive_score}/7")
    print(f"")
    
    if adaptive_score > aggressive_score:
        print(f"🏆 RECOMMENDED STRATEGY: MARKET-ADAPTIVE BOT")
        print(f"   ✅ Superior risk-adjusted returns")
        print(f"   ✅ Intelligent regime adaptation")
        print(f"   ✅ Better downside protection")
        print(f"   ✅ Suitable for sustained growth")
    elif aggressive_score > adaptive_score:
        print(f"🏆 RECOMMENDED STRATEGY: AGGRESSIVE GROWTH BOT")
        print(f"   ✅ Higher absolute returns")
        print(f"   ✅ Maximum capital growth")
        print(f"   ✅ Suitable for risk-tolerant investors")
        print(f"   ✅ Faster wealth building potential")
    else:
        print(f"🤝 TIE: Both strategies have merit")
        print(f"   📊 Choice depends on risk tolerance and timeline")
    
    # Market Condition Insights
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   • Test period included diverse market conditions")
    print(f"   • BTC gained {buy_hold_return:+.1f}% during this period")
    print(f"   • Adaptive strategy made regime adjustments for market changes")
    print(f"   • Aggressive strategy maintained consistent high-risk approach")
    print(f"   • Both strategies {'outperformed' if min(adaptive_return, aggressive_return) > buy_hold_return else 'underperformed'} buy & hold")
    
    # Future Considerations
    print(f"\n🔮 STRATEGY SELECTION GUIDE:")
    print(f"   💰 Choose ADAPTIVE if:")
    print(f"      • You want consistent risk-adjusted returns")
    print(f"      • You prefer intelligent strategy adaptation")
    print(f"      • You want better downside protection")
    print(f"      • You're building long-term wealth steadily")
    print(f"")
    print(f"   🚀 Choose AGGRESSIVE if:")
    print(f"      • You can tolerate higher volatility")
    print(f"      • You want maximum growth potential")
    print(f"      • You have small account to grow quickly")
    print(f"      • You can handle larger drawdowns")
    
    return {
        'adaptive_return': adaptive_return,
        'aggressive_return': aggressive_return,
        'buy_hold_return': buy_hold_return,
        'adaptive_max_dd': adaptive_max_dd,
        'aggressive_max_dd': aggressive_max_dd,
        'adaptive_score': adaptive_score,
        'aggressive_score': aggressive_score,
        'winner': 'Adaptive' if adaptive_score > aggressive_score else 'Aggressive' if aggressive_score > adaptive_score else 'Tie'
    }

def main():
    """Run comprehensive strategy comparison."""
    results = load_and_analyze_results()
    
    if results:
        print(f"\n📊 Comparison complete! Results summary:")
        print(f"   🔄 Adaptive: {results['adaptive_return']:+.1f}% return, {results['adaptive_max_dd']:.1f}% max DD")
        print(f"   🔥 Aggressive: {results['aggressive_return']:+.1f}% return, {results['aggressive_max_dd']:.1f}% max DD")
        print(f"   🏆 Winner: {results['winner']}")
    else:
        print("❌ Comparison failed - check that both strategy results are available")

if __name__ == "__main__":
    main() 