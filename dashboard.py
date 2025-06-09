from flask import Flask, render_template, Response, send_file
import pandas as pd
import sqlite3
import io
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates

app = Flask(__name__)

TRADES_FILE = 'trades.csv'
PORTFOLIO_LOG_FILE = 'portfolio_log.csv'

def get_stats():
    """Calculates performance statistics from the trades log."""
    stats = {
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'win_rate': 0,
        'total_pnl': 0,
        'last_trade_time': 'N/A'
    }
    
    if not os.path.exists(TRADES_FILE):
        return stats, pd.DataFrame()

    try:
        df = pd.read_csv(TRADES_FILE)
        if df.empty:
            return stats, df

        # PNL is only recorded for SELL trades that close a position
        sell_trades = df[df['type'] == 'SELL'].copy()
        sell_trades['pnl'] = pd.to_numeric(sell_trades['pnl'], errors='coerce').fillna(0)
        
        closings = sell_trades[sell_trades['pnl'] != 0]

        stats['total_trades'] = len(df)
        stats['winning_trades'] = len(closings[closings['pnl'] > 0])
        stats['losing_trades'] = len(closings[closings['pnl'] < 0])
        
        if (stats['winning_trades'] + stats['losing_trades']) > 0:
            stats['win_rate'] = (stats['winning_trades'] / (stats['winning_trades'] + stats['losing_trades'])) * 100
        
        stats['total_pnl'] = closings['pnl'].sum()
        stats['last_trade_time'] = pd.to_datetime(df['timestamp']).max().strftime('%Y-%m-%d %H:%M:%S')

        return stats, df.tail(20).sort_index(ascending=False)
        
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return stats, pd.DataFrame()

@app.route('/')
def dashboard():
    """Renders the main dashboard page."""
    stats, recent_trades = get_stats()
    return render_template('dashboard.html', stats=stats, trades=recent_trades.to_dict(orient='records'))

@app.route('/plot/portfolio.png')
def plot_portfolio():
    """Generates and serves the portfolio value chart."""
    try:
        df = pd.read_csv(PORTFOLIO_LOG_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = Figure(figsize=(10, 5), tight_layout=True)
        ax = fig.add_subplot(111)

        ax.plot(df['timestamp'], df['total_value_usdt'], marker='o', linestyle='-', color='#007bff')
        
        ax.set_title('Portfolio Value Over Time (USDT)', fontsize=16)
        ax.set_ylabel('Total Value ($)', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='x', labelrotation=45)
        
        # Format date on x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M'))
        
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#ffffff')

        # Send plot to browser
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')

    except (FileNotFoundError, pd.errors.EmptyDataError, IndexError):
        # Return a placeholder image if data is not available
        return send_file('static/no_data.png', mimetype='image/png')

if __name__ == '__main__':
    # Use PORT provided by environment or default to 8080
    port = int(os.environ.get('PORT', 8080))
    # For local testing, you might want debug=True
    # For production on Railway, debug=False is essential
    app.run(host='0.0.0.0', port=port, debug=False) 