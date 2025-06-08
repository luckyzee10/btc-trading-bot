#!/bin/bash
# Cloud Deployment Script for Bitcoin Trading Bot
# This script sets up the bot on a fresh cloud instance

set -e  # Exit on any error

echo "🚀 Starting Bitcoin Trading Bot Cloud Deployment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv git

# Create virtual environment
echo "🐍 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Make scripts executable
chmod +x btc_trading_bot.py
chmod +x run_bot_hourly.py

# Create systemd service for continuous running (optional)
echo "⚙️  Setting up systemd service..."
sudo tee /etc/systemd/system/btc-trading-bot.service > /dev/null <<EOF
[Unit]
Description=Bitcoin Trading Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PWD
Environment=PATH=$PWD/venv/bin
ExecStart=$PWD/venv/bin/python $PWD/run_bot_hourly.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable btc-trading-bot
sudo systemctl start btc-trading-bot

echo "✅ Deployment completed!"
echo "📊 Bot is now running continuously in the background"
echo ""
echo "Useful commands:"
echo "  Check status: sudo systemctl status btc-trading-bot"
echo "  View logs:    sudo journalctl -u btc-trading-bot -f"
echo "  Stop bot:     sudo systemctl stop btc-trading-bot"
echo "  Start bot:    sudo systemctl start btc-trading-bot"
echo ""
echo "📈 Trades will be logged to: $PWD/trades.csv" 