"""
Configuration template - Copy to config.py and fill in your keys
NEVER commit config.py to Git!
"""

# Alpaca API (Paper Trading)
API_KEY = "YOUR_API_KEY_HERE"
API_SECRET = "YOUR_API_SECRET_HERE"
BASE_URL = "https://paper-api.alpaca.markets"
DATA_FEED = "iex"

# Account
INITIAL_EQUITY = 25000.0

# Risk Parameters
MAX_DAILY_LOSS = 0.008  # 0.8%
MAX_POSITIONS = 3
MAX_TRADES_PER_DAY = 5