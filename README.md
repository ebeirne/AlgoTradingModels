# Algorithmic Trading Model - Phase 1

## Overview
Risk-focused trading system using Poisson event modeling for statistical signal generation.

## Features
- ✅ Event-aware trading (avoids earnings/Fed)
- ✅ Adaptive stop loss optimization
- ✅ Transaction cost analysis
- ✅ Market regime filtering
- ✅ Kelly criterion position sizing

## Setup

### Requirements
- Python 3.9+
- Alpaca paper trading account

### Installation
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### Configuration
1. Copy `config_example.py` to `config.py`
2. Add your Alpaca API keys to `config.py`
3. Never commit `config.py` to Git!

### Running
\`\`\`bash
python phase1_improved.py
\`\`\`

## Results
- **Phase 1 Original:** -$217 in 22 minutes (0% win rate)
- **Phase 1 Improved:** Testing in progress...

## Safety
- Paper trading only
- Max 5 trades per day
- 0.8% daily stop loss
- Only trades 10:30 AM - 2:30 PM EST
- Requires SPY to be green

## Files
- `phase1_improved.py` - Production model with all fixes
- `phase1_original.py` - Original version (for comparison)
- `docs/analysis.md` - Performance analysis
- `docs/improvements.md` - Detailed fix explanations

## Disclaimer
This is for educational purposes only. Trading involves risk of loss.