#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUICK WINS MODEL

RECENT PATCH UPDATES (ver. 1.1):

PHASE 1 IMPROVED: Critical Fixes Applied
- Fixed position sizing for available capital
- Market regime filter (no trading on red days)
- Signal strength threshold
- Symbol cooldown (prevent whipsaws)
- Optimal trading hours only
- Wider stops for small accounts
- Spread filtering by time of day
"""

from __future__ import annotations
import os, math, csv, sys, time, pathlib, collections
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta, time as dt_time

import numpy as np
import pandas as pd
import pytz
from scipy import stats
import alpaca_trade_api as tradeapi

# ========== CONFIGURATION ==========
API_KEY = "your-key-here"
API_SECRET = "your-secret-here"
BASE_URL = "https://paper-api.alpaca.markets"
DATA_FEED = "iex"

@dataclass
class ImprovedConfig:
    initial_equity: float = 25000.0  # Actual paper account size
    
    # Risk limits (ADJUSTED FOR SMALL ACCOUNT)
    max_portfolio_risk: float = 0.012  # 1.2% (was 1.5%)
    max_position_risk: float = 0.0025  # 0.25% (was 0.3%)
    max_daily_loss: float = 0.008      # 0.8% hard stop
    max_positions: int = 3             # Reduced from 8 for small account
    max_correlated_positions: int = 2  # Reduced from 3
    max_trades_per_day: int = 5        # NEW: Hard limit
    
    # Position sizing (CRITICAL FIX #1)
    capital_usage_per_position: float = 0.15  # Max 15% capital per trade
    capital_buffer: float = 0.90       # Use only 90% of available
    
    # Signal quality (CRITICAL FIX #3)
    min_signal_strength: float = 3.5   # Require strong signals only
    require_confirmation_bars: int = 2  # Need 2+ bars confirming
    
    # Timing (CRITICAL FIX #10)
    optimal_start_hour: int = 10
    optimal_start_minute: int = 30
    optimal_end_hour: int = 14
    optimal_end_minute: int = 30
    avoid_first_minutes: int = 60      # Don't trade first hour
    avoid_last_minutes: int = 90       # Don't trade last 90 min
    
    # Market regime (CRITICAL FIX #2)
    require_positive_market: bool = True  # Only trade on green days
    min_spy_return: float = 0.002      # SPY must be up 20+ bps
    max_spy_range: float = 0.025       # SPY range < 2.5% (not too choppy)
    
    # Symbol management (CRITICAL FIX #4)
    symbol_cooldown_seconds: int = 600  # 10 min between trades per symbol
    min_hold_seconds: int = 300        # 5 min minimum hold
    
    # Stops (CRITICAL FIX #7)
    base_stop_mult_small_account: float = 2.5  # Wider for small accounts
    
    # Execution (CRITICAL FIX #8)
    max_spread_bps_open: float = 8.0   # Tight at market open
    max_spread_bps_normal: float = 12.0
    max_spread_bps_close: float = 10.0
    
    # Position sizing
    kelly_fraction: float = 0.20       # Reduced from 0.25 (more conservative)
    min_edge: float = 0.18             # Higher bar (was 0.15)
    min_win_rate: float = 0.54         # Higher bar (was 0.52)
    
    # Poisson parameters
    lookback_events: int = 100
    min_events: int = 25               # Higher threshold
    lambda_scale: float = 1.5
    
    min_liquidity: float = 500_000
    slippage_bps: float = 4.0          # More conservative
    
    # Event awareness
    event_lookback_hours: int = 4
    event_exit_hours: int = 1
    avoid_earnings: bool = True
    avoid_fed_days: bool = True
    
    # Adaptive stops
    stop_optimization_lookback: int = 90
    stop_range_min: float = 1.2        # Tighter minimum
    stop_range_max: float = 3.5
    vol_adjust_stops: bool = True
    
    # TCA
    track_execution_quality: bool = True
    min_tca_samples: int = 10
    max_acceptable_slippage_bps: float = 7.0  # Tighter
    
    symbols: Tuple[str, ...] = (
        "SPY", "QQQ", "IWM",
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
        "JPM", "BAC", "GS",
        "XLE", "XOM"
    )

CFG = ImprovedConfig()
NY = pytz.timezone("America/New_York")
rest = tradeapi.REST(API_KEY, API_SECRET, base_url=BASE_URL, api_version='v2')

# ========== UTILITIES ==========
def now_ny() -> datetime:
    return datetime.now(NY)

def minutes_since_open() -> int:
    dt = now_ny()
    open_time = dt.replace(hour=9, minute=30, second=0, microsecond=0)
    return max(0, int((dt - open_time).total_seconds() / 60))

def is_market_open() -> bool:
    try:
        return bool(rest.get_clock().is_open)
    except:
        dt = now_ny()
        if dt.weekday() >= 5:
            return False
        t = dt.time()
        return datetime.strptime("09:30", "%H:%M").time() <= t <= datetime.strptime("16:00", "%H:%M").time()

def is_in_optimal_trading_hours() -> Tuple[bool, str]:
    """NEW: Check if we're in optimal trading window"""
    now = now_ny()
    current_time = now.time()
    
    # Optimal window: 10:30 AM - 2:30 PM
    start = dt_time(CFG.optimal_start_hour, CFG.optimal_start_minute)
    end = dt_time(CFG.optimal_end_hour, CFG.optimal_end_minute)
    
    if current_time < start:
        return False, f"before_optimal_window_{start}"
    if current_time > end:
        return False, f"after_optimal_window_{end}"
    
    # Also check minutes since open
    mins = minutes_since_open()
    if mins < CFG.avoid_first_minutes:
        return False, f"too_early_{mins}min"
    if mins > (390 - CFG.avoid_last_minutes):
        return False, f"too_late_{mins}min"
    
    return True, "optimal_hours"

def get_max_spread_bps() -> float:
    """NEW: Dynamic spread threshold by time of day"""
    mins = minutes_since_open()
    
    if mins < 60:
        return CFG.max_spread_bps_open  # Tight during open volatility
    elif mins > 330:
        return CFG.max_spread_bps_close  # Tight near close
    else:
        return CFG.max_spread_bps_normal

# ========== CRITICAL FIX #2: MARKET REGIME FILTER ==========
class MarketRegimeFilter:
    """
    NEW: Don't trade on red/choppy days
    Single biggest improvement from analysis
    """
    
    def __init__(self):
        self.last_regime_check = 0
        self.cached_regime = None
        print("[IMPROVED] MarketRegimeFilter initialized")
    
    def get_market_regime(self, spy_bars: pd.DataFrame) -> Dict:
        """Check if market conditions allow trading"""
        
        # Cache for 60 seconds
        if time.time() - self.last_regime_check < 60 and self.cached_regime:
            return self.cached_regime
        
        if spy_bars.empty or len(spy_bars) < 30:
            return {'regime': 'unknown', 'trade': False, 'reason': 'insufficient_data'}
        
        # Get today's data only
        try:
            today_date = now_ny().date()
            spy_bars.index = pd.to_datetime(spy_bars.index).tz_localize(None)
            today = spy_bars[spy_bars.index.date == today_date]
        except:
            today = spy_bars.tail(390)  # Fallback: last day
        
        if len(today) < 30:
            return {'regime': 'unknown', 'trade': False, 'reason': 'insufficient_today_data'}
        
        # Calculate metrics
        spy_open = float(today['o'].iloc[0])
        spy_now = float(today['c'].iloc[-1])
        spy_high = float(today['h'].max())
        spy_low = float(today['l'].min())
        
        spy_return = (spy_now - spy_open) / spy_open
        spy_range = (spy_high - spy_low) / spy_open
        
        regime = {
            'spy_return': spy_return,
            'spy_range': spy_range,
            'spy_price': spy_now,
            'regime': None,
            'trade': False,
            'reason': ''
        }
        
        # GREEN DAY RULES
        if spy_return >= CFG.min_spy_return:
            if spy_range <= CFG.max_spy_range:
                regime['regime'] = 'uptrend_orderly'
                regime['trade'] = True
                regime['reason'] = f"green_orderly_ret={spy_return:.3f}_range={spy_range:.3f}"
            else:
                regime['regime'] = 'uptrend_choppy'
                regime['trade'] = False
                regime['reason'] = f"too_choppy_range={spy_range:.3f}"
        
        # RED DAY RULES
        elif spy_return < -0.002:
            regime['regime'] = 'downtrend'
            regime['trade'] = False
            regime['reason'] = f"red_day_ret={spy_return:.3f}"
        
        # FLAT DAY
        else:
            if spy_range < 0.015:
                regime['regime'] = 'range_bound'
                regime['trade'] = True
                regime['reason'] = f"range_bound_narrow_{spy_range:.3f}"
            else:
                regime['regime'] = 'choppy'
                regime['trade'] = False
                regime['reason'] = f"choppy_range={spy_range:.3f}"
        
        self.last_regime_check = time.time()
        self.cached_regime = regime
        
        return regime

# ========== EVENT RISK MANAGER (from Phase 1) ==========
class EventRiskManager:
    def __init__(self):
        self.earnings_calendar = self._load_earnings_calendar()
        self.economic_calendar = self._load_economic_calendar()
        print("[IMPROVED] EventRiskManager initialized")
    
    def _load_earnings_calendar(self) -> Dict[str, datetime]:
        calendar = {}
        for symbol in CFG.symbols:
            if symbol not in ['SPY', 'QQQ', 'IWM', 'XLF', 'XLE']:
                days_ahead = hash(symbol) % 30
                calendar[symbol] = now_ny() + timedelta(days=days_ahead, hours=7)
        return calendar
    
    def _load_economic_calendar(self) -> List[Dict]:
        events = []
        next_fed = now_ny() + timedelta(days=14, hours=2)
        events.append({
            'type': 'FOMC',
            'datetime': next_fed,
            'importance': 'critical'
        })
        return events
    
    def get_upcoming_events(self, symbol: str, hours_ahead: int) -> List[Dict]:
        events = []
        cutoff = now_ny() + timedelta(hours=hours_ahead)
        
        if CFG.avoid_earnings and symbol in self.earnings_calendar:
            earnings_time = self.earnings_calendar[symbol]
            if now_ny() <= earnings_time <= cutoff:
                events.append({
                    'type': 'earnings',
                    'datetime': earnings_time,
                    'importance': 'high',
                    'symbol': symbol
                })
        
        if CFG.avoid_fed_days and symbol in ['SPY', 'QQQ', 'IWM']:
            for eco_event in self.economic_calendar:
                if now_ny() <= eco_event['datetime'] <= cutoff:
                    events.append(eco_event)
        
        return events
    
    def should_avoid_entry(self, symbol: str) -> Tuple[bool, str]:
        events = self.get_upcoming_events(symbol, CFG.event_lookback_hours)
        if events:
            event = events[0]
            hours_away = (event['datetime'] - now_ny()).total_seconds() / 3600
            reason = f"{event['type']}_in_{hours_away:.1f}h"
            return True, reason
        return False, ""
    
    def should_exit_before_event(self, symbol: str) -> Tuple[bool, str]:
        events = self.get_upcoming_events(symbol, CFG.event_exit_hours)
        if events:
            event = events[0]
            reason = f"{event['type']}_imminent"
            return True, reason
        return False, ""

# ========== ADAPTIVE STOP LOSS (from Phase 1) ==========
class AdaptiveStopLoss:
    def __init__(self):
        self.stop_cache: Dict[str, float] = {}
        self.historical_trades: Dict[str, List[Dict]] = collections.defaultdict(list)
        print("[IMPROVED] AdaptiveStopLoss initialized")
    
    def record_trade(self, symbol: str, trade_data: Dict):
        self.historical_trades[symbol].append(trade_data)
        cutoff = time.time() - (90 * 86400)
        self.historical_trades[symbol] = [
            t for t in self.historical_trades[symbol] 
            if t.get('entry_time', 0) > cutoff
        ]
    
    def calculate_optimal_stop(self, symbol: str, setup_type: str, 
                              direction: int, atr: float, account_size: float) -> float:
        """NEW: Account for account size in stop calculation"""
        cache_key = f"{symbol}_{setup_type}_{direction}"
        if cache_key in self.stop_cache:
            base_mult = self.stop_cache[cache_key]
        else:
            trades = [t for t in self.historical_trades[symbol] 
                     if t['setup_type'] == setup_type and t['direction'] == direction]
            
            if len(trades) < 20:
                # CRITICAL FIX #7: Wider stops for small accounts
                if account_size < 50000:
                    base_mult = CFG.base_stop_mult_small_account
                else:
                    base_mult = 2.0
            else:
                # Optimize from historical data
                test_multipliers = np.linspace(CFG.stop_range_min, CFG.stop_range_max, 20)
                best_expectancy = -float('inf')
                best_mult = 2.0
                
                for mult in test_multipliers:
                    wins = 0
                    losses = 0
                    win_amounts = []
                    loss_amounts = []
                    
                    for trade in trades:
                        stop_price = trade['entry_price'] - (direction * mult * trade['atr'])
                        
                        if direction > 0:
                            hit_stop = trade['min_price_after_entry'] <= stop_price
                        else:
                            hit_stop = trade['max_price_after_entry'] >= stop_price
                        
                        if hit_stop:
                            losses += 1
                            loss_amounts.append(mult * trade['atr'])
                        else:
                            wins += 1
                            if direction > 0:
                                profit = trade['max_price_after_entry'] - trade['entry_price']
                            else:
                                profit = trade['entry_price'] - trade['min_price_after_entry']
                            win_amounts.append(profit)
                    
                    if wins + losses > 0:
                        win_rate = wins / (wins + losses)
                        avg_win = np.mean(win_amounts) if win_amounts else 0
                        avg_loss = np.mean(loss_amounts) if loss_amounts else mult * atr
                        
                        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
                        
                        if expectancy > best_expectancy:
                            best_expectancy = expectancy
                            best_mult = mult
                
                base_mult = best_mult
                self.stop_cache[cache_key] = base_mult
        
        return base_mult
    
    def adjust_for_volatility(self, symbol: str, base_stop: float, 
                             current_vol: float, avg_vol: float) -> float:
        if not CFG.vol_adjust_stops:
            return base_stop
        vol_ratio = current_vol / max(avg_vol, 1e-6)
        adjusted = base_stop * (0.6 + 0.4 * vol_ratio)
        adjusted = np.clip(adjusted, base_stop * 0.7, base_stop * 1.6)
        return adjusted

# ========== TRANSACTION COST ANALYZER (from Phase 1) ==========
class TransactionCostAnalyzer:
    def __init__(self):
        self.execution_history: List[Dict] = []
        self.log_file = pathlib.Path("logs/tca_log.csv")
        self.log_file.parent.mkdir(exist_ok=True)
        
        if not self.log_file.exists():
            with open(self.log_file, 'w') as f:
                csv.writer(f).writerow([
                    'timestamp', 'symbol', 'side', 'shares', 
                    'arrival_price', 'fill_price', 'slippage_bps',
                    'time_to_fill_sec', 'hour', 'spread_bps'
                ])
        
        print("[IMPROVED] TransactionCostAnalyzer initialized")
    
    def record_execution(self, symbol: str, side: str, shares: int,
                        arrival_quote: Dict, fill_price: float,
                        fill_time: datetime):
        arrival_mid = (arrival_quote['bid'] + arrival_quote['ask']) / 2.0
        slippage = abs(fill_price - arrival_mid)
        slippage_bps = (slippage / arrival_mid) * 10000
        spread_bps = ((arrival_quote['ask'] - arrival_quote['bid']) / arrival_mid) * 10000
        
        record = {
            'timestamp': fill_time,
            'symbol': symbol,
            'side': side,
            'shares': shares,
            'arrival_price': arrival_mid,
            'fill_price': fill_price,
            'slippage_bps': slippage_bps,
            'time_to_fill_sec': 0,
            'hour': fill_time.hour,
            'spread_bps': spread_bps
        }
        
        self.execution_history.append(record)
        
        with open(self.log_file, 'a') as f:
            csv.writer(f).writerow([
                fill_time.isoformat(), symbol, side, shares,
                round(arrival_mid, 4), round(fill_price, 4),
                round(slippage_bps, 2), 0, fill_time.hour,
                round(spread_bps, 2)
            ])
    
    def analyze_execution_quality(self) -> Dict:
        if len(self.execution_history) < CFG.min_tca_samples:
            return {'status': 'insufficient_data'}
        
        df = pd.DataFrame(self.execution_history)
        
        analysis = {
            'avg_slippage_bps': float(df['slippage_bps'].mean()),
            'median_slippage_bps': float(df['slippage_bps'].median()),
            'p95_slippage_bps': float(df['slippage_bps'].quantile(0.95)),
            'worst_hour': int(df.groupby('hour')['slippage_bps'].mean().idxmax()),
            'best_hour': int(df.groupby('hour')['slippage_bps'].mean().idxmin()),
            'by_symbol': df.groupby('symbol')['slippage_bps'].mean().to_dict()
        }
        
        return analysis
    
    def should_trade_now(self, symbol: str) -> Tuple[bool, str]:
        if len(self.execution_history) < CFG.min_tca_samples:
            return True, "insufficient_data"
        
        current_hour = now_ny().hour
        df = pd.DataFrame(self.execution_history)
        similar = df[(df['symbol'] == symbol) & (df['hour'] == current_hour)]
        
        if len(similar) < 3:
            return True, "insufficient_data_for_hour"
        
        avg_cost = similar['slippage_bps'].mean()
        
        if avg_cost > CFG.max_acceptable_slippage_bps:
            return False, f"high_cost_hour_avg_{avg_cost:.1f}bps"
        
        return True, "acceptable_cost"

# ========== BASE POISSON COMPONENTS ==========
class PoissonEventDetector:
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.event_history: collections.deque = collections.deque(maxlen=lookback)
    
    def record_event(self, timestamp: float, magnitude: float):
        self.event_history.append((timestamp, magnitude))
    
    def estimate_lambda(self, time_window: float = 3600.0) -> float:
        if len(self.event_history) < CFG.min_events:
            return 0.0
        now = time.time()
        recent = [t for t, _ in self.event_history if now - t <= time_window]
        if len(recent) < 5:
            return 0.0
        lambda_est = len(recent) / (time_window / 3600.0)
        return lambda_est * CFG.lambda_scale
    
    def is_significant_event(self, current_value: float, threshold_std: float = 2.5) -> bool:
        if len(self.event_history) < CFG.min_events:
            return False
        magnitudes = [m for _, m in self.event_history]
        mean = np.mean(magnitudes)
        std = np.std(magnitudes, ddof=1)
        if std == 0:
            return False
        z_score = (current_value - mean) / std
        return abs(z_score) >= threshold_std

class MarketRegime:
    @staticmethod
    def classify_volatility(returns: pd.Series) -> str:
        if len(returns) < 20:
            return "unknown"
        vol = returns.std() * np.sqrt(252)
        if vol < 0.12:
            return "low_vol"
        elif vol < 0.25:
            return "normal_vol"
        else:
            return "high_vol"
    
    @staticmethod
    def classify_trend(prices: pd.Series, windows: List[int] = [20, 50]) -> str:
        if len(prices) < max(windows) + 5:
            return "neutral"
        smas = {w: prices.rolling(w).mean().iloc[-1] for w in windows}
        current = prices.iloc[-1]
        if all(current > sma for sma in smas.values()):
            return "uptrend"
        elif all(current < sma for sma in smas.values()):
            return "downtrend"
        else:
            return "neutral"
    
    @staticmethod
    def get_regime(df: pd.DataFrame) -> Dict[str, str]:
        if df.empty or len(df) < 50:
            return {"vol": "unknown", "trend": "neutral"}
        returns = df['c'].pct_change().dropna()
        return {
            "vol": MarketRegime.classify_volatility(returns),
            "trend": MarketRegime.classify_trend(df['c'])
        }

class SignalGenerator:
    def __init__(self):
        self.breakout_detector = PoissonEventDetector()
        self.reversion_detector = PoissonEventDetector()
        print("[IMPROVED] SignalGenerator initialized")
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['atr'] = (df['h'] - df['l']).rolling(14, min_periods=5).mean()
        returns = df['c'].pct_change()
        df['z_ret'] = (returns - returns.rolling(60).mean()) / returns.rolling(60).std()
        df['z_vol'] = (df['v'] - df['v'].rolling(60).mean()) / df['v'].rolling(60).std()
        ma20 = df['c'].rolling(20).mean()
        df['dist_ma'] = (df['c'] - ma20) / ma20
        df['mom'] = df['c'].pct_change(5)
        return df.dropna()
    
    def check_confirmation(self, df: pd.DataFrame, direction: int) -> bool:
        """NEW: CRITICAL FIX #3 - Require confirmation bars"""
        if len(df) < CFG.require_confirmation_bars + 1:
            return False
        
        last_bars = df.tail(CFG.require_confirmation_bars + 1)
        
        if direction > 0:
            # For long: need majority of bars closing higher
            higher_closes = (last_bars['c'].diff() > 0).sum()
            return higher_closes >= CFG.require_confirmation_bars
        else:
            # For short: need majority of bars closing lower
            lower_closes = (last_bars['c'].diff() < 0).sum()
            return lower_closes >= CFG.require_confirmation_bars
    
    def detect_breakout_event(self, df: pd.DataFrame, regime: Dict) -> Optional[Dict]:
        if len(df) < 50:
            return None
        feat = self.compute_features(df)
        if feat.empty:
            return None
        
        breakout_score = (
            abs(feat['z_ret'].iloc[-1]) * 0.4 +
            abs(feat['z_vol'].iloc[-1]) * 0.3 +
            abs(feat['dist_ma'].iloc[-1]) * 100 * 0.3
        )
        
        self.breakout_detector.record_event(time.time(), breakout_score)
        
        if not self.breakout_detector.is_significant_event(breakout_score, threshold_std=2.0):
            return None
        
        direction = 1 if feat['mom'].iloc[-1] > 0 else -1
        
        # NEW: Check confirmation
        if not self.check_confirmation(df, direction):
            return None
        
        if regime.get("trend") == "neutral":
            return None
        if direction > 0 and regime.get("trend") != "uptrend":
            return None
        if direction < 0 and regime.get("trend") != "downtrend":
            return None
        
        return {
            "type": "breakout",
            "direction": direction,
            "strength": breakout_score,
            "atr": feat['atr'].iloc[-1]
        }
    
    def detect_reversion_event(self, df: pd.DataFrame, regime: Dict) -> Optional[Dict]:
        if len(df) < 50:
            return None
        feat = self.compute_features(df)
        if feat.empty:
            return None
        
        z_ret = feat['z_ret'].iloc[-1]
        reversion_score = abs(z_ret) * (1.0 - min(1.0, feat['z_vol'].iloc[-1] / 3.0))
        
        self.reversion_detector.record_event(time.time(), reversion_score)
        
        if regime.get("trend") != "neutral":
            return None
        if abs(z_ret) < 2.0:
            return None
        if not self.reversion_detector.is_significant_event(reversion_score, threshold_std=1.8):
            return None
        
        direction = -1 if z_ret > 0 else 1
        
        # NEW: Check confirmation
        if not self.check_confirmation(df, direction):
            return None
        
        return {
            "type": "reversion",
            "direction": direction,
            "strength": reversion_score,
            "atr": feat['atr'].iloc[-1]
        }
    
    def generate_signal(self, df: pd.DataFrame, regime: Dict) -> Optional[Dict]:
        if regime.get("trend") in ["uptrend", "downtrend"]:
            signal = self.detect_breakout_event(df, regime)
            if signal:
                return signal
        if regime.get("trend") == "neutral":
            signal = self.detect_reversion_event(df, regime)
            if signal:
                return signal
        return None

# ========== POSITION SIZING & RISK ==========
class KellyPositionSizer:
    def __init__(self):
        self.trade_history: List[Dict] = []
    
    def record_trade(self, pnl: float, risk: float):
        self.trade_history.append({"pnl": pnl, "risk": risk})
    
    def estimate_edge(self) -> Tuple[float, float]:
        if len(self.trade_history) < 10:
            return 0.0, 0.0
        recent = self.trade_history[-50:]
        wins = [t["pnl"] for t in recent if t["pnl"] > 0]
        losses = [abs(t["pnl"]) for t in recent if t["pnl"] < 0]
        if not wins or not losses:
            return 0.0, 0.0
        win_rate = len(wins) / len(recent)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        edge = win_rate * avg_win - (1 - win_rate) * avg_loss
        return win_rate, edge / avg_loss if avg_loss > 0 else 0.0
    
    def kelly_fraction(self, win_rate: float, win_loss_ratio: float) -> float:
        if win_rate <= 0.5 or win_loss_ratio <= 1.0:
            return 0.0
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        return max(0.0, min(kelly * CFG.kelly_fraction, CFG.max_position_risk))
    
    def calculate_position_size(self, signal_strength: float, atr: float,
                               price: float, equity: float) -> int:
        win_rate, edge_ratio = self.estimate_edge()
        if len(self.trade_history) < 10:
            win_rate = 0.55
            edge_ratio = 1.3
        if win_rate < CFG.min_win_rate or edge_ratio < 1.0:
            return 0
        kelly_f = self.kelly_fraction(win_rate, edge_ratio)
        if kelly_f <= 0:
            return 0
        strength_mult = 0.5 + min(1.0, signal_strength / 5.0)
        position_risk = equity * kelly_f * strength_mult
        stop_distance = CFG.base_stop_mult_small_account * atr
        shares = int(position_risk / stop_distance)
        max_shares = int((equity * CFG.max_position_risk) / stop_distance)
        return min(shares, max_shares)

class RiskManager:
    def __init__(self, initial_equity: float):
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.daily_pnl = 0.0
        self.positions: Dict[str, Dict] = {}
        self.trades_today = 0
        self.daily_peak = 0.0
        self.position_sizer = KellyPositionSizer()
        
        # CRITICAL FIX #4: Symbol cooldown tracking
        self.symbol_last_trade: Dict[str, float] = {}
        self.symbol_trade_count: Dict[str, int] = collections.defaultdict(int)
        
        print("[IMPROVED] RiskManager initialized with cooldowns")
    
    def update_equity(self, equity: float):
        self.current_equity = equity
        self.daily_pnl = equity - self.initial_equity
        if self.daily_pnl > self.daily_peak:
            self.daily_peak = self.daily_pnl
    
    def check_daily_stop(self) -> bool:
        loss_pct = -self.daily_pnl / self.initial_equity
        return loss_pct >= CFG.max_daily_loss
    
    def can_trade_symbol(self, symbol: str) -> Tuple[bool, str]:
        """CRITICAL FIX #4: Check cooldown"""
        last_trade = self.symbol_last_trade.get(symbol, 0)
        time_since = time.time() - last_trade
        
        if time_since < CFG.symbol_cooldown_seconds:
            remaining = CFG.symbol_cooldown_seconds - time_since
            return False, f"cooldown_{remaining:.0f}s"
        
        return True, ""
    
    def record_symbol_trade(self, symbol: str):
        """Mark that we just traded this symbol"""
        self.symbol_last_trade[symbol] = time.time()
        self.symbol_trade_count[symbol] += 1
    
    def can_add_position(self, symbol: str, direction: int) -> Tuple[bool, str]:
        # Max trades per day
        if self.trades_today >= CFG.max_trades_per_day:
            return False, f"max_daily_trades_{CFG.max_trades_per_day}"
        
        # Max positions
        if len(self.positions) >= CFG.max_positions:
            return False, f"max_positions_{CFG.max_positions}"
        
        # Already have position
        if symbol in self.positions:
            return False, "already_have_position"
        
        # Correlated positions
        same_direction = sum(1 for p in self.positions.values() if p["direction"] == direction)
        if same_direction >= CFG.max_correlated_positions:
            return False, f"max_correlated_{CFG.max_correlated_positions}"
        
        # Daily stop
        if self.check_daily_stop():
            return False, "daily_stop_active"
        
        return True, ""
    
    def total_risk_exposure(self) -> float:
        total_risk = sum(p["risk_dollars"] for p in self.positions.values())
        return total_risk / self.current_equity
    
    def calculate_position_size(self, symbol: str, signal: Dict, price: float) -> int:
        """CRITICAL FIX #1: Check available capital"""
        
        # Check risk budget
        current_risk = self.total_risk_exposure()
        if current_risk >= CFG.max_portfolio_risk:
            return 0
        
        remaining_risk = CFG.max_portfolio_risk - current_risk
        max_position_risk = min(CFG.max_position_risk, remaining_risk)
        
        # Kelly sizing
        shares = self.position_sizer.calculate_position_size(
            signal["strength"], signal["atr"], price, self.current_equity
        )
        
        # Position risk check
        position_risk = shares * CFG.base_stop_mult_small_account * signal["atr"]
        if position_risk / self.current_equity > max_position_risk:
            shares = int((self.current_equity * max_position_risk) / 
                        (CFG.base_stop_mult_small_account * signal["atr"]))
        
        # CRITICAL FIX #1: Check actual available capital
        try:
            account = rest.get_account()
            buying_power = float(account.buying_power)
            cash = float(account.cash)
            
            # Use conservative estimate
            available = min(buying_power, cash) * CFG.capital_buffer
            
            # Don't exceed capital per position limit
            max_notional = available * CFG.capital_usage_per_position
            max_shares_by_capital = int(max_notional / price)
            
            shares = min(shares, max_shares_by_capital)
            
            print(f"[CAPITAL] {symbol}: Kelly={shares} CapLimit={max_shares_by_capital} "
                  f"Available=${available:,.0f}")
            
        except Exception as e:
            print(f"[ERROR] Capital check failed: {e}")
            return 0
        
        return max(0, shares)
    
    def add_position(self, symbol: str, direction: int, shares: int,
                    entry_price: float, signal: Dict, stop_mult: float):
        stop_distance = stop_mult * signal["atr"]
        stop_price = entry_price - (direction * stop_distance)
        target_price = entry_price + (direction * 3.0 * signal["atr"])
        
        self.positions[symbol] = {
            "direction": direction,
            "shares": shares,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "risk_dollars": shares * stop_distance,
            "entry_time": time.time(),
            "signal_type": signal["type"],
            "stop_mult": stop_mult,
            "atr": signal["atr"]
        }
        
        self.trades_today += 1
        self.record_symbol_trade(symbol)
    
    def remove_position(self, symbol: str, exit_price: float):
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        pnl = (exit_price - pos["entry_price"]) * pos["direction"] * pos["shares"]
        self.position_sizer.record_trade(pnl, pos["risk_dollars"])
        del self.positions[symbol]
    
    def check_exits(self, quotes: Dict[str, Dict], 
                   event_mgr: EventRiskManager) -> List[Tuple[str, str]]:
        to_exit = []
        
        for symbol, pos in list(self.positions.items()):
            # Event risk
            should_exit_event, event_reason = event_mgr.should_exit_before_event(symbol)
            if should_exit_event:
                to_exit.append((symbol, f"event_risk_{event_reason}"))
                continue
            
            if symbol not in quotes:
                continue
            
            quote = quotes[symbol]
            if not quote or quote.get("mid") is None:
                continue
            
            mid = quote["mid"]
            direction = pos["direction"]
            time_held = time.time() - pos["entry_time"]
            
            # CRITICAL FIX #6: Minimum hold time (prevent noise exits)
            if time_held < CFG.min_hold_seconds:
                # Only exit on hard stop during min hold period
                if direction > 0:
                    if mid <= pos["stop_price"]:
                        to_exit.append((symbol, "stop_loss"))
                else:
                    if mid >= pos["stop_price"]:
                        to_exit.append((symbol, "stop_loss"))
                continue
            
            # Normal exit logic after min hold
            # Time stop
            if time_held > 180 * 60:  # 3 hours
                to_exit.append((symbol, "time_stop"))
                continue
            
            # Price stops
            if direction > 0:
                if mid <= pos["stop_price"]:
                    to_exit.append((symbol, "stop_loss"))
                elif mid >= pos["target_price"]:
                    to_exit.append((symbol, "take_profit"))
            else:
                if mid >= pos["stop_price"]:
                    to_exit.append((symbol, "stop_loss"))
                elif mid <= pos["target_price"]:
                    to_exit.append((symbol, "take_profit"))
        
        return to_exit

# ========== DATA & EXECUTION ==========
def get_bars(symbols: List[str], minutes: int) -> Dict[str, pd.DataFrame]:
    end = datetime.now(pytz.UTC)
    start = end - timedelta(minutes=minutes + 30)
    
    result = {}
    try:
        multi = rest.get_bars(
            symbols,
            tradeapi.rest.TimeFrame.Minute,
            start.isoformat(),
            end.isoformat(),
            feed=DATA_FEED
        ).df
        
        for sym in symbols:
            try:
                df = multi.xs(sym, level=0) if isinstance(multi.index, pd.MultiIndex) else multi[multi.get('symbol') == sym]
                df = df.rename(columns={'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v'})
                df = df[['o', 'h', 'l', 'c', 'v']].dropna()
                result[sym] = df
            except:
                result[sym] = pd.DataFrame()
    except Exception as e:
        print(f"[ERROR] Fetching bars: {e}")
        result = {s: pd.DataFrame() for s in symbols}
    
    return result

def get_quotes(symbols: List[str]) -> Dict[str, Dict]:
    quotes = {}
    try:
        quote_map = rest.get_latest_quotes(symbols, feed=DATA_FEED)
        for sym in symbols:
            q = quote_map.get(sym)
            if q:
                bid = getattr(q, 'bp', None) or getattr(q, 'bid_price', None)
                ask = getattr(q, 'ap', None) or getattr(q, 'ask_price', None)
                if bid and ask:
                    quotes[sym] = {
                        "bid": float(bid),
                        "ask": float(ask),
                        "mid": (float(bid) + float(ask)) / 2.0,
                        "spread_bps": ((float(ask) - float(bid)) / ((float(bid) + float(ask)) / 2.0)) * 10000
                    }
    except Exception as e:
        print(f"[ERROR] Fetching quotes: {e}")
    
    return quotes

def execute_order(symbol: str, side: str, qty: int) -> bool:
    try:
        rest.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='day'
        )
        print(f"[EXEC] {side.upper()} {qty} {symbol}")
        return True
    except Exception as e:
        print(f"[ERROR] Order execution: {e}")
        return False

# ========== MAIN LOOP ==========
def main():
    print("="*80)
    print("PHASE 1 IMPROVED: ALL CRITICAL FIXES APPLIED")
    print("="*80)
    print("Fixes:")
    print("  ✓ Position sizing with capital checks")
    print("  ✓ Market regime filter (green days only)")
    print("  ✓ Signal strength threshold")
    print("  ✓ Symbol cooldown (10 min)")
    print("  ✓ Optimal hours only (10:30 AM - 2:30 PM)")
    print("  ✓ Minimum hold time (5 min)")
    print("  ✓ Wider stops for small account")
    print("  ✓ Dynamic spread filtering")
    print("  ✓ Confirmation bars required")
    print("  ✓ Max 5 trades per day")
    print("="*80)
    
    # Initialize
    try:
        equity = float(rest.get_account().equity)
        print(f"[INFO] Broker equity: ${equity:,.2f}")
    except:
        equity = CFG.initial_equity
        print(f"[INFO] Using config equity: ${equity:,.2f}")
    
    risk_mgr = RiskManager(equity)
    signal_gen = SignalGenerator()
    event_mgr = EventRiskManager()
    adaptive_stops = AdaptiveStopLoss()
    tca = TransactionCostAnalyzer()
    market_filter = MarketRegimeFilter()
    
    # Logging
    log_dir = pathlib.Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "phase1_improved.csv"
    
    if not log_file.exists():
        with open(log_file, 'w') as f:
            csv.writer(f).writerow([
                "timestamp", "action", "symbol", "direction", "shares", "price",
                "signal_type", "signal_strength", "stop_mult", "reason", "market_regime"
            ])
    
    def log(action, symbol="", direction=0, shares=0, price=0.0, 
            signal_type="", strength=0.0, stop_mult=0.0, reason="", market_regime=""):
        with open(log_file, 'a') as f:
            csv.writer(f).writerow([
                datetime.now(NY).isoformat(), action, symbol, direction, shares,
                round(price, 2), signal_type, round(strength, 2), 
                round(stop_mult, 2), reason, market_regime
            ])
    
    print(f"\n[INFO] Max positions: {CFG.max_positions}")
    print(f"[INFO] Max trades/day: {CFG.max_trades_per_day}")
    print(f"[INFO] Min signal strength: {CFG.min_signal_strength}")
    print(f"[INFO] Symbol cooldown: {CFG.symbol_cooldown_seconds}s")
    print(f"[INFO] Trading hours: {CFG.optimal_start_hour}:{CFG.optimal_start_minute:02d} - "
          f"{CFG.optimal_end_hour}:{CFG.optimal_end_minute:02d}")
    print()
    
    loop_count = 0
    last_market_regime_log = ""
    
    while True:
        try:
            if not is_market_open():
                print("[WAIT] Market closed")
                time.sleep(60)
                continue
            
            # CRITICAL: Check optimal trading hours
            in_hours, hours_reason = is_in_optimal_trading_hours()
            if not in_hours:
                if loop_count % 10 == 0:
                    print(f"[WAIT] Outside optimal hours: {hours_reason}")
                time.sleep(30)
                continue
            
            # Update equity
            try:
                equity = float(rest.get_account().equity)
                risk_mgr.update_equity(equity)
            except:
                pass
            
            # Check daily stop
            if risk_mgr.check_daily_stop():
                print(f"\n{'='*80}")
                print(f"[STOP] Daily stop hit. P&L: ${risk_mgr.daily_pnl:,.2f}")
                print(f"{'='*80}\n")
                
                # Flatten all
                for symbol in list(risk_mgr.positions.keys()):
                    pos = risk_mgr.positions[symbol]
                    side = "sell" if pos["direction"] > 0 else "buy"
                    execute_order(symbol, side, pos["shares"])
                    risk_mgr.remove_position(symbol, 0.0)
                
                time.sleep(300)
                continue
            
            # Get data
            bars = get_bars(list(CFG.symbols), 300)
            quotes = get_quotes(list(CFG.symbols))
            
            # CRITICAL FIX #2: Check market regime
            spy_bars = bars.get('SPY', pd.DataFrame())
            market_regime = market_filter.get_market_regime(spy_bars)
            
            regime_str = f"{market_regime.get('regime', 'unknown')}"
            if regime_str != last_market_regime_log:
                print(f"\n[REGIME] {regime_str} | SPY: {market_regime.get('spy_return', 0):.3f} "
                      f"Range: {market_regime.get('spy_range', 0):.3f} | "
                      f"Trade: {market_regime.get('trade', False)}")
                last_market_regime_log = regime_str
            
            if not market_regime.get('trade', False):
                if loop_count % 10 == 0:
                    log("SKIP_REGIME", "", 0, 0, 0, "", 0, 0, 
                        market_regime.get('reason', 'bad_regime'), regime_str)
                time.sleep(60)  # Wait longer in bad regime
                loop_count += 1
                continue
            
            # Check exits
            exits = risk_mgr.check_exits(quotes, event_mgr)
            for symbol, reason in exits:
                pos = risk_mgr.positions[symbol]
                side = "sell" if pos["direction"] > 0 else "buy"
                quote = quotes.get(symbol, {})
                exit_price = quote.get("mid", pos["entry_price"])
                
                if CFG.track_execution_quality and quote:
                    tca.record_execution(symbol, side, pos["shares"], 
                                        quote, exit_price, now_ny())
                
                if execute_order(symbol, side, pos["shares"]):
                    log("EXIT", symbol, 0, pos["shares"], exit_price, 
                        pos["signal_type"], 0, pos.get("stop_mult", 2.0), 
                        reason, regime_str)
                    
                    trade_data = {
                        'entry_time': pos['entry_time'],
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'direction': pos['direction'],
                        'setup_type': pos['signal_type'],
                        'atr': pos['atr'],
                        'min_price_after_entry': exit_price,
                        'max_price_after_entry': exit_price
                    }
                    adaptive_stops.record_trade(symbol, trade_data)
                    
                    risk_mgr.remove_position(symbol, exit_price)
            
            # Look for new entries
            if len(risk_mgr.positions) < CFG.max_positions:
                for symbol in CFG.symbols:
                    if symbol == 'SPY':
                        continue
                    
                    if symbol in risk_mgr.positions:
                        continue
                    
                    # Check symbol cooldown
                    can_trade, cooldown_reason = risk_mgr.can_trade_symbol(symbol)
                    if not can_trade:
                        continue
                    
                    # Event check
                    should_avoid, avoid_reason = event_mgr.should_avoid_entry(symbol)
                    if should_avoid:
                        continue
                    
                    # Get data
                    df = bars.get(symbol, pd.DataFrame())
                    quote = quotes.get(symbol, {})
                    
                    if df.empty or not quote:
                        continue
                    
                    # CRITICAL FIX #8: Dynamic spread check
                    max_spread = get_max_spread_bps()
                    if quote.get("spread_bps", 999) > max_spread:
                        continue
                    
                    # TCA check
                    if CFG.track_execution_quality:
                        can_trade_tca, tca_reason = tca.should_trade_now(symbol)
                        if not can_trade_tca:
                            continue
                    
                    # Get regime
                    regime = MarketRegime.get_regime(df)
                    
                    # Generate signal
                    signal = signal_gen.generate_signal(df, regime)
                    
                    if not signal:
                        continue
                    
                    # CRITICAL FIX #3: Check signal strength
                    if signal['strength'] < CFG.min_signal_strength:
                        if loop_count % 20 == 0:
                            log("SKIP", symbol, 0, 0, 0, signal['type'], 
                                signal['strength'], 0, 
                                f"weak_signal_{signal['strength']:.2f}", regime_str)
                        continue
                    
                    direction = signal["direction"]
                    
                    # Check if can add
                    can_add, add_reason = risk_mgr.can_add_position(symbol, direction)
                    if not can_add:
                        if loop_count % 20 == 0:
                            log("SKIP", symbol, direction, 0, 0, signal['type'],
                                signal['strength'], 0, add_reason, regime_str)
                        continue
                    
                    # Calculate size
                    shares = risk_mgr.calculate_position_size(symbol, signal, quote["mid"])
                    
                    if shares == 0:
                        log("SKIP", symbol, direction, 0, 0, signal['type'],
                            signal['strength'], 0, "size_calc_zero", regime_str)
                        continue
                    
                    # Calculate adaptive stop
                    base_stop_mult = adaptive_stops.calculate_optimal_stop(
                        symbol, signal["type"], direction, signal["atr"], equity
                    )
                    
                    if CFG.vol_adjust_stops:
                        returns = df['c'].pct_change().dropna()
                        current_vol = returns.tail(20).std() * np.sqrt(252)
                        avg_vol = returns.tail(60).std() * np.sqrt(252)
                        stop_mult = adaptive_stops.adjust_for_volatility(
                            symbol, base_stop_mult, current_vol, avg_vol
                        )
                    else:
                        stop_mult = base_stop_mult
                    
                    # Execute
                    side = "buy" if direction > 0 else "sell"
                    arrival_quote = quote
                    
                    if execute_order(symbol, side, shares):
                        fill_price = quote["mid"]
                        
                        if CFG.track_execution_quality:
                            tca.record_execution(symbol, side, shares, 
                                                arrival_quote, fill_price, now_ny())
                        
                        risk_mgr.add_position(symbol, direction, shares, 
                                             quote["mid"], signal, stop_mult)
                        
                        log("ENTRY", symbol, direction, shares, quote["mid"],
                            signal["type"], signal["strength"], stop_mult,
                            f"regime={regime['trend']}_stop={stop_mult:.2f}ATR", 
                            market_regime.get('regime', 'unknown'))
                        
                        print(f"\n[TRADE] {side.upper()} {shares} {symbol} @ ${quote['mid']:.2f}")
                        print(f"        Signal: {signal['type']} strength={signal['strength']:.2f}")
                        print(f"        Stop: {stop_mult:.2f} ATR = ${quote['mid'] - (direction * stop_mult * signal['atr']):.2f}")
                        print(f"        Risk: ${shares * stop_mult * signal['atr']:.2f}\n")
            
            # TCA report
            if loop_count > 0 and loop_count % 100 == 0:
                analysis = tca.analyze_execution_quality()
                if analysis.get('status') != 'insufficient_data':
                    print(f"\n[TCA] Execution Report:")
                    print(f"  Avg slippage: {analysis['avg_slippage_bps']:.2f} bps")
                    print(f"  P95 slippage: {analysis['p95_slippage_bps']:.2f} bps\n")
            
            # Status
            if loop_count % 5 == 0:
                print(f"[STATUS] Loop: {loop_count} | "
                      f"Pos: {len(risk_mgr.positions)}/{CFG.max_positions} | "
                      f"Trades: {risk_mgr.trades_today}/{CFG.max_trades_per_day} | "
                      f"P&L: ${risk_mgr.daily_pnl:+,.2f} | "
                      f"Risk: {risk_mgr.total_risk_exposure()*100:.1f}%")
            
            loop_count += 1
            time.sleep(60)  # CRITICAL FIX #5: Slower loop (was 30s)
            
        except KeyboardInterrupt:
            print("\n[STOP] Shutting down...")
            break
        except Exception as e:
            print(f"[ERROR] Loop error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(30)
    
    # Final report
    print("\n" + "="*80)
    print("SESSION SUMMARY")
    print("="*80)
    print(f"Total trades: {risk_mgr.trades_today}")
    print(f"Final P&L: ${risk_mgr.daily_pnl:+,.2f}")
    print(f"Win rate: {len([t for t in risk_mgr.position_sizer.trade_history if t['pnl']>0])} / {len(risk_mgr.position_sizer.trade_history)}")
    
    analysis = tca.analyze_execution_quality()
    if analysis.get('status') != 'insufficient_data':
        print(f"\nExecution Quality:")
        print(f"  Average slippage: {analysis['avg_slippage_bps']:.2f} bps")
        print(f"  95th percentile: {analysis['p95_slippage_bps']:.2f} bps")
    
    print(f"\nLogs saved to: {log_file}")
    print("="*80)

if __name__ == "__main__":
    main()
