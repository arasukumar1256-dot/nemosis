
import os
import time
from datetime import datetime

import cv2
import mss
import numpy as np
import pandas as pd
import pyautogui

# ==================================================
# CONFIGURATION
# ==================================================

CAPTURE_INTERVAL = 5
MIN_CONFIDENCE = 50

# Keep demo mode enabled. This program must not be
# used to execute real-money binary trades.
DEMO_MODE = True
ALLOW_MOUSE_CLICK = False

# Set these to the actual button centers on your
# DEMO screen. The supplied coordinates were identical.
CALL_POSITION = (1141, 426)
PUT_POSITION = (1140, 484)

DATA_FILE = "candles.csv"
SCREENSHOT_DIR = "screenshots1"

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Safety: moving the mouse to a screen corner can
# trigger PyAutoGUI's emergency fail-safe.
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.25


# ==================================================
# SCREEN CAPTURE
# ==================================================

def capture_screen():
    """Capture and save the entire screen."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = np.array(sct.grab(monitor))
        image = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)

    filename = os.path.join(
        SCREENSHOT_DIR,
        datetime.now().strftime("screen_%Y%m%d_%H%M%S.png")
    )

    cv2.imwrite(filename, image)
    print(f"Screenshot saved: {filename}")

    return image


# ==================================================
# INDICATORS
# ==================================================

def calculate_indicators(df):
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Exponential moving averages
    df["ema9"] = close.ewm(span=9, adjust=False).mean()
    df["ema21"] = close.ewm(span=21, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()

    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands (20, 2)
    middle = close.rolling(20).mean()
    std = close.rolling(20).std()

    df["bb_upper"] = middle + 2 * std
    df["bb_lower"] = middle - 2 * std

    # ADX (14), Wilder-style smoothing
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move, 0.0
        ), index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move, 0.0
        ), index=df.index
    )

    previous_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    plus_di = 100 * (
        plus_dm.ewm(alpha=1 / 14, adjust=False).mean()
        / atr.replace(0, np.nan)
    )

    minus_di = 100 * (
        minus_dm.ewm(alpha=1 / 14, adjust=False).mean()
        / atr.replace(0, np.nan)
    )

    dx = 100 * (
        (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    df["adx"] = dx.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    return df


# ==================================================
# CANDLE PATTERNS
# ==================================================

def detect_candle_patterns(df):
    current = df.iloc[-1]
    previous = df.iloc[-2]

    body = abs(current["close"] - current["open"])
    candle_range = current["high"] - current["low"]

    if candle_range <= 0:
        return []

    lower_wick = min(
        current["open"], current["close"]
    ) - current["low"]

    upper_wick = current["high"] - max(
        current["open"], current["close"]
    )

    patterns = []

    # Bullish engulfing
    if (
        previous["close"] < previous["open"]
        and current["close"] > current["open"]
        and current["open"] <= previous["close"]
        and current["close"] >= previous["open"]
    ):
        patterns.append("bullish_engulfing")

    # Bearish engulfing
    if (
        previous["close"] > previous["open"]
        and current["close"] < current["open"]
        and current["open"] >= previous["close"]
        and current["close"] <= previous["open"]
    ):
        patterns.append("bearish_engulfing")

    # Hammer-like candle; requires context for confirmation
    if (
        lower_wick >= 2 * max(body, 1e-12)
        and upper_wick <= candle_range * 0.20
    ):
        patterns.append("hammer_like")

    # Shooting-star-like candle
    if (
        upper_wick >= 2 * max(body, 1e-12)
        and lower_wick <= candle_range * 0.20
    ):
        patterns.append("shooting_star_like")

    return patterns


# ==================================================
# SIGNAL ENGINE
# ==================================================

def analyze_market(df):
    if len(df) < 100:
        return {
            "signal": "NO TRADE",
            "confidence": 0,
            "reason": "Need at least 100 candles"
        }

    df = calculate_indicators(df)
    row = df.iloc[-1]

    required = [
        "ema9", "ema21", "ema50", "rsi",
        "macd", "macd_signal", "bb_upper",
        "bb_lower", "adx", "plus_di", "minus_di"
    ]

    if row[required].isna().any():
        return {
            "signal": "NO TRADE",
            "confidence": 0,
            "reason": "Indicators are not ready"
        }

    patterns = detect_candle_patterns(df)

    # This is a rule score, NOT a calibrated win probability.
    call_score = 0
    put_score = 0
    reasons = []

    # EMA trend alignment
    if row["ema9"] > row["ema21"] > row["ema50"]:
        call_score += 2
        reasons.append("Bullish EMA alignment")

    elif row["ema9"] < row["ema21"] < row["ema50"]:
        put_score += 2
        reasons.append("Bearish EMA alignment")

    # MACD momentum
    if row["macd"] > row["macd_signal"]:
        call_score += 1
        reasons.append("MACD bullish")

    elif row["macd"] < row["macd_signal"]:
        put_score += 1
        reasons.append("MACD bearish")

    # RSI: avoid treating extreme readings as automatic entries
    if 50 < row["rsi"] < 70:
        call_score += 1
        reasons.append("RSI supports bullish momentum")

    elif 30 < row["rsi"] < 50:
        put_score += 1
        reasons.append("RSI supports bearish momentum")

    # ADX confirms trend strength, not its direction
    if row["adx"] >= 20:
        if row["plus_di"] > row["minus_di"]:
            call_score += 1
            reasons.append("ADX directional support: bullish")
        elif row["minus_di"] > row["plus_di"]:
            put_score += 1
            reasons.append("ADX directional support: bearish")
    else:
        reasons.append("Weak trend strength")

    # Candlestick confirmation
    if "bullish_engulfing" in patterns:
        call_score += 2
        reasons.append("Bullish engulfing")

    if "bearish_engulfing" in patterns:
        put_score += 2
        reasons.append("Bearish engulfing")

    if "hammer_like" in patterns:
        call_score += 1

    if "shooting_star_like" in patterns:
        put_score += 1

    # Rule score is capped at 100 but is NOT a probability.
    total = call_score + put_score

    if total == 0 or call_score == put_score:
        return {
            "signal": "NO TRADE",
            "confidence": 0,
            "reason": "No clear directional agreement"
        }

    signal = "CALL" if call_score > put_score else "PUT"
    winning_score = max(call_score, put_score)

    # Agreement score only. Do not interpret as win probability.
    confidence = round(100 * winning_score / total)

    # Require minimum trend strength and score agreement.
    if row["adx"] < 20:
        return {
            "signal": "NO TRADE",
            "confidence": confidence,
            "reason": "Trend-strength filter rejected the setup"
        }

    if confidence < MIN_CONFIDENCE:
        return {
            "signal": "NO TRADE",
            "confidence": confidence,
            "reason": "Directional score below threshold"
        }

    return {
        "signal": signal,
        "confidence": confidence,
        "reason": "; ".join(reasons) or "Rule agreement",
        "patterns": patterns
    }


# ==================================================
# DEMO-ONLY MOUSE ACTION
# ==================================================

def execute_demo_signal(signal, confidence):
    if signal not in ("CALL", "PUT"):
        print("NO TRADE: mouse will not click.")
        return

    if confidence < MIN_CONFIDENCE:
        print("Below threshold: mouse will not click.")
        return

    if not DEMO_MODE or not ALLOW_MOUSE_CLICK:
        print("Demo mouse execution disabled.")
        print("Signal was recorded, but no click was made.")
        return

    # Do not enable automatic order execution for live funds.
    print("Demo execution is intentionally locked.")
    print("Use the signal for manual demo testing instead.")


# ==================================================
# MAIN LOOP
# ==================================================

def main():
    print("Binary market analysis agent started.")
    print("Screen capture is enabled.")
    print("Real-money execution is disabled.")
    print("Waiting for candles.csv ...")

    while True:
        try:
            capture_screen()

            if not os.path.exists(DATA_FILE):
                print("Missing candles.csv. No signal generated.")
                time.sleep(CAPTURE_INTERVAL)
                continue

            df = pd.read_csv(DATA_FILE)
            df.columns = [
                column.strip().lower() for column in df.columns
            ]

            required = {"open", "high", "low", "close"}

            if not required.issubset(df.columns):
                print(
                    "CSV needs columns: open, high, low, close"
                )
                time.sleep(CAPTURE_INTERVAL)
                continue

            for column in required:
                df[column] = pd.to_numeric(
                    df[column], errors="coerce"
                )

            df = df.dropna(
                subset=["open", "high", "low", "close"]
            ).reset_index(drop=True)

            result = analyze_market(df)

            print("\n", "=" * 45)
            print("Time:", datetime.now().strftime("%H:%M:%S"))
            print("Signal:", result["signal"])
            print("Agreement score:", result["confidence"], "%")
            print("Reason:", result["reason"])
            print("=" * 45)

            execute_demo_signal(
                result["signal"], result["confidence"]
            )

        except pyautogui.FailSafeException:
            print("Emergency stop activated.")
            break

        except Exception as error:
            print("Analysis error:", error)

        time.sleep(CAPTURE_INTERVAL)


if __name__ == "__main__":
    main()
