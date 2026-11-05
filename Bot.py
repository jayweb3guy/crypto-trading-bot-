import os
import json
import requests
from datetime import datetime, timedelta, timezone
from google import genai
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

# 1. Load Secrets from Environment Variables (matching your GitHub Secrets)
ALPACA_API_KEY = os.environ.get("CRYPTO_TRADING_BOT")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_ID")

# Trading Configuration
SYMBOL = "BTC/USD"
TRADE_NOTIONAL_USD = 25.0  # Dollar amount per BUY order
PAPER_TRADING = True       # Set to False ONLY if using live real-money keys


def send_telegram_message(message: str):
    """Sends a status alert to your Telegram chat."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing. Skipping notification.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")


def get_market_data(symbol: str):
    """Fetches the last 24 hours of hourly price bars from Alpaca."""
    data_client = CryptoHistoricalDataClient()
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(hours=24)

    request_params = CryptoBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Hour,
        start=start_time,
        end=now
    )
    bars = data_client.get_crypto_bars(request_params)
    df = bars.df

    if df.empty:
        return None

    recent_bars = df.tail(12)[['open', 'high', 'low', 'close', 'volume']]
    return recent_bars.to_string()


def get_ai_decision(symbol: str, market_summary: str, cash: float, current_qty: float):
    """Queries Gemini to analyze market data and return a JSON trading decision."""
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""
You are a disciplined quantitative crypto trading assistant.
Analyze the following recent hourly OHLCV data for {symbol}:

{market_summary}

Current Portfolio Status:
- Available Cash: ${cash:.2f}
- Current {symbol} Holdings: {current_qty}

Decide whether to BUY, SELL, or HOLD.
Rules:
- Only output BUY if there is clear upward momentum and Available Cash >= {TRADE_NOTIONAL_USD}.
- Only output SELL if there is clear downward momentum/reversal and Current Holdings > 0.
- Otherwise, output HOLD.

Respond strictly in valid JSON format with two keys:
{{"action": "BUY" | "SELL" | "HOLD", "reason": "Brief 1-sentence explanation"}}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"response_mime_type": "application/json"}
    )

    try:
        return json.loads(response.text)
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        return {"action": "HOLD", "reason": "Failed to parse AI response safely."}


def main():
    # Verify all secrets are present
    missing = [
        name for name, val in [
            ("CRYPTO_TRADING_BOT", ALPACA_API_KEY),
            ("ALPACA_SECRET_KEY", ALPACA_SECRET_KEY),
            ("GEMINI_KEY", GEMINI_API_KEY),
            ("TELEGRAM_TOKEN", TELEGRAM_TOKEN),
            ("TELEGRAM_ID", TELEGRAM_CHAT_ID),
        ] if not val
    ]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    # Initialize Alpaca Trading Client
    trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER_TRADING)
    account = trading_client.get_account()
    cash = float(account.cash)

    # Check current position for BTCUSD (Alpaca removes the slash for position symbols)
    asset_symbol = SYMBOL.replace("/", "")
    current_qty = 0.0
    try:
        position = trading_client.get_open_position(asset_symbol)
        current_qty = float(position.qty)
    except Exception:
        current_qty = 0.0  # No open position

    # Fetch Market Data
    market_summary = get_market_data(SYMBOL)
    if not market_summary:
        send_telegram_message(f"⚠️ *Crypto Bot*: No market data returned for {SYMBOL}.")
        return

    # Get AI Decision
    decision = get_ai_decision(SYMBOL, market_summary, cash, current_qty)
    action = decision.get("action", "HOLD").upper()
    reason = decision.get("reason", "No reason provided.")

    order_status = "No trade executed."

    # Execute Trade if applicable
    if action == "BUY" and cash >= TRADE_NOTIONAL_USD:
        order_req = MarketOrderRequest(
            symbol=SYMBOL,
            notional=TRADE_NOTIONAL_USD,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.IOC
        )
        trading_client.submit_order(order_data=order_req)
        order_status = f"✅ Executed BUY order for ${TRADE_NOTIONAL_USD} of {SYMBOL}."

    elif action == "SELL" and current_qty > 0:
        order_req = MarketOrderRequest(
            symbol=SYMBOL,
            qty=current_qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.IOC
        )
        trading_client.submit_order(order_data=order_req)
        order_status = f"✅ Executed SELL order for {current_qty} {SYMBOL}."

    # Send Summary to Telegram
    report = (
        f"🤖 *Crypto Trading Bot Report*\n"
        f"• *Pair:* `{SYMBOL}`\n"
        f"• *Decision:* *{action}*\n"
        f"• *Reason:* {reason}\n"
        f"• *Action:* {order_status}\n"
        f"• *Cash Balance:* ${cash:.2f}"
    )
    print(report)
    send_telegram_message(report)


if __name__ == "__main__":
    main()
