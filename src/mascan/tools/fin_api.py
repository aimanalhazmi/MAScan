import yfinance as yf
from langchain_core.tools import tool


@tool
def get_weekly_stock_prices(ticker: str, start_date: str, end_date: str) -> str:
    """
    Get weekly stock closing prices for a ticker between two dates.

    Args:
        ticker: Yahoo Finance ticker, e.g. BMW.DE.
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
    """
    stock = yf.Ticker(ticker)

    history = stock.history(
        start=start_date,
        end=end_date,
        interval="1wk",
    )

    if history.empty:
        return f"No weekly price data found for {ticker} from {start_date} to {end_date}."

    lines = [
        f"Weekly stock prices for {ticker} from {start_date} to {end_date}:"
    ]

    for index, row in history.iterrows():
        date = index.date().isoformat()
        close = row["Close"]
        lines.append(f"- {date}: close={close:.2f}")

    result = "\n".join(lines)

    print(result)
    return result


if __name__ == "__main__":
    output = get_weekly_stock_prices.invoke(
        {
            "ticker": "BMW.DE",
            "start_date": "2025-01-01",
            "end_date": "2025-04-30",
        }
    )

    print("Tool returned:")
    print(output)