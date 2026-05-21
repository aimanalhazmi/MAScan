import yfinance as yf
from langchain_core.tools import tool
from mascan.contracts.tools import ToolResult



@tool
def get_weekly_stock_prices(ticker: str, start_date: str, end_date: str) -> str:
    """
    Get weekly stock prices and compact fundamental data for a ticker.

    Args:
        ticker: Yahoo Finance ticker, e.g. BMW.DE.
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        history = stock.history(
            start=start_date,
            end=end_date,
            interval="1wk",
        )

        weekly_prices = []

        for index, row in history.iterrows():
            weekly_prices.append(
                {
                    "date": index.date().isoformat(),
                    "open": row.get("Open"),
                    "high": row.get("High"),
                    "low": row.get("Low"),
                    "close": row.get("Close"),
                    "volume": row.get("Volume"),
                }
            )

        data = {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "fundamentals": {
                "company_name": info.get("longName"),
                "currency": info.get("currency"),
                "exchange": info.get("exchange"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "current_price": info.get("currentPrice"),
                "total_revenue": info.get("totalRevenue"),
                "gross_profit": info.get("grossProfits"),
                "operating_income": info.get("operatingIncome"),
                "net_income": info.get("netIncomeToCommon"),
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "revenue_growth": info.get("revenueGrowth"),
                "debt_to_equity": info.get("debtToEquity"),
            },
            "weekly_prices": weekly_prices,
        }

        result = ToolResult(
            success=True,
            data=data,
            source=f"yfinance:{ticker}",
            metadata={
                "provider": "yfinance",
                "interval": "1wk",
                "price_points": len(weekly_prices),
            },
        )

    except Exception as exc:
        result = ToolResult(
            success=False,
            data=None,
            source=f"yfinance:{ticker}",
            error=str(exc),
            metadata={
                "provider": "yfinance",
                "interval": "1wk",
                "start_date": start_date,
                "end_date": end_date,
            },
        )

    return result.model_dump_json()