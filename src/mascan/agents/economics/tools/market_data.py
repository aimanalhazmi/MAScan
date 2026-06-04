"""Economics tools"""

import yfinance as yf

from typing import Any, ClassVar

from pydantic import BaseModel, Field

from mascan.contracts.tools import ToolResult
from mascan.tools.base import BaseTool

class WebQueryInput(BaseModel):
    query: str = Field(..., description="Search query for economic or market context.")


class WeeklyStockPricesInput(BaseModel):
    ticker: str = Field(..., description="Yahoo Finance ticker, e.g. BMW.DE.")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format.")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format.")


class WeeklyStockPricesTool(BaseTool):
    name = "get_weekly_stock_prices"
    description = (
        "Get weekly stock prices and compact fundamentals from Yahoo Finance. "
        "Use this for a public company, stock ticker, stock performance, valuation, "
        "or company-specific equity-market impact question."
    )
    input_schema: ClassVar[type[BaseModel] | None] = WeeklyStockPricesInput

    def run(self, ticker: str, start_date: str, end_date: str, **_: Any) -> ToolResult[Any]:
        try:
            raw_result = self.get_stock_prices(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date
            )
            if isinstance(raw_result, ToolResult):
                return raw_result
            if isinstance(raw_result, str):
                return ToolResult[Any].model_validate_json(raw_result)
            return ToolResult(
                success=True,
                data=raw_result,
                source=f"yfinance:{ticker}",
                metadata={
                    "provider": "yfinance",
                    "interval": "1wk",
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )
        except Exception as exc:
            self.logger.exception("get_weekly_stock_prices failed for ticker=%r", ticker)
            return ToolResult(
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
        
    def get_stock_prices(self, ticker: str, start_date: str, end_date:str) -> str:
        """Fetches weekly stock prices and fundamentals from Yahoo Finance.

        Args:
            ticker (str): The stock ticker symbol.
            start_date (str): The start date in YYYY-MM-DD format.
            end_date (str): The end date in YYYY-MM-DD format.

        Returns:
            str: A JSON string containing the stock price data.
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
                        "open": row["Open"],
                        "high": row["High"],
                        "low": row["Low"],
                        "close": row["Close"],
                        "volume": row["Volume"],
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

            result =  ToolResult(
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