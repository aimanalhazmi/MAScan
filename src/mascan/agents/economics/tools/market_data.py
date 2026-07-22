"""Economics tools"""

from datetime import date, timedelta
from typing import Any, ClassVar

import yfinance as yf
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
    MAX_HISTORY_DAYS: ClassVar[int] = 730
    MAX_WEEKLY_ROWS: ClassVar[int] = 104

    def run(self, ticker: str, start_date: str, end_date: str, **_: Any) -> ToolResult[Any]:
        try:
            bounded_start, bounded_end, date_limit_applied = self._bounded_date_range(
                start_date,
                end_date,
            )
            raw_result = self.get_stock_prices(
                ticker=ticker,
                start_date=bounded_start,
                end_date=bounded_end,
            )
            if isinstance(raw_result, ToolResult):
                result = raw_result
            elif isinstance(raw_result, str):
                result = ToolResult[Any].model_validate_json(raw_result)
            else:
                result = ToolResult(
                    success=True,
                    data=raw_result,
                    source=f"yfinance:{ticker}",
                    metadata={
                        "provider": "yfinance",
                        "interval": "1wk",
                        "start_date": bounded_start,
                        "end_date": bounded_end,
                    },
                )
            return self._limit_result(
                result,
                start_date=bounded_start,
                end_date=bounded_end,
                date_limit_applied=date_limit_applied,
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

    @classmethod
    def _bounded_date_range(
        cls,
        start_date: str,
        end_date: str,
    ) -> tuple[str, str, bool]:
        try:
            start = date.fromisoformat(start_date)
            end = date.fromisoformat(end_date)
        except ValueError:
            return start_date, end_date, False
        if (end - start).days <= cls.MAX_HISTORY_DAYS:
            return start_date, end_date, False
        return (
            (end - timedelta(days=cls.MAX_HISTORY_DAYS)).isoformat(),
            end_date,
            True,
        )

    @classmethod
    def _limit_result(
        cls,
        result: ToolResult[Any],
        *,
        start_date: str,
        end_date: str,
        date_limit_applied: bool,
    ) -> ToolResult[Any]:
        metadata = {
            **result.metadata,
            "limit_applied": date_limit_applied,
        }
        if not result.success or not isinstance(result.data, dict):
            return result.model_copy(update={"metadata": metadata})

        data = dict(result.data)
        data["start_date"] = start_date
        data["end_date"] = end_date
        prices = data.get("weekly_prices")
        if isinstance(prices, list):
            rows_limited = len(prices) > cls.MAX_WEEKLY_ROWS
            data["weekly_prices"] = prices[-cls.MAX_WEEKLY_ROWS :]
            metadata["price_points"] = len(data["weekly_prices"])
            metadata["limit_applied"] = date_limit_applied or rows_limited
        return result.model_copy(update={"data": data, "metadata": metadata})

    def get_stock_prices(self, ticker: str, start_date: str, end_date: str) -> str:
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

            # yfinance does not raise for an unknown/delisted ticker — it logs a
            # warning and returns an empty frame. Treat that as a failure so the
            # agent does not cite a phantom source with no underlying data.
            if history.empty:
                return ToolResult(
                    success=False,
                    data=None,
                    source=f"yfinance:{ticker}",
                    error=(
                        f"No price data for {ticker!r} between {start_date} and "
                        f"{end_date} (ticker may be delisted, renamed, or invalid)."
                    ),
                    metadata={
                        "provider": "yfinance",
                        "interval": "1wk",
                        "start_date": start_date,
                        "end_date": end_date,
                        "price_points": 0,
                    },
                ).model_dump_json()

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
                "sources": [
                    {
                        "name": f"Yahoo Finance price history: {ticker}",
                        "category": "prices",
                        "url": f"https://finance.yahoo.com/quote/{ticker}/history",
                    },
                    {
                        "name": f"Yahoo Finance company summary: {ticker}",
                        "category": "summary",
                        "url": f"https://finance.yahoo.com/quote/{ticker}",
                    },
                    {
                        "name": f"Yahoo Finance financials: {ticker}",
                        "category": "financials",
                        "url": f"https://finance.yahoo.com/quote/{ticker}/financials",
                    },
                    {
                        "name": f"Yahoo Finance statistics: {ticker}",
                        "category": "statistics",
                        "url": f"https://finance.yahoo.com/quote/{ticker}/key-statistics",
                    },
                ],
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

            result: ToolResult[Any] = ToolResult(
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
