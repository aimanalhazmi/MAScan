"""Economics tool adapters backed by shared tool functions."""

from typing import Any, ClassVar

from pydantic import BaseModel, Field

from mascan.contracts.tools import ToolResult
from mascan.tools.base import BaseTool
from mascan.tools.fin_api import get_weekly_stock_prices

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
            raw_result = get_weekly_stock_prices.invoke(
                {
                    "ticker": ticker,
                    "start_date": start_date,
                    "end_date": end_date,
                }
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
