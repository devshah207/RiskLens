# Beginner Financial Glossary

| Term | Plain-language meaning | RiskLens example |
| --- | --- | --- |
| Stock/share | A unit of ownership in a company | One listed company's share |
| Index | A measure representing a selected set of securities | NIFTY 50 price index |
| Index level | The numeric value of the index, not a stock's share price | NIFTY 50 closing level |
| Trading session | A day/period when the exchange is open for trading | Next **trading** session differs from next calendar day |
| OHLC | Open, high, low, close prices/levels for one session | Market-data CSV columns |
| Close | Reference level at the end of a trading session | Feature cutoff after session close |
| Simple return | Relative change: `P_t / P_(t-1) - 1` | From 25,000 to 24,750 = -1% |
| Volatility | Dispersion of returns, commonly measured by their standard deviation | Rolling historical standard deviation |
| Downside risk | Risk of unfavorable outcomes | A next-session decline of 1% or more |
| Market risk | Risk from market-price movements | Index fall |
| Credit risk | Risk of borrower/counterparty failure to meet obligations | Out of scope initially |
| Operational risk | Risk from failures of systems, people, processes, or external events | Out of scope initially |
| Probability estimate | A number from 0 to 1 representing modeled event likelihood | 0.18 means an estimated 18% probability |
| VaR (Value at Risk) | A loss quantile for a chosen confidence level and horizon, subject to assumptions | Separate later risk estimate, not the classification probability |
| Model validation | Independent checks of model reliability, assumptions, performance and limitations | Chronological tests and calibration |
| Look-ahead leakage | Using information unavailable at prediction time | Tomorrow's close as today's input |
| Backtesting | Comparing past model forecasts with subsequently observed outcomes | Later VaR exception checks |

The project is educational; this glossary is not financial advice.
