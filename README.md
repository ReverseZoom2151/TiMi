<h1 align="center">TiMi</h1>

<p align="center"><strong>Trade in Minutes: a rationality-driven multi-agent system for quantitative trading</strong></p>

An implementation of the ICLR-submitted paper
[*Trade in Minutes! Rationality-Driven Agentic System for Quantitative Financial
Trading*](https://arxiv.org/abs/2510.04787). Four LLM agents develop and refine
grid-trading strategies offline, and a low-latency bot executes them, so the
expensive reasoning is decoupled from the time-sensitive execution.

This repository is under active correction. It is a research implementation, not
a trading product, and the section below on what is not yet safe is the most
important part of this document.

## Status

The system trades **spot**, on Binance, through ccxt. It does not trade futures:
there is no leverage, no margin and no liquidation in this codebase, and the
sell side of a grid can only sell inventory that is actually held.

Recent work has closed three defects that could each have caused real money to
move without the operator intending it:

- **A testnet setting that did not reach testnet.** The connector overrode two
  ccxt endpoint entries by hand, which left the other endpoint groups pointed at
  production while logging "initialized in TESTNET mode". It now uses
  `set_sandbox_mode`, which moves every group, and this is covered by a test.
- **A paper-trading gate that failed open.** `is_paper_trading()` compared the
  mode string with no validation anywhere, so any value other than exactly
  `paper` transmitted live orders. `backtest` was one of those values. Mode is
  now validated against an allow-list and refuses anything else by name.
- **Safety flags parsed case-sensitively.** `PAPER_TRADING_MODE=True` resolved to
  live, and `BINANCE_TESTNET=True` resolved to production, because both were
  compared with `== 'true'`. Both failed towards danger. Parsing is now
  case-insensitive and an unreadable value falls back towards safety.

## What still requires validation

**Nothing in this repository has been run against an exchange, testnet or
otherwise, since the corrections began.** Verification is a 215 test suite that
executes the logic with the network blocked, plus reading of the ccxt source. No
order has been placed, no fill has been observed, and no strategy has been
evaluated for profitability by anyone working on this code.

The execution engine still carries known defects, and they are the reason live
trading is gated. As of this commit:

- The grid is re-placed on every cycle and resting orders are never cancelled,
  so open orders accumulate against a fixed capital allocation.
- There is no stop loss in the engine. `risk.stop_loss_pct` is configured and
  nothing reads it.
- The risk checks exist and are not called. `check_order_risk`,
  `check_drawdown`, `check_position_risk` and `check_price_deviation` have no
  call sites, so drawdown protection and position limits cannot fire, and the
  emergency stop cannot be raised.
- `PositionManager` and `OrderManager` are never fed, so position and order
  statistics remain empty and the risk layer sees no state.
- Paper mode does not simulate fills, so its statistics are structurally zero. A
  paper run cannot currently demonstrate that a strategy works.

Work on these is in progress. Until it lands, treat this repository as something
to read and test rather than something to run with capital.

The published performance figures in the paper (annual return, Sharpe ratio,
action latency) are the paper's own measurements of the authors' system. Nothing
in this repository has reproduced them, and nothing here measures latency.

## Safety model

Live trading requires two deliberate acts, in different places, plus a
confirmation:

```bash
export TIMI_ALLOW_LIVE=1
python run_timi.py --mode live --i-understand-the-risk --pairs BTC/USDT
```

Missing either the environment variable or the flag refuses to start, and the
check runs before an authenticated exchange client is constructed. The
confirmation prompt then prints the resolved endpoint host, the mode and the
pairs, so the operator confirms against facts rather than a bare question.

Separately, startup asserts that the resolved endpoint matches the requested
mode: any mode other than live must have landed on a testnet host, or the
process refuses to continue. That assertion is what catches a testnet setting
that silently did not take effect.

## Install

```bash
git clone https://github.com/ReverseZoom2151/TiMi.git
cd TiMi

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # then fill in your keys
```

Python 3.9 or later. The `.env` file holds exchange and LLM credentials and is
gitignored; `.env.example` documents every variable, including the live unlock.

## Architecture

The paper separates expensive reasoning from time-sensitive execution across
three stages. Stage I develops strategies offline, stage II refines their
parameters, and stage III executes them with low latency.

Stages I and III exist here and run. Stage II is implemented but never called:
`BotEvolutionAgent` and `FeedbackReflectionAgent` are constructed in `main.py`
and no code path invokes them. The bot code the first of them produces is
parsed for syntax and stored as a string.

**Nothing in this repository executes LLM-generated code**, and there is no path
by which it could: a search for `exec`, `eval`, `compile` and for any write of a
`.py` file returns nothing.

## Configuration

`config.yaml` holds the system configuration and `.env` holds credentials and
the safety flags. Every risk value is expressed as a percentage in the file and
converted to a fraction internally, so `max_drawdown: 20` means 20 percent.

`min_volatility` is the exception worth knowing about: it is a **fraction**, not
a percentage. It shipped as `0.5`, which demanded a 50 percent hourly range and
meant no pair ever qualified, so the system placed no orders while appearing
healthy. It is now `0.005`, which is 0.5 percent.

## Usage

```bash
# Paper trading, the default and the only mode that needs no unlock
python run_timi.py --mode paper --pairs BTC/USDT --duration 1
python run_timi.py --mode paper --pairs BTC/USDT ETH/USDT --duration 24
```

Note that paper mode does not yet simulate fills, so its statistics will be
zero. See the validation section above.

## Technical indicators

Seven are implemented, in `timi/data/indicators.py`: SMA, EMA, RSI, MACD,
Bollinger Bands, ATR and a volume moving average. Each returns NaN rather than a
number for any window the available data cannot satisfy, so a short frame from a
newly listed pair reports "not enough data" instead of a fabricated zero.

The volatility measure used to size the grid is not a standard deviation. It is
a normalised range over opens and closes, it ignores wicks and therefore
understates the true range, and it scales with the lookback window so two
different windows are not comparable. This is documented at the function rather
than corrected, because the grid geometry is calibrated to it.

## Tests

```bash
python -m pytest
```

215 tests, no network access and no credentials required.

The suite cannot reach an exchange account, and that is enforced rather than
assumed. Two autouse fixtures apply to every test: outbound connections to
remote hosts raise and name the offending test, and every exchange and LLM
credential is removed from the environment. Tests that genuinely need the
network must be marked, and that marker is deselected by default. A meta-test
suite asserts the guard rails themselves still work.

Exchange behaviour is tested against a recording double rather than a mock, so
assertions are about exactly what would have been transmitted.

## Repository layout

```text
timi/
  agents/       four LLM agents; two are not currently called
  core/         bot engine, position manager, order manager
  data/         market data with caching, technical indicators
  exchange/     ccxt connector, spot, with a testnet assertion
  llm/          provider clients for OpenAI and Anthropic
  risk/         risk manager and constraints; not yet wired into the order path
  utils/        configuration, mode validation, structured logging
tests/          215 tests, network blocked by default
config.yaml     system configuration
.env.example    credentials and safety flags
```

## Contributing

Run `python -m pytest` before submitting. Keep claims tied to what has been
executed: if a change has only been unit tested, say so, and leave the
validation section above honest.

## License and citation

Released under the [MIT License](LICENSE). If this implementation informs
research, cite the original paper:

```bibtex
@article{song2025timi,
  title={Trade in Minutes! Rationality-Driven Agentic System for Quantitative Financial Trading},
  author={Song, Zifan and Song, Kaitao and Hu, Guosheng and Qi, Ding and Gao, Junyao and Wang, Xiaohua and Li, Dongsheng and Zhao, Cairong},
  journal={arXiv preprint arXiv:2510.04787},
  year={2025}
}
```

Research conducted at Tongji University, Microsoft Research Asia, the University
of Bristol and Fudan University.

## Disclaimer

Trading involves substantial risk of loss. This software is for educational and
research purposes and is not financial advice. The authors are not registered
investment advisers and make no recommendation regarding any security or
strategy. You are responsible for compliance with the regulations that apply to
you.

Only trade with capital you can afford to lose completely. Given the open
defects listed above, the honest recommendation today is not to trade with this
at all.
