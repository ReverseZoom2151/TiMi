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
otherwise, since the corrections began.** Verification is a 419 test suite that
executes the logic with the network blocked, plus reading of the ccxt source. No
order has been placed, no fill has been observed, and no strategy has been
evaluated for profitability by anyone working on this code.

The execution defects that made earlier versions unsafe have been fixed and
are covered by tests: the grid is diffed and its resting orders cancelled
rather than re-placed every cycle, there is a stop loss driven by the
configured percentage, the risk checks are called before every order, the
position book is fed from fills and is the source of cost basis, and paper
mode simulates fills so its statistics are real.

What has not changed is that none of it is proven against a venue. A test
shows the code does what its author intended; it cannot show that the
intention matches how the exchange behaves, how orders queue and fill, what a
partial fill does, or how the account is configured. The paper-mode
simulation is optimistic by construction, with no queue position, no fees and
no slippage, and says so in its own docstring.

Specific things only a real run can settle: whether the volatility measure
produces a sensible grid on live prices, whether the turn and rediscover
timings suit real fill latency, whether the exchange's precision and minimum
notional rules match what the connector rounds to, and whether the strategy
makes money, which nothing here has ever tested.

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

All three stages run. Stage II executes once before deployment rather than in
a loop: bot evolution produces its artefact, and feedback reflection refines
parameters only when there is feedback to reflect on, so a cold start does not
pay for a call that cannot change anything.

Nothing a model returns is trusted. JSON is extracted by scanning for balanced
braces, so prose containing a brace cannot swallow the payload and a reply
truncated at the token limit yields nothing rather than a lucky decode. Every
numeric is then checked for type, finiteness and range before use, and the risk
layer independently refuses an allocation outside its configured band, because
it knows the account size the agents do not.

The bot code stage II produces is parsed for syntax and stored as a string.

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

Paper mode simulates fills, so its statistics are real. The simulation is
optimistic: it assumes no queue position, no fees and no slippage, so treat a
profitable paper run as a floor on how hard live trading is, not a forecast.

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

419 tests, no network access and no credentials required.

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
  risk/         risk manager and constraints, checked before every order
  utils/        configuration, mode validation, structured logging
tests/          419 tests, network blocked by default
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
