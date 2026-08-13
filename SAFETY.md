# Safety

Operator-facing notes for anyone about to run TiMi. Read this before you run
anything, not after.

## 1. The one-paragraph version

This software can place real orders on a real exchange with real money, and
those orders cannot be recalled once they are accepted. The defaults are safe:
with no environment file and no flags, the mode is `paper`, the exchange client
is pointed at testnet, and no order is transmitted. Every path that leads to
real money requires a deliberate act by the operator, in more than one place,
and none of those acts happens by default or by accident. Nothing in this
repository has been run against an exchange, so treat the safety machinery
described below as designed and unit tested rather than as proven in
production. Section 7 says exactly what that means.

## 2. The two-part live unlock

Live trading needs two separate deliberate acts. Neither one alone is enough,
and the check runs in `timi/main.py` **before** an authenticated exchange
client is constructed, so a live run cannot get as far as holding credentials
by mistake.

1. The environment variable `TIMI_ALLOW_LIVE=1`. Exactly `1`, after stripping
   whitespace: `true`, `yes` and `TRUE` do not unlock it.
2. The command line flag `--i-understand-the-risk`.

The exact command:

```bash
export TIMI_ALLOW_LIVE=1
python run_timi.py --mode live --i-understand-the-risk --pairs BTC/USDT
```

Missing either one exits with status 2 and names what is missing
(`assert_live_trading_unlocked` in `timi/utils/config.py`). If both are
present, a third barrier follows: the process prints the resolved mode, the
resolved endpoint host and the pairs, then waits for you to type `yes`. That
prompt exists so you confirm against displayed facts rather than against your
memory of what you meant to configure. Anything other than `yes` cancels the
run.

Two separate acts, in two different places, is the point. A copied command line
does not carry your environment, and an exported environment variable does not
carry a flag. Reaching live requires both.

## 3. Modes

There are exactly three modes, and they are an allow-list in
`timi/utils/config.py`:

| Mode | Transmits orders | Unlock required |
| --- | --- | --- |
| `backtest` | No | None |
| `paper` | No | None |
| `live` | **Yes** | Both parts of section 2 |

`paper` is the default when no mode is configured. Only `live` transmits: every
other mode routes through the simulated path, and `is_paper_trading()` answers
"is this not live?" rather than "is the mode exactly paper?".

Anything outside the allow-list is refused by name, with the allowed values
listed in the error. The comparison is exact: no case folding, no whitespace
stripping, no aliasing. `Paper`, `paper_trading` and `live ` are all
configuration errors, not near-misses to be guessed at. Guessing what an
operator meant is how real orders get placed by accident.

History worth knowing: `is_paper_trading()` once compared the mode string with
no validation anywhere, so any value other than exactly `paper` transmitted
live orders, and `backtest` was one of those values. That is why the allow-list
now exists and why a typo is a hard failure.

## 4. Boolean environment flags

`PAPER_TRADING_MODE`, `BINANCE_TESTNET` and `EMERGENCY_STOP` are parsed by
`parse_bool_env` in `timi/utils/config.py`.

- Case-insensitive, with surrounding whitespace ignored.
- True: `true`, `1`, `yes`, `on`. False: `false`, `0`, `no`, `off`.
- Unset or empty means the documented default.
- A value that cannot be read at all resolves towards **safety** (paper
  trading, testnet, emergency stop engaged) and logs a warning naming the
  variable and the value. An unreadable safety flag is an unknown state, never
  a disabled one. Fix the value: do not leave the warning in place, because
  what it resolved to is not what you asked for.

History worth knowing: these flags were once compared with `== 'true'`. That
meant `PAPER_TRADING_MODE=True`, which is how most people write a boolean, did
not match, and so resolved to **LIVE**. `BINANCE_TESTNET=True` resolved to
production the same way. Both failed towards danger rather than away from it.
If you have an old `.env` written when that was true, re-read every boolean in
it before you run anything.

## 5. Testnet

The connector calls ccxt's `set_sandbox_mode`, which moves **every** endpoint
group on the client at once, rather than overwriting individual URLs.

On top of that, startup calls `assert_mode_matches_endpoint`, which reads the
endpoint the constructed client would actually call and checks it against the
requested mode. Any mode other than `live` must have resolved to a testnet or
sandbox host, or the process refuses to continue and exits with status 2. This
runs before any order path is reachable, and it is the check that catches a
testnet setting which silently did not take effect. If the endpoint cannot be
read at all, that is also a refusal: an endpoint that cannot be verified is not
a safe one.

History worth knowing: the connector previously overrode two entries under
`urls.api` by hand. Every other endpoint group was left pointing at
production, while the log line cheerfully said "initialized in TESTNET mode".
No amount of reviewing the configuration would have found that, which is why
the assertion checks the resolved endpoint rather than trusting the flag.

## 6. Pre-flight checklist before any live run

Work through this in order. Do not skip a step because you did it last time.

1. **Run the test suite on the exact commit you intend to run.**
   `python -m pytest` must be fully green. It needs no credentials and no
   network.
2. **Read section 7 and the "What still requires validation" section of the
   README.** If the open defects listed there are still open, stop here.
3. **Confirm the code you are running is the code you reviewed.** Check the
   commit, and check that your working tree has no uncommitted changes.
4. **Read your `.env` line by line.** Confirm every boolean is a value from
   section 4, and confirm the file is the one you think it is.
5. **Confirm the credentials are the ones you intend.** Use an API key scoped
   to spot trading only, with withdrawals disabled and, if the exchange
   supports it, an IP allow-list.
6. **Do a testnet run first,** with `BINANCE_TESTNET=true` and testnet keys,
   and watch it to completion. Confirm from the logs that the endpoint
   assertion passed and which host it reported.
7. **Set the capital limits deliberately.** `MAX_TOTAL_CAPITAL`,
   `MAX_POSITION_SIZE` and `strategy.capital_per_pair` in `config.yaml` should
   be an amount you would be willing to lose in full, not a placeholder.
8. **Confirm `EMERGENCY_STOP=false`** is what you want, and know how to set it
   to `true` (section 8) before you need to.
9. **Start with one pair and a short duration.** `--pairs BTC/USDT --duration
   1`, not a basket for a day.
10. **Read the confirmation prompt rather than typing through it.** Check the
    mode, the endpoint host and the pairs it prints against what you intended.
    If any of the three surprises you, answer anything other than `yes`.
11. **Watch the run.** Do not start a live run you are not going to sit with,
    and do not leave one running unattended overnight.
12. **Know your exchange's order-cancellation screen before you start,** not
    while you need it.

## 7. What is still unverified

**Nothing in this repository has been run against an exchange, testnet or
otherwise.** No order has been placed. No fill has been observed. No strategy
has been evaluated for profitability by anyone working on this code, and no
latency has been measured.

Everything asserted in this document is supported by unit tests that execute
the logic with the network blocked, and by reading the ccxt source. That is
real evidence and it is not the same as evidence from a live venue. In
particular, the endpoint assertion in section 5 has been verified against a
recording test double that mirrors ccxt's shape, not against a ccxt client
talking to Binance.

The engine defects that made earlier versions unsafe have since been fixed:
the grid is diffed and its resting orders cancelled rather than re-placed
every cycle, there is a stop loss driven by the configured percentage, the
risk checks are called from the order path, and paper mode simulates fills so
its statistics are real. Each of those is covered by tests.

What has not changed is that the fixes themselves are unproven against a
venue. A test proves the logic does what the author intended; it cannot prove
the intention matched how Binance behaves, how orders queue and fill, what
partial fills look like, or how the account is configured. The paper-mode
simulation in particular is optimistic by construction: no queue position, no
fees, no slippage, and it says so in its own docstring.

Given that, the honest recommendation today is still to treat a first live run
as an experiment with money you can afford to lose, run at the smallest size
the exchange will accept, and watched.

## 8. Incident response

If something is going wrong, stop the flow of new orders first, then deal with
what is already resting on the exchange. These are in order of speed.

1. **Interrupt the process.** `Ctrl+C` in the terminal running it. The
   `KeyboardInterrupt` path logs and shuts down, including closing the exchange
   client. This stops new orders being placed.
2. **Kill it if it does not stop.** Do not wait politely for a process that is
   placing orders.
   - Linux or macOS: `pkill -f run_timi.py`, then `pkill -9 -f run_timi.py` if
     it is still alive.
   - Windows PowerShell:
     `Get-Process python | Where-Object { $_.CommandLine -like '*run_timi*' } | Stop-Process -Force`
3. **Set the emergency stop before anything is restarted.** In `.env`:

   ```bash
   EMERGENCY_STOP=true
   ```

   The risk manager reads this from configuration when it is constructed, and
   the engine checks it every cycle and halts when it is set. Because the value
   is read at construction, setting it in `.env` takes effect on the next
   start-up rather than on a process already running: it prevents the next run,
   it does not interrupt the current one. Stopping the process (steps 1 and 2)
   is what stops a run that is already going.
4. **Check what is still resting on the exchange.** A graceful shutdown
   cancels the bot's resting orders: `stop()` calls `_cancel_all_orders()`
   before it returns, and the same happens when the failure counter halts a
   bot. A hard kill does not, because nothing runs after `SIGKILL`. So if you
   used `Ctrl+C` and the process exited cleanly, the orders should already be
   gone; if you used `pkill -9`, or the process died some other way, assume
   every open order is still live on the venue and can still fill.

   Either way, verify rather than assume. Log in to the exchange, open the
   open-orders view, cancel anything left, per pair, and then reconcile your
   balances against what you expected. The bot's own record of what it placed
   is not authoritative after an abnormal exit.
5. **Revoke the API key** if you suspect the credentials themselves are the
   problem, or if a key may have been committed or shared. Revoke at the
   exchange first. Removing a key from a file does not un-leak it.
6. **Write down what happened while it is fresh,** including the exact command
   line, the commit, the contents of `.env` at the time (with the secrets
   redacted) and the log file from `logs/`. Then fix the cause before the next
   run, rather than restarting and hoping.
