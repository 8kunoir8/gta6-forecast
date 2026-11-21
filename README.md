# gta6-forecast

Entry for make.fun `gta6-trophy-rarity-forecast-01`: forecast GTA VI trophy earn
rates on Exophase at a target time, as five quantiles per trophy.

## Approach

Count-space structural model, standard library only, deterministic:

- Each history read gives players tracked `P(t)` and per-trophy rate `r_i(t)`;
  earned counts `E_i(t) = r_i(t) * P(t) / 100`.
- `P` follows a logistic adoption curve, profiled over carrying capacity `K`
  and fitted to the read series (cache-refresh plateaus kept at their read
  times, so the fit sees the jump process, not a smoothed lie).
- Each trophy earns as `E_i(t) = f_i * P(t - lag_i)`: registrations convert to
  earns after a playtime delay. `(f_i, lag_i)` fitted per trophy over a grid,
  weighted by count-space SSE; a mixture over the fit's near-optimal members
  carries parameter uncertainty.
- Forecast rate `f_i * P(target - lag_i) / P(target)`; predictive quantiles are
  the weighted mixture of logit-normal components (process noise scaled by the
  observed per-refresh logit move RMS and the horizon) plus a persistence
  component weighted by the probability no cache refresh lands before target,
  inverted by bisection at the five scored levels.
- Stale-flat or single-read histories fall back to tier/description priors
  anchored to the observed adoption level, with wide spread.

## Run

    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

The grader runs `python3 -P -s /bot/main.py`; `main.py` restores its directory
on `sys.path` so `model.py` beside it imports. No network, no dependencies.

## License

MIT (see LICENSE).
