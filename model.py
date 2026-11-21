"""Count-space lag forecaster for make.fun gta6-trophy-rarity-forecast-01.

Model: earned counts E_i(t) = f_i * P(t - lag_i); players P a logistic curve
profiled over carrying capacity K. Rate forecast = f_i * P(target-lag_i)/P(target).
Quantiles: weighted mixture over (K, lag, f) fit uncertainty + horizon-scaled
process noise, CDF inverted by bisection. Stdlib only, deterministic.
"""
import math
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

LAUNCH = datetime(2026, 11, 19, tzinfo=timezone.utc)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
K_GRID = (60e3, 90e3, 130e3, 180e3, 250e3, 350e3, 500e3, 750e3)
LAG_GRID = tuple(range(0, 73, 4))
MAX_READS = 120


def _hours(at):
    dt = datetime.fromisoformat(at.replace("Z", "+00:00"))
    return (dt - LAUNCH).total_seconds() / 3600.0


def _logit_pct(p):
    p = min(99.99, max(0.01, p)) / 100.0
    return math.log(p / (1 - p))


def _sig(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _p_at(m, t):
    K, a, b = m[0], m[1], m[2]
    return K / (1.0 + math.exp(-min(60.0, max(-60.0, a * t + b))))


def _prep(case):
    reads = [(_hours(h["at"]), float(h["players_tracked"]),
              [None if r is None else float(r) for r in h["rates"]])
             for h in case["history"]]
    k = 0
    while k + 1 < len(reads) and reads[k + 1][1:] == reads[k][1:]:
        k += 1
    changed = k + 1 < len(reads)
    if changed and k >= 1:
        reads = reads[k + 1:]  # drop stale pre-refresh cache prefix
    if len(reads) > MAX_READS:
        step = math.ceil(len(reads) / MAX_READS)
        last = reads[-1]
        reads = reads[::step]
        if reads[-1] is not last:
            reads.append(last)
    return reads, not changed


def _fit_p(reads):
    ts = [r[0] for r in reads]
    ps = [r[1] for r in reads]
    n = len(ts)
    mt = sum(ts) / n
    members = []
    for K in K_GRID:
        ys = [math.log(min(0.999, max(1e-6, p / K)) / (1 - min(0.999, max(1e-6, p / K))))
              for p in ps]
        my = sum(ys) / n
        sxx = sum((t - mt) ** 2 for t in ts)
        if sxx < 1e-9:
            a, b = 0.0, my
        else:
            a = sum((t - mt) * (y - my) for t, y in zip(ts, ys)) / sxx
            b = my - a * mt
        sse = sum((y - (a * t + b)) ** 2 for t, y in zip(ts, ys))
        members.append((K, a, b, sse))
    smin = min(m[3] for m in members)
    scale = max(smin / max(n - 2, 1), 1e-3)
    ws = [math.exp(-(m[3] - smin) / (2 * scale)) for m in members]
    tot = sum(ws)
    return [(m[0], m[1], m[2], w / tot) for m, w in zip(members, ws)]


def _fit_trophy(reads, idx, pmembers):
    obs = [(t, P, r[idx] * P / 100.0) for t, P, r in reads if r[idx] is not None]
    if not obs:
        return None
    members = []
    for m in pmembers:
        fits = {}
        for lag in LAG_GRID:
            pp = [_p_at(m, t - lag) for t, P, E in obs]
            ws = [1.0 / (E + 0.0005 * P + 1.0) for t, P, E in obs]
            den = sum(w * p * p for w, p in zip(ws, pp))
            f = min(1.0, max(0.0, sum(w * E * p for w, (t, P, E), p in zip(ws, obs, pp)) / den)) if den > 0 else 0.0
            sse = sum(w * (E - f * p) ** 2 for w, (t, P, E), p in zip(ws, obs, pp))
            fits[lag] = (sse, f, den, ws, pp)
        smin = min(v[0] for v in fits.values())
        # noise floor: rate quantization (0.01pp) plus 5% relative profile band
        tol = max(sum((P * 1e-4) ** 2 * w for (t, P, E), w in zip(obs, fits[0][3])) / len(obs),
                  0.05 * smin)
        keep = sorted(lag for lag, v in fits.items() if v[0] <= smin + tol)
        best = min(keep, key=lambda lag: fits[lag][0])
        sse_b, f_b, den_b, ws_b, pp_b = fits[best]
        se = math.sqrt(sse_b / (max(len(obs) - 1, 1) * den_b)) if den_b > 0 else 0.0
        jit = max(se, 0.005 * f_b)
        wl = 0.8 / len(keep)
        for lag in keep:
            members.append((m[3] * wl, m, lag, fits[lag][1]))
        members.append((m[3] * 0.1, m, best, max(0.0, f_b - jit)))
        members.append((m[3] * 0.1, m, best, min(1.0, f_b + jit)))
    return members


def _prior_members(tr, reads):
    tier = tr.get("tier", "bronze")
    story = "Chapter" in (tr.get("description") or "")
    table = {("bronze", True): (0.30, 10), ("silver", True): (0.18, 24),
             ("gold", True): (0.10, 40), ("platinum", True): (0.03, 60),
             ("bronze", False): (0.04, 36), ("silver", False): (0.06, 40),
             ("gold", False): (0.04, 50), ("platinum", False): (0.02, 60)}
    f, lag = table.get((tier, story), (0.05, 36))
    members = []
    curves = []
    if reads and reads[-1][0] > 1.0 and reads[-1][1] > 700:
        t_last, p_last = reads[-1][0], reads[-1][1]
        for K in (80e3, 150e3, 300e3, 600e3):
            if p_last >= K * 0.999:
                continue
            b = math.log((638.0 / K) / (1 - 638.0 / K))
            y = math.log((p_last / K) / (1 - p_last / K))
            curves.append((K, (y - b) / t_last, b, 0.25))
    if not curves:
        for K in (80e3, 150e3, 300e3):
            b = math.log((638.0 / K) / (1 - 638.0 / K))  # P(0) = launch-night tracked count
            for a in (0.08, 0.12, 0.18):
                curves.append((K, a, b, 1.0 / 9))
    for m in curves:
        for fj, lj, w in ((f, lag, 0.5), (f * 0.5, lag * 0.6, 0.25),
                          (min(1.0, f * 1.6), lag * 1.5, 0.25)):
            members.append((m[3] * w, m, lj, fj))
    return members


def _jump_rms(reads):
    """RMS logit move per cache refresh, the process-noise scale."""
    moves = []
    for k in range(len(reads) - 1):
        if reads[k + 1][1:] == reads[k][1:]:
            continue
        for a, b in zip(reads[k][2], reads[k + 1][2]):
            if a is not None and b is not None:
                moves.append((_logit_pct(b) - _logit_pct(a)) ** 2)
    return min(0.5, max(0.02, math.sqrt(sum(moves) / len(moves)))) if moves else 0.05


def _refresh_ttl(reads):
    """Mean cache-refresh interval, shrunk toward a 3 h prior (weight 2)."""
    gaps = [reads[k + 1][0] - reads[k][0] for k in range(len(reads) - 1)
            if reads[k + 1][1:] != reads[k][1:]]
    return (6.0 + sum(gaps)) / (2 + len(gaps))


def _mix_quantile(centers, weights, sigmas, tau):
    lo, hi = -14.0, 14.0
    for _ in range(48):
        mid = (lo + hi) / 2
        c = sum(w * _sig((mid - c0) / s) for c0, w, s in zip(centers, weights, sigmas))
        if c < tau:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def forecast_case(case):
    reads, flat = _prep(case)
    target = _hours(case["target"])
    d = case["horizon_minutes"] / 1440.0
    pm = None if flat or len(reads) < 2 else _fit_p(reads)
    h_last = target - reads[-1][0] if reads else d * 24.0
    p_jump = 1.0 - math.exp(-max(h_last, 0.0) / _refresh_ttl(reads))
    s_jump = _jump_rms(reads)
    sigma_g = math.sqrt(s_jump * s_jump * max(h_last, 0.0) / _refresh_ttl(reads) + 0.0064 * d)
    rows = []
    for i, tr in enumerate(case["trophies"]):
        fit = None if pm is None else _fit_trophy(reads, i, pm)
        if fit is None:
            members = _prior_members(tr, reads)
            r_last = reads[-1][2][i] if reads else None
            if r_last is not None:
                t_last = reads[-1][0]
                members = [(w, m, lag, min(1.0, (r_last / 100.0) * _p_at(m, t_last)
                                           / max(_p_at(m, t_last - lag), 1e-9)))
                           for w, m, lag, f in members]
            sigma = 0.9 * math.sqrt(d) + 0.6 if flat else sigma_g + 0.3
        else:
            members = fit
            sigma = sigma_g + (0.15 if len(reads) < 6 else 0.0)
        centers, weights, sigmas = [], [], []
        for w, m, lag, f in members:
            rate = 100.0 * f * _p_at(m, target - lag) / _p_at(m, target)
            centers.append(_logit_pct(rate))
            weights.append(w)
            sigmas.append(sigma)
        tot = sum(weights)
        weights = [w * p_jump / tot for w in weights]
        r_last = reads[-1][2][i] if reads else None
        if r_last is not None and p_jump < 1.0:
            centers.append(_logit_pct(r_last))
            weights.append(1.0 - p_jump)
            sigmas.append(0.01)
        prev = 0.0
        qs = []
        for tau in LEVELS:
            x = _mix_quantile(centers, weights, sigmas, tau)
            p = min(100.0, max(0.0, 100.0 / (1.0 + math.exp(-min(60.0, max(-60.0, x))))))
            p = float(Decimal(str(max(p, prev))).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))
            prev = p
            qs.append(p)
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}