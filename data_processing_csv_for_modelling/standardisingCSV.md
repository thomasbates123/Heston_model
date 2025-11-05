Methodology Note — Quote Cleaning and Basic No-Arbitrage Check(Step 1)

We validate raw option quotes before computing implied volatilities.

Inputs per row:
date, exdate, cp_flag∈{C,P}, strike_price (K), best_bid, best_offer, impl_volatility (vendor), mid (vendor), T (years),
S(spot)

Key assumption at this step: we do not yet use discount curves or dividends. Bounds are checked in spot space with a conservative convention (see below). A forward-space pass comes in Step 2.

Requirement for checks:


Tradable upper bounds (check on the BID):
Calls: bid ≤ 𝑆
Puts: bid ≤ 𝐾
Rationale: you can short an option for the bid and instantly hedge with the underlying/strike; if bid exceeded the bound, you could lock a profit.


Intrinsic lower bound:

American options (exercise allowed now):
    ask≥max(0,S−K) for calls,
    ask≥max(0,K−S) for puts.

Rationale: the option holder can always exercise immediately; the seller must at least be compensated for that payoff. We check this against the ask because that’s what you’d pay to acquire the right.

European caveat: European options can legitimately trade below intrinsic because you cannot exercise now. The theoretical lower bounds are

$$C >= Se^{-qT} - Ke^{-rT}$$
$$P ≥ Ke^{−rT} - Se^{−qT}$$

We therefore treat intrinsic as a conservative diagnostic in Step 1 and re-check the true lower bounds in forward space in Step 2.

Adaptive Espilon notes:

For the intrinsic lower bound check
    1.Price discretization (ticks)
    2.Quote uncertainty (spread)
    3.Underlying slippage during latency
Solution:
    1.Let tick_opt = option price tick (infer if unknown).
    2.Let spread = ask - bid.
    3.Let sigma = per-row IV (fallback to expiry ATM/median if NaN).
    4.Let dt_lat = “latency window” in years (e.g., 1 second ≈ 1/(252*23400)).
![alt text](image.png)

2:Cross-strike static no-arbitrage (per expiry, diagnostic)



