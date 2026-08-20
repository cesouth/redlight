# Directional speeds — feature request

**Status:** awaiting decision (build for 0.7 / build later / decline)
**Raised:** 2026-08-20, out of the v0.6.0 ship review
**Requested by:** the customer-facing question "why does every road show one
speed when the two directions are obviously different at rush hour?"

## What is being asked for

Per-direction speeds. On a commuter corridor at 08:00, inbound might run at
4 m/s while outbound runs at 15 m/s. Today the package reports one number for
the road, and it is not either of those — it is a value no vehicle experienced.

Concretely, the customer wants `to_geojson(directional=True)` to mean something,
and `congestion_report` to be able to say a road is congested *one way*.

## Why it does not work today

Not an oversight, and not fixable at the export layer. Three things stack up.

**1. Every observation is attributed to both directions.**
`speeds._hop_distance:237-239` ends with:

```python
# attribute to both directions of every traversed road (one-way roads have one)
eids = set(network.road_edge_ids(edge_a))
eids |= set(network.road_edge_ids(edge_b))
```

So the two directed edges of a two-way road carry byte-identical
`edge_observations`. Measured on `examples/sample_data`: **31 two-way roads,
every one showing a gap of exactly 0.0 mph between directions, with identical
sample counts** (23/23, 34/34, 76/76…).

**2. `to_geojson(directional=False)` therefore loses nothing.**
`mapping._collect_edges` averages the two directions — but the inputs are always
equal, so the mean is a no-op. `directional=True` today yields two map features
per road carrying the same number. The flag is real; the information behind it
is not.

**3. The matcher does not determine direction, so you cannot simply stop
merging.** This is the blocker, and it is the reason the merge exists.

Task 3 established (`03-numerical-accuracy.md`, and the m7 experiment behind
F-3.2) that the two directed edges of a two-way road are **exactly degenerate**
under the model: identical geometry gives identical perpendicular snap distance,
so identical emission; and the same-edge transition is identical too. Nothing in
the HMM breaks that tie on a property of the data. Measured: the matcher decoded
a due-east drive as *westbound* on all four seeds tested.

`_SourceDistCache`'s own docstring says why the undirected treatment was chosen:

> Speed is a magnitude, so on-road distance is measured undirected: this makes
> the estimate robust to the matcher flip-flopping between the two directed
> edges of a two-way road (a common HMM ambiguity that would otherwise inflate
> distances by a full edge length per fix).

That is a correct decision given an arbitrary direction. **The feature is not
"stop merging" — it is "determine the direction first."** Stopping the merge
without that would assign real speeds to coin-flip directions, which is worse
than today.

This is also why the fix cannot be a `to_geojson` flag: by export time the
information is two releases gone.

## The good news: the information is already computed

Direction of travel does not have to come from the matcher. It is in the
*sequence*, and `derive_speeds` already has it.

`speeds._canonical_arc:189-197` puts both directed edges of a road into one
frame:

```python
P, Q = (u, v) if u <= v else (v, u)
arc_from_P = s_dir if u == P else (L - s_dir)
```

For two consecutive fixes on the same road, the **sign of
`arc_from_P(fix_b) - arc_from_P(fix_a)`** is the direction of travel in that
canonical frame — derived from where the vehicle actually moved, not from which
directed edge the matcher happened to pick. Mapping that sign back to a directed
edge id is a lookup.

For the middle edges of a multi-edge hop, direction comes from the order of the
node path, which `_hop_distance` already computes as `best_path` and currently
walks at `:240-245` before discarding.

So the input is free. What changes is the bookkeeping.

## What would change

| Piece | Change |
|---|---|
| `speeds._hop_distance` | return the traversed edges **with a direction**, using the sign of the canonical arc difference for the endpoint roads and the node-path order for the middle |
| `speeds.derive_speeds` | attribute each observation to one directed edge instead of both; add a `directional=False` keyword |
| `edge_observations` | unchanged schema — `edge_id` simply becomes directional when the flag is on |
| `aggregate_speeds(by_edge=True)` | works unchanged, and now yields per-direction rows |
| `to_geojson(directional=True)` | becomes meaningful with no code change |
| `assign_speeds`, `assign_segment_speeds`, `congestion_report` | work unchanged; their per-edge writes become per-direction |

Note how little of this is new machinery. The change is concentrated in one
function's attribution step.

## Costs and risks

**It halves the sample size per directed edge.** A road with 23 observations
becomes roughly 12 each way, widening every confidence interval by about √2.
For a road with sparse coverage this may turn a usable estimate into two
unusable ones. `min_samples` and the `n` column already surface this, but the
default reporting would get noisier, and that is a real cost, not a rounding
error.

**It changes every downstream number** on two-way roads. That is not a patch
release. Making it opt-in via an additive keyword-only argument with a
back-compatible default (`directional=False`) keeps the frozen-API contract and
lets it ship in a minor version — which is how the ship review's Global
Constraints permit new arguments.

**It entangles with F-3.2.** The undirected `_hop_distance` and the same-edge
`rd = gc_step` shortcut are two halves of the same "direction is not
determined" design. Anyone opening this should read F-3.2's correction first —
its evidence was withdrawn, but the modelling gap it names is real and sits in
the same code.

**The invariance pins must be regenerated.** `tests/test_speeds_invariance.py`
canonicalises edges to physical roads precisely *because* direction is not a
portable property today. If direction becomes determined, that canonicalisation
should be revisited — it would then be hiding a real signal.

## Recommendation

**Build it for 0.7, behind `directional=False`, after a spike.**

The mechanism is sound, the input data is already computed, and the change is
contained. But before writing it, spend an hour answering the question that
decides whether it is worth anything:

> On real customer data, how large is the directional asymmetry, and does it
> survive halving the sample size?

That is measurable today without changing the package: match a real day, then
for each hop compute the sign of the canonical arc difference by hand, split the
observations, and compare the two distributions per road and hour. If the
asymmetry is smaller than the widened confidence intervals, the feature is a
worse answer than the current one and should be declined. If rush-hour
corridors separate cleanly — which is the expectation — it is straightforwardly
worth building.

Declining is a perfectly good outcome here, and the spike is what tells you.

## Related

- `.plans/reviews/2026-08-17-ship/03-numerical-accuracy.md` — F-3.2 and the
  behaviour (a)/(b)/(c) analysis; the m7 degeneracy experiment
- `.plans/reviews/2026-08-17-ship/07b-derive-speeds-profile.md` — the current
  cost profile of `derive_speeds`, if this lands alongside further optimization
- `src/redlight/speeds.py:189-197` (`_canonical_arc`), `:237-245`
  (the both-directions attribution)
