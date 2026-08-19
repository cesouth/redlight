# Fast Map Matching (UBODT) for redlight — proposal

**Status:** awaiting decision (build now / build later / decline)
**Written:** 2026-08-19, as Task 8 Part B of the v0.6.0 ship review
**Evidence:** `.plans/reviews/2026-08-17-ship/07-performance-baseline.md`,
`07b-derive-speeds-profile.md`, and measurements taken for this document.

## Recommendation up front

**Decline.** Not "later" — decline, and record why so it is not re-proposed.

The measured reason is short: `_CSRDistCache` **is already a UBODT**. It is
built lazily instead of eagerly, bounded by the queries a run actually makes
instead of a global delta, and sized to what the run touches (790 entries of a
900-node network) instead of all pairs. It runs at a **99.5 % hit rate**, so
99.5 % of transition-distance requests are already a hash lookup. FMM's entire
proposition is to turn those requests into hash lookups. They already are.

The remaining 0.5 % — 857 real Dijkstra runs out of 159,600 queries at 40k
points — is the only thing a UBODT could remove, and it is a fraction of the
4–6 % that the whole bounded-Dijkstra layer costs.

## What FMM is

Fast Map Matching (Yang & Gidofalvi, 2018, *IJGIS* 32(3)) precomputes an
**Upper-Bounded Origin-Destination Table**: for every node pair whose shortest
path is within a distance bound δ, it stores the distance and the next hop.
The HMM's per-transition shortest-path search then becomes a table lookup, and
the authors report order-of-magnitude speedups over a Dijkstra-per-transition
baseline.

The word doing the work in that comparison is *per-transition*. FMM's baseline
recomputes a shortest path for every candidate pair at every step. This package
has not done that since `_CSRDistCache` was introduced.

## What it would change here

| Piece | Change |
|---|---|
| `Network.csgraph()` | unchanged — already produces the CSR the build needs |
| new `Network.ubodt(delta)` | `scipy.sparse.csgraph.dijkstra(csr, limit=delta)` over all sources, flattened to `(source, target, next_hop, distance)` |
| `matching._CSRDistCache.lookup` | consult the table first, fall back to bounded Dijkstra on a miss |
| invalidation | the table is a pure function of `(csgraph, delta)`; `Network` mutates edge *attributes* (`assign_speeds` writes `speed_mps`) but not topology or `length_m`, so a table built once per `Network` stays valid for its lifetime |
| storage | in memory as four numpy arrays, or on disk as `.npz` beside the network |

## What it would cost — measured, not estimated

Built with `scipy.sparse.csgraph.dijkstra(..., limit=3000.0)` (δ = 3 km, which
covers the matcher's cutoff regime for GPS steps up to ~375 m), 16 bytes per
row (3 × int32 + float32):

| Network | Nodes | Edges | Pairs ≤ δ | Table | Build |
|---|---|---|---|---|---|
| `examples/sample_data/network.geojson` | 20 | 62 | 248 | <0.1 MB | 0.00 s |
| benchmark grid 30×30 | 900 | 3,480 | 614,960 | 9.8 MB | 0.07 s |
| benchmark grid 60×60 | 3,600 | 14,160 | 3,778,820 | 60.5 MB | 0.53 s |

**Where it stops being practical.** Note the 30×30 grid is smaller in extent
than δ, so nearly every pair qualifies (614,960 of a possible 810,000) and the
table is O(n²). At 60×60 the bound starts binding and growth falls back toward
O(n·k), k = nodes within δ. Extrapolating the measured k ≈ 1,050 to real
networks:

- a small town, ~5,000 nodes → ~5 M rows, **~84 MB**, build ~1 s
- a mid-sized city, ~50,000 nodes → ~53 M rows, **~840 MB**, build ~25 s
- a metro area, ~200,000 nodes → ~210 M rows, **~3.4 GB**, build ~160 s

So the practical ceiling is somewhere around a small city, and it is set by
memory, not build time. For scale, the package's entire current working set on
a 40k-point run is a **7.7 MB** peak traced allocation and 134 MB RSS.

## Can it be exact?

Only with a fallback, and the fallback is the thing being replaced.

The current cutoff is `max(max_route_dist_factor * gc_step, max_dist * 4)` and
**varies per step** — it scales with the GPS step, which is unbounded (a long
gap between fixes produces a large step). A fixed-δ table therefore gives
identical answers **only when δ ≥ every cutoff the run uses**, which cannot be
guaranteed in advance.

On a miss the table cannot distinguish "beyond δ" from "genuinely unreachable".
Reporting unreachable would push the transition into the saturating-penalty
branch (`matching.py:320-332`) and change the decoded path — forbidden under
Task 8's invariance constraint. So a correct implementation must keep the
bounded Dijkstra as a fallback, which means keeping all the machinery FMM was
supposed to remove, plus the table.

## The honest verdict

Against the measured baseline, the ceiling on the win is small and the floor on
the cost is not:

- Transition-distance lookups already hit cache **99.5 %** of the time.
- The entire bounded-Dijkstra layer is **4–6 %** of `derive_speeds` and a
  minority of `HMMMatcher.match`'s `lookup` cost, most of which is dict access
  and LRU bookkeeping — work a UBODT does *not* remove, because a table lookup
  is also a dict access. Measured directly: skipping the LRU bookkeeping
  entirely produced **no measurable gain** (Task 8 Part A, reverted).
- In exchange: 10 MB–3.4 GB of memory, a build step, an invalidation contract,
  and a δ parameter users must reason about.

There is also a scope argument. Task 7b found `HMMMatcher.match` is **39 %** of
pipeline wall time and `derive_speeds` **60 %**. FMM targets a fraction of the
smaller share. The largest measured win available to this package is
`derive_speeds`' row-by-row frame construction at **60 % of its own runtime**
(F-7b.1) — ordinary work, no new concepts, no new parameters.

## Alternatives, on the same terms

| Option | Verdict |
|---|---|
| **Contraction hierarchies** | Strictly more complex than a UBODT, same 99.5 %-cached problem, and a preprocessing step that must be invalidated. Worse trade than FMM. |
| **A\* with a geometric heuristic** | Would speed up the 0.5 % of lookups that miss cache. Cheap to implement and needs no precomputation or new parameter — the only idea here worth keeping on the table, and still worth <1 % overall. |
| **More aggressive caching** | Nothing to win: the cache never fills (790 entries of 10,000) and the hit rate is 99.5 % (F-7.4). Cutoff quantization is explicitly ruled out — rounding a cutoff *down* changes output. |
| **Document `NearestMatcher` as the fast path** | Already true and already measured: 3.7–10× faster than the HMM. Free. The honest recommendation for users who need throughput and can accept the accuracy cost quantified in `methodology.md` §2.3. |

## If this is ever revisited

Three things would have to be true, and none is today:

1. `_CSRDistCache`'s hit rate collapses on some real workload — plausible only
   on a network with far more distinct source nodes than the runs measured
   here. Instrument before assuming.
2. `derive_speeds` has already been optimized, so matching is the majority of
   pipeline time rather than 39 % of it.
3. The target networks are small enough that the table fits comfortably — under
   roughly 50,000 nodes on typical hardware.

## References

- Yang, C., & Gidófalvi, G. (2018). Fast map matching, an algorithm integrating
  hidden Markov model with precomputation. *International Journal of
  Geographical Information Science*, 32(3), 547–570.
- Newson, P., & Krumm, J. (2009). Hidden Markov map matching through noise and
  sparseness. *ACM SIGSPATIAL GIS*, 336–343.
