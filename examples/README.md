# AAC Examples

Three self-contained demos showing how AAC works in practice. No dataset downloads required -- all use synthetic data.

## Quick Start

```bash
pip install -e .  # install aac package
python examples/demo_grid_navigation.py
python examples/demo_road_routing.py
python examples/demo_terrain_routing.py
```

## Demo 1: Grid Navigation (`demo_grid_navigation.py`)

**What it shows:** How AAC learns a compressed heuristic that speeds up A* search on a 2D grid with obstacles.

**Key comparison (matched memory):**
- **Dijkstra** -- no heuristic, explores all reachable nodes
- **ALT (K=16)** -- classical landmark heuristic, stores 16 values per vertex
- **AAC (m=16, K₀=32)** -- learned compressed heuristic, stores 16 values per vertex

**Expected output:**
```
[Dijkstra]  Cost: 28.04  Expansions: 253
[ALT K=16]  Cost: 28.04  Expansions: 36   (85.8% reduction)
[AAC m=16]  Cost: 28.04  Expansions: 55   (78.3% reduction)

Memory: ALT = 16 values/vertex, AAC = 16 values/vertex (matched)
All paths optimal (cost = 28.04)
```

AAC achieves competitive quality at matched memory by learning to select the best 16 landmarks from a pool of 32. All paths are provably optimal. Over 50 random queries the gap narrows to <1 pp (ALT: 87.1%, AAC: 86.7%).

## Demo 2: Road Routing (`demo_road_routing.py`)

**What it shows:** The Pareto memory-accuracy tradeoff on a 500-node synthetic road network. ALT with K landmarks stores K values/vertex; AAC with K₀ candidates compressed to m stores m values/vertex.

**Key insight:** the row-stochastic compressor lets AAC learn which subset of K₀ candidates to retain, so the memory-accuracy tradeoff is differentiable and pool-agnostic at deployment.

**Expected output (illustrative; small synthetic graph, exact numbers depend on seed):**
```
Memory = 16 values/vertex:
  AAC (K0=32)    ~94% reduction
  ALT (K=16)     ~94% reduction
```

> On the paper's matched-memory benchmarks (9 DIMACS + OSMnx road networks), pure FPS-ALT modestly leads AAC by 0.9-3.9 pp at B=64 B/v on expansion count (with AAC 1.24-1.51x faster at p50 wall-clock latency); this small-graph demo is *not* a benchmark and is not meant to imply AAC is preferable to FPS-ALT on road networks. See the paper (Sec.5) for the full headline.

## Demo 3: Terrain Routing (`demo_terrain_routing.py`)

**What it shows:** The end-to-end differentiable pipeline -- a neural network observes terrain features and produces an admissible heuristic in a single forward pass.

**Pipeline:** terrain features -> CNN encoder -> edge costs -> smooth Bellman-Ford -> compress -> heuristic -> A*

**Two modes compared:**
- **Static AAC** -- compressed from true distances, guaranteed optimal
- **Contextual AAC** -- single forward pass through encoder, very fast, near-optimal

**Expected output:**
```
Static AAC (true distances): cost=41.40, expansions=30 (69.4% reduction) [optimal]
Contextual AAC (1 fwd pass): cost=41.61, expansions=16 (83.7% reduction) [0.5% regret]
```

The row-stochastic compression guarantees `h(u,t) ≤ d(u,t)` for static AAC. The contextual mode trades this guarantee for single-pass inference speed.

## When to Use What

| Use Case | Method | Guarantee |
|----------|--------|-----------|
| Static road networks at matched memory | FPS-ALT | Optimal paths; modestly tighter heuristic than AAC (paper Sec.5) |
| Differentiable pipeline (learned edge costs) | Contextual AAC | Deployment-time admissibility; differentiable training |
| ALT-family research / matched-memory diagnostic | AAC | Admissibility by architecture; pool-agnostic; exposes ALT-on-subset at deployment |
| Maximum static-oracle speed (road networks) | CH / Hub Labeling | Orders-of-magnitude faster than ALT; not differentiable |
