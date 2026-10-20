---
title: "A5's pentagons are equal-area. Their parents are another story"
description: A5 v1.0 promises a global grid where every cell has exactly the same area. I measured that claim against H3 and it holds to six decimal places. Then I checked the hierarchy. Rolling a point up with cell_to_parent puts it in the wrong coarse cell about 35% of the time. H3's rate is 6–7%.
date: 2026-10-20
draft: false
---

[A5 hit v1.0 last week](https://a5geo.org/blog/2026/10/14/a5-v1) and it's on the
lobste.rs front page today. A5 is a discrete global grid, the same kind of thing as Uber's H3
and Google's S2: it cuts the Earth into cells and gives each cell a 64-bit ID. Its cells are
pentagons (built on a dodecahedron), and the main selling point is that **every cell at a given
resolution has exactly the same area**. H3 can't promise that. The v1.0 post also introduces a
new space-filling curve and set operations on compacted coverings.

The announcement includes one caveat in passing: the hierarchy is only *approximately*
spatial, because you can't split a pentagon exactly into smaller pentagons. The source code
says the same thing. Neither one tells you **how approximate**, and that number matters as soon
as you write `GROUP BY cell_to_parent(cell, 8)`. So I measured it, along with the equal-area
claim and cell shape, and compared the results with H3.

Setup: `pya5` 1.0.0 (Rust backend) and `h3` 4.5.0 from PyPI. I computed areas geodesically on
the WGS84 ellipsoid with `pyproj.Geod` and did polygon overlap in a local Lambert
equal-area projection with shapely. Random points are uniform on the sphere. All numbers below
come from my own runs. The scripts are short and the important parts are inline.

## 1. Equal area: it's real

I sampled about 3,000 distinct cells at two resolutions per system and measured each cell's
boundary on the ellipsoid:

| | cells | max / min area | relative s.d. | roundness (IQ) |
|---|---|---|---|---|
| A5 res 6 (≈8,300 km²) | 2,928 | **1.000059** | 2.0e-6 | 0.717 – 0.772 |
| H3 res 3 (≈12,600 km²) | 2,901 | 2.377 | 0.127 | 0.877 – 0.939 |
| A5 res 10 (≈32 km²) | 2,999 | **1.000004** | 1.2e-7 | 0.721 – 0.765 |
| H3 res 5 (≈257 km²) | 2,996 | 1.940 | 0.124 | 0.891 – 0.910 |

The 0.006% spread left for A5 comes from my measurement, not from A5. When I densify the
boundary further, the worst-case deviation from the nominal `cell_area(6)` keeps falling:

```
segments per edge:   1        4        16       64
max |area/nominal-1|: 1.0e-2   6.8e-4   4.2e-5   2.6e-6
```

It drops about 16× for each 4× increase in segments, which is the quadratic convergence you'd
expect if the true cells are exactly equal. A5 also handles the ellipsoid properly: the code
converts between geodetic and authalic latitude, so the cells are equal on WGS84 and not just
on a sphere. H3's 2× spread at a fixed resolution is what A5 removes.

A small gotcha follows from that table. From res 6 down, `cell_to_boundary()` by default returns
just 5 vertices per cell. If you hand that ring to a GIS tool and ask for its area, you can be
off by up to **1%** (res 6) or 0.5% (res 10). If you need areas, use `cell_area(res)` or pass
`{'segments': 16}`.

**Shape.** The "IQ" column is the isoperimetric quotient, 4πA/P². It's 1 for a circle,
0.907 for a regular hexagon, 0.865 for a regular pentagon and 0.785 for a square. A5's cells come
in at a median of **0.746**, so they're less compact than a square. They're irregular pentagons
(the dodecahedron-to-plane mapping stretches them so they all come out equal). That matters
for anything that treats a cell as "a blob around its center", like nearest-neighbor
smoothing or kernel density. Equal area is paid for with shape.

## 2. The hierarchy: half of the children live next door

A5 has aperture 4: each cell has four children one resolution down. Here's what that looks
like geometrically. I took 300 random parents at res 8 and res 12, built the union of their
four `cell_to_children`, and measured how much of that union falls outside the parent:

| | children's area outside the parent | children with >50% outside |
|---|---|---|
| **A5** (res 8 and res 12) | **36.8%** mean (31.9% – 41.8%) | **2.0 of 4**, every time |
| H3 (res 4 and res 7) | 7.1% (constant) | 0 of 7 |

Each A5 parent has two children that are mostly in a *neighboring* cell. The four children
have the right total area, but the index puts them in the wrong place. H3's well-known
"children don't quite tile the parent" effect is 7.1%. A5's is five times that.

### What that does to a roll-up

The usual pattern is to index points at a fine resolution and aggregate at a coarse one by
truncating IDs. That's cheap, 142 ns per call here against 1.4 µs to recompute from
lon/lat. I took 20,000 random points, indexed each at res `r + k`, called
`cell_to_parent(·, r)`, and compared the result with indexing the same point directly at
res `r`:

```python
truth = a5.lonlat_to_cell((lon, lat), r)
fine  = a5.lonlat_to_cell((lon, lat), r + k)
wrong += a5.cell_to_parent(fine, r) != truth
```

| levels rolled up (k) | A5 `cell_to_parent` | A5 `cell_to_supercell` | H3 `cell_to_parent` |
|---|---|---|---|
| 1 | **37.4%** wrong | 19.6% | 7.0% |
| 2 | 35.3% | 11.9% | 5.9% |
| 4 | 34.5% | 2.7% | 6.3% |
| 8 | 34.5% | 0.2% | 6.3% |

(These are res 10 for A5 and res 5 for H3. Res 6 and res 3 give the same numbers to within
a point.)

What these numbers mean:

- **More than a third of your points end up in the wrong coarse cell.** It's not noise along the
  edges. I checked where the misattributed points go: in 5,000 trials every one landed in an
  *edge-sharing neighbor* of the correct cell, never farther away. So the error is local, but
  it's large. For a choropleth with sharp gradients (coastlines, city limits, a single hot
  cell), about 35% of the mass moves one cell over.
- **The error doesn't grow with depth, but it doesn't shrink either.** Whether you roll up 1
  level or 8, about 35% of points are misplaced. Repeated rounds of "two children stick out"
  settle into a fractal-edged region around the parent that overlaps it by only about 65%.
- **A5 has a fix, and it's in the API: `cell_to_supercell`.** It returns the coarse cell that
  contains the fine cell's *center*, so the only error left is the fine cell's own size,
  which drops by half with each level. Eight levels down it's wrong for 0.2% of points. It
  costs 659 ns per call, about 4.6× more than `cell_to_parent` but still half the cost of a
  full `lonlat_to_cell`. The docstring is clear that this is the function for spatial
  aggregation. The trouble is that "parent" is the name everyone already knows from H3, where
  it's wrong only 6% of the time.

## 3. Compacted coverings: correct as sets, wrong when drawn

The v1.0 post's main feature is set algebra (`union`, `intersect`, `difference`, `area`) on
*compacted* coverings, where any four complete children are replaced by their parent ID. As
sets of fine cells these behave correctly. I filled a 2°×1.2° box over Germany at res 11 and
`a5.area()` of the covering matched the box to **0.07%**.

The trouble starts when you draw a compacted covering, the way you'd draw any list of cell IDs
in a map viewer or with `cell_to_boundary`. The coarse cells are drawn in their parent's
geometry, and we've just seen that this isn't where their fine cells are:

| | fine cells → compacted | drawn shape vs. fine cells (sym. diff.) | drawn cells overlapping each other |
|---|---|---|---|
| A5 res 9 | 149 → 53 | 12.2% | 9.6% |
| A5 res 11 | 2,362 → 244 | **14.2%** | **12.1%** |
| H3 res 6 | 576 → 126 | 2.6% | 2.3% |
| H3 res 7 | 4,014 → 372 | 2.7% | 2.6% |

Draw A5's compacted covering and about one-seventh of the outline is wrong, with drawn cells
overlapping each other by about 12%. H3 has the same problem at a fifth of the size, which is
small enough that most people never notice it. To render a compacted covering, `uncompact` it
first, or make sure your visualization layer does.

## So, should you use it?

My read on the numbers:

- **If you need equal area, A5 delivers.** Cell counts become areas with no
  correction, and that's a real gain over H3's 2× spread for density maps and for anything
  statistical that assumes equal-sized cells.
- **Treat `cell_to_parent` as an index operation, not a spatial one.** It's right for
  compaction and ID-range tricks. For aggregating by location it's wrong 35% of the time. Use
  `cell_to_supercell`, or index at the coarse resolution directly. If you're porting an
  H3 pipeline, this is the line to change. The function names are the same and the error is
  more than five times larger.
- **Expect cells that are less round than squares.** If your analysis depends on neighborhoods
  (k-rings, smoothing), A5 cells have 5 edge neighbors and 7 neighbors in total (I checked
  5,000 cells; it was 5 and 7 every time), and they're stretched.

None of this is hidden. The A5 docs call the hierarchy approximate and the library ships the
spatial alternative. What they don't say is that "approximate" means two of every four children
are mostly outside their parent. That should be in the docs next to `cell_to_parent`.

*Sources: [A5 v1.0 announcement](https://a5geo.org/blog/2026/10/14/a5-v1);
[`pya5` 1.0.0 on PyPI](https://pypi.org/project/pya5/), whose source comments describe
children that "stick out" and supercells as the spatial counterpart;
[H3](https://h3geo.org/) 4.5.0 for comparison. All measurements are my own, on a single run
with fixed seeds.*
