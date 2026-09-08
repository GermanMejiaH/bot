# Surviving False Positives Analysis Report

**Total Surviving False Positives**: `63`

| Candidate ID | Method | Category Label | Edge Density | Aspect Ratio | Visual Property & Survival Reason |
| --- | --- | --- | --- | --- | --- |
| `0002` | `contour` | `unknown` | 0.1696 | 0.5833 | Non-standard terrain artifact with isolated sharp contrast edge. |
| `0006` | `combat_base` | `bag` | 0.1175 | 0.4167 | Compact vertical adorno/flower sprite with detailed line work mimicking character contour. |
| `0006` | `contour` | `bag` | 0.0898 | 0.6071 | Compact vertical adorno/flower sprite with detailed line work mimicking character contour. |
| `0007` | `combat_base` | `movement_cell` | 0.0817 | 0.3542 | Compact vertical adorno/flower sprite with detailed line work mimicking character contour. |
| `0007` | `contour` | `movement_cell` | 0.0810 | 1.2407 | Compact vertical adorno/flower sprite with detailed line work mimicking character contour. |
| `0010` | `combat_base` | `movement_cell` | 0.2897 | 0.2368 | Compact vertical adorno/flower sprite with detailed line work mimicking character contour. |
| `0010` | `contour` | `movement_cell` | 0.0905 | 1.3103 | Compact vertical adorno/flower sprite with detailed line work mimicking character contour. |
| `0018` | `combat_base` | `decoration` | 0.1429 | 0.4167 | Vertical banner/pillar sprite with high internal line detail. |
| `0019` | `combat_base` | `decoration` | 0.2263 | 0.4172 | Vertical banner/pillar sprite with high internal line detail. |
| `0020` | `combat_base` | `decoration` | 0.2198 | 0.2248 | Vertical banner/pillar sprite with high internal line detail. |
| `0020` | `contour` | `decoration` | 0.1660 | 1.3684 | Vertical banner/pillar sprite with high internal line detail. |
| `0021` | `combat_base` | `decoration` | 0.2309 | 0.2361 | Vertical banner/pillar sprite with high internal line detail. |
| `0021` | `contour` | `decoration` | 0.1290 | 1.1613 | Vertical banner/pillar sprite with high internal line detail. |
| `0022` | `contour` | `decoration` | 0.1302 | 1.1481 | Vertical banner/pillar sprite with high internal line detail. |
| `0023` | `combat_base` | `decoration` | 0.0989 | 0.3333 | Vertical banner/pillar sprite with high internal line detail. |
| `0023` | `contour` | `decoration` | 0.1302 | 1.1481 | Vertical banner/pillar sprite with high internal line detail. |
| `0024` | `combat_base` | `decoration` | 0.1037 | 0.4030 | Vertical banner/pillar sprite with high internal line detail. |
| `0024` | `contour` | `decoration` | 0.2460 | 1.1667 | Vertical banner/pillar sprite with high internal line detail. |
| `0025` | `contour` | `box` | 0.0896 | 0.4876 | Compact vertical adorno/flower sprite with detailed line work mimicking character contour. |
| `0026` | `combat_base` | `decoration` | 0.1651 | 0.2381 | Vertical banner/pillar sprite with high internal line detail. |

## Why do these FPs survive?
1. **Visual Mimicry**: They possess vertical aspect ratios (`0.4 - 1.2`) and rich internal lines (`edge_density > 0.10`), making them geometrically indistinguishable from entity silhouettes using basic single-channel heuristics.
2. **Next Steps for Phase 4**: Require multi-channel consensus (HSV color mask + contour edge presence) or spatial texture variance to eliminate remaining 39 survivors.
