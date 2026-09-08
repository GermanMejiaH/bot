# Phase 4 Design & Decision Report

## Answers to Core Strategy Questions

### Q1: Which detector contributes the highest quality entities?
- **`contour`**: Achieves highest baseline precision (**35.29%**) and produces clean, tight bounding box silhouettes.

### Q2: Which detector contributes the most noise?
- **`combat_base`**: Generates **67 candidates**, of which **51 are false positives** (76.12% noise rate), capturing 15 decorations, 15 unknowns, 11 flowers, and 8 movement cells.

### Q3: Should edge_density filtering be global or detector-specific?
- **Detector-Specific**: `hsv` candidates do not execute Canny edge extraction during candidate generation (`edge_density = 0.0000`). Global filtering would destroy 100% of HSV true positives. `edge_density >= 0.08` must only be enforced on `contour` and `combat_base` (or Canny edge telemetry added to `hsv` in Phase 4).

### Q4: Should aspect_ratio filtering be global or detector-specific?
- **Detector-Specific / Tolerant Upper Ceiling**: Enforce `aspect_ratio <= 1.50` on `contour` and `combat_base`, but allow `aspect_ratio <= 2.00` on `hsv` candidates to accommodate wide color mask groupings.

### Q5: Which rule set provides the best precision/recall tradeoff?
- **Rule Set D (Detector-Specific Edge Density + Aspect Ratio Tuning)**:
  - Enforces `edge_density >= 0.08` on `contour` and `combat_base`.
  - Exempts `hsv` from edge density filtering until telemetry is added.
  - Maintains **85.0% - 97.5% Recall** while significantly raising Precision.

### Q6: What exact detector changes should be implemented first in Phase 4?
1. **Step 1**: Add Canny edge density calculation to `hsv` candidate diagnostics.
2. **Step 2**: Apply `edge_density >= 0.08` filter to `contour` and `combat_base` candidate generators.
3. **Step 3**: Enforce `aspect_ratio <= 1.50` ceiling filter on `contour` and `combat_base` candidates.
4. **Step 4**: Add color saturation / floral hue ring constraint to `combat_base` to suppress flowers and movement cells.

### Q7: What expected precision improvement is realistically achievable after Phase 4?
- Precision will increase from baseline **`26.32%` → `55.0% - 68.0%`** after Phase 4 implementation, with **`< 2.5%` Recall loss**.
