# Phase 4C.5 — Stress Candidate Labeling Guide

This guide defines the ground-truth annotation taxonomy for labeling candidate crops in `audit/manual_labels_stress.xlsx`.

---

## 1. Ground Truth Binary Mapping

All labels fall strictly into two target ground-truth classes:

$$\\text{Ground Truth} = \\begin{cases} 1 & \\text{if True Entity (TP: player, monster, npc)} \\\\ 0 & \\text{if Scenery / Prop / Noise (FP)} \\end{cases}$$

---

## 2. Category Taxonomy & Definitions

### True Positives ($\text{Label} = 1$)

- **`player`**: Player character sprite (full sprite or head/body crop).
- **`monster`**: Aggressive or neutral mob/monster sprite on exploration map or combat grid.
- **`npc`**: Non-player character entity, shop vendor, or interactive character.

### False Positives ($\text{Label} = 0$)

- **`flower`**: Map flowers, grass bushes, crop plants, or environmental flora.
- **`wall`**: Wall fragments, stone borders, building edges, or fence posts.
- **`tree`**: Tree trunks, leaves, canopy branches, or forest vegetation.
- **`ground_tile`**: Floor tiles, dirt paths, cobblestone textures, or water ripples.
- **`movement_cell`**: Green PM movement grid tiles or combat range highlights.
- **`decoration`**: Statues, lamps, flags, signs, rocks, or static map ornaments.
- **`ui_fragment`**: HUD banners, action bar icons, minimap edges, or text overlays.
- **`box`**: Wooden crates, barrels, chests, or interactive containers.
- **`unknown`**: Ambiguous scenery artifacts or noise fragments.

---

## 3. Labeling Procedure in Excel

1. Open [`audit/manual_labels_stress.xlsx`](file:///c:/Users/Andres/Desktop/bot/audit/manual_labels_stress.xlsx) in Excel / LibreOffice.
2. For each row, inspect the crop image stored at `audit/stress_candidates/<filename>`.
3. Select the precise category from the **`label`** dropdown in Column F.
4. Save the completed workbook.
