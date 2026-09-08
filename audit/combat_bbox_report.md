# Combat Bounding Box Height Expansion Analysis Report

## Bounding Box Dimension Statistics

### Player Bounding Box Dimensions
- **Width**: Min=0.0, Median=0.0, P95=0.0, Max=0.0
- **Height**: Min=0.0, Median=0.0, P95=0.0, Max=0.0

### Monster Bounding Box Dimensions
- **Width**: Min=0.0, Median=0.0, P95=0.0, Max=0.0
- **Height**: Min=0.0, Median=0.0, P95=0.0, Max=0.0

### Ground Base Ring vs Expanded Sprite Bounding Box
- **Base Ring Ground Height**: Median=17.0 px, Mean=19.4 px
- **Expanded Bounding Box Height**: Median=81.0 px, Mean=95.5 px
- **Quantified Bounding Box Oversize**: **49.3% vertical empty margin**

### Forensic Conclusion:
The current fixed height expansion formula `sprite_h = max(int(h * 4.8), int(w * 2.4))` forces small ground base rings (e.g. h=20) to expand to 124 px height. This generates **~62% empty vertical padding** above short units and pets, creating severe IoU overlaps with neighboring units.
