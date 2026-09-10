# Phase 4D.1 — Formal Entity Taxonomy & Classification Rules

This document establishes the official ground-truth taxonomy for validating character entity candidate crops in the DTA / AURA perception pipeline.

---

## 1. Formal Definition of a Valid Entity

A **Valid Entity** is defined as an active character sprite present on the game map screen that represents a controllable or interactive game entity actor (Player character, Monster mob, or Non-Player Character vendor/questgiver).

$$\text{Valid Entity} \iff \text{Sprite Actor} \in \{\text{Player}, \text{Monster}, \text{NPC}\}$$

To be labeled **Positive ($\text{Label} = 1$)**, the bounding box crop MUST contain the primary visual core (head, torso, or complete sprite body) of a Player, Monster, or NPC actor with sufficient spatial coverage ($\ge 30\%$ target occupancy).

---

## 2. Taxonomy Categorization Matrix

### Positive Ground Truth Classes ($\text{Label} = 1$)

| Class | Visual Characteristics | Examples |
| :--- | :--- | :--- |
| **`Player`** | Controllable player avatar sprite in idle, walking, or combat animation state. | Full player character body, torso crop, or head/body combo. |
| **`Monster`** | Aggressive or neutral monster mob sprite on exploration map or combat grid. | Mob sprites (Gobball, Tofu, Bouftou, Dark Vlad, etc.). |
| **`NPC`** | Non-player character vendor, questgiver, or interactive shopkeeper sprite. | Map NPCs, guard sprites, interactive shopkeepers. |

### Negative Ground Truth Classes ($\text{Label} = 0$)

| Class Category | Visual Description & Noise Artifacts | Ground Truth |
| :--- | :--- | :---: |
| **`Decorations`** | Static map ornaments: statues, street lamps, signposts, flags, rocks, fountains. | **0** |
| **`Trees & Plants`** | Environmental flora: flowers, grass bushes, crop plants, tree trunks, canopy foliage. | **0** |
| **`Walls & Props`** | Structural scenery: stone borders, fences, building walls, wooden barrels, crates. | **0** |
| **`UI & HUD Elements`** | Interface overlays: action bar icons, spell panels, minimap borders, HUD banners. | **0** |
| **`Combat Overlays`** | Grid highlights: green PM movement cells, red AP/range attack highlights, turn indicators. | **0** |
| **`Visual Effects`** | Dynamic particle effects: spell animations, aura glows, particle bursts, highlight halos. | **0** |
| **`Partial Scenery`** | Bounding box crops where background ground tile covers $> 70\%$ of the image area. | **0** |

---

## 3. Labeling Quality Rules for Human Annotators

1. **Primary Target Occupancy**: If the crop contains a valid entity, but the entity occupies less than 20% of the bounding box area (e.g. 80% ground tile), mark as **0 (Invalid BBox)**.
2. **Multiple Entities**: If multiple entities exist in one crop, mark **1** if the central entity is a valid player/monster/NPC.
3. **Occluded Entities**: Mark **1** if the character head/torso is visible through foliage or scenery.
