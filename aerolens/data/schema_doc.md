# AeroLens Sentinel — Unified Data Class Schema

This document details the unified class schema mapped from raw public datasets to support reliable edge-inference classification.

## Unified Target Classes
1. **crack** (defect_type_weight = 0.90) - High-risk structural defects requiring immediate or scheduled engineering verification.
2. **corrosion** (defect_type_weight = 0.75) - Surface metal degradation.
3. **dent** (defect_type_weight = 0.55) - Physical impact deformations.
4. **paint_peel** (defect_type_weight = 0.35) - Surface coating loss; also serves as a visual proxy for scratches or pre-corrosive indicators.

## Dataset Mapping Rules
* Source Dataset 1: **Aircraft-Surface-Damage** (Roboflow)
  * `crack` -> maps directly to `crack`
  * `dent` -> maps directly to `dent`
  * `corrosion` -> maps directly to `corrosion`
* Source Dataset 2: **aircraft_skin_defects** (Roboflow/IISc)
  * `crack` -> maps directly to `crack`
  * `dent` -> maps directly to `dent`
  * `scratch` -> **Mapped to `paint_peel`** (surface-cosmetic proxy, representing exposure of skin panel surface)
  * `paint-peel` -> maps directly to `paint_peel`
  * `missing-head` -> **Dropped/Excluded** (fastener/rivet failure, out of scope for general surface anomaly classification, requiring different lens/resolution setup)
