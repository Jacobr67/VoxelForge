"""
src/voxeliser/colour_mapper.py

Maps each occupied voxel's RGB colour to the closest matching Minecraft
block from the active palette using nearest-neighbour search in LAB
colour space (perceptually uniform — much better than raw RGB distance).

The palette is loaded from JSON files in assets/block_palettes/.
Multiple palette groups can be combined; the active set is their union.
"""

import json
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Path to block palette JSON files relative to project root
PALETTE_DIR = Path(__file__).parent.parent.parent / "assets" / "block_palettes"


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class MinecraftBlock:
    """A single Minecraft block entry from the palette JSON."""
    id:           str           # e.g. "minecraft:stone"
    display_name: str           # e.g. "Stone"
    rgb:          tuple         # (R, G, B) 0-255
    groups:       list          # e.g. ["full_blocks", "survival_obtainable"]


@dataclass
class MappedGrid:
    """
    Output of ColourMapper.map() — a 3D grid of Minecraft block IDs.

    Attributes:
        block_grid:  (X, Y, Z) array of block ID strings. Empty = "air".
        occupied:    (X, Y, Z) bool — which voxels are filled.
        palette:     List of MinecraftBlock entries used.
    """
    block_grid: np.ndarray      # dtype object (str), shape (X, Y, Z)
    occupied:   np.ndarray      # bool, shape (X, Y, Z)
    palette:    list            # List[MinecraftBlock]


# ── Colour space helpers ───────────────────────────────────────────────────────

def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert (N, 3) uint8 RGB array to CIE LAB colour space.
    LAB is perceptually uniform so nearest-neighbour gives better
    colour matches than raw Euclidean RGB distance.
    """
    # Normalise to [0, 1]
    rgb_f = rgb.astype(np.float32) / 255.0

    # Linearise (sRGB gamma removal)
    mask = rgb_f > 0.04045
    rgb_f[mask]  = ((rgb_f[mask]  + 0.055) / 1.055) ** 2.4
    rgb_f[~mask] = rgb_f[~mask] / 12.92

    # RGB → XYZ (D65 illuminant)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)

    xyz = rgb_f @ M.T

    # Normalise by D65 white point
    xyz[:, 0] /= 0.95047
    xyz[:, 2] /= 1.08883

    # XYZ → LAB
    epsilon = 0.008856
    kappa   = 903.3

    fx = np.where(xyz[:, 0] > epsilon,
                  xyz[:, 0] ** (1/3),
                  (kappa * xyz[:, 0] + 16) / 116)
    fy = np.where(xyz[:, 1] > epsilon,
                  xyz[:, 1] ** (1/3),
                  (kappa * xyz[:, 1] + 16) / 116)
    fz = np.where(xyz[:, 2] > epsilon,
                  xyz[:, 2] ** (1/3),
                  (kappa * xyz[:, 2] + 16) / 116)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return np.stack([L, a, b], axis=1)


# ── Main class ─────────────────────────────────────────────────────────────────

class ColourMapper:
    """
    Maps voxel colours to Minecraft blocks using perceptual colour matching.

    Usage:
        mapper = ColourMapper(active_groups=["full_blocks", "survival_obtainable"])
        mapped = mapper.map(voxel_grid, processed_mesh.colour_map)
    """

    # Available palette group names (must match JSON filenames in assets/)
    AVAILABLE_GROUPS = [
        "full_blocks",
        "survival_obtainable",
        "natural",
        "stone_and_ores",
        "wood_and_leaves",
        "coloured_blocks",
    ]

    def __init__(self, active_groups: Optional[list] = None):
        """
        Args:
            active_groups: List of palette group names to include.
                           If None, all groups are used.
                           The active palette is the UNION of all selected groups.
        """
        if active_groups is None:
            active_groups = self.AVAILABLE_GROUPS

        invalid = [g for g in active_groups if g not in self.AVAILABLE_GROUPS]
        if invalid:
            raise ValueError(
                f"Unknown palette groups: {invalid}. "
                f"Valid groups: {self.AVAILABLE_GROUPS}"
            )

        self.active_groups = active_groups
        self._palette: list[MinecraftBlock] = []
        self._palette_lab: Optional[np.ndarray] = None  # (M, 3) LAB

        self._load_palette()

    def map(self, voxel_grid, colour_map: np.ndarray) -> MappedGrid:
        """
        Assign a Minecraft block to every occupied voxel.

        Args:
            voxel_grid:  VoxelGrid from Voxeliser.voxelise().
            colour_map:  (N_faces, 3) uint8 array from ProcessedMesh.colour_map.
                         Maps face_id → RGB colour.

        Returns:
            MappedGrid with per-voxel block ID strings.
        """
        if not self._palette:
            raise RuntimeError(
                "No blocks loaded in palette. Check assets/block_palettes/."
            )

        res = voxel_grid.resolution
        occupied = voxel_grid.occupied
        face_ids = voxel_grid.face_ids

        logger.info(
            f"Mapping {occupied.sum():,} voxels to Minecraft blocks "
            f"(palette size: {len(self._palette)} blocks)..."
        )

        # ── 1. Gather unique face IDs used in occupied voxels ─────────────────
        occ_mask    = occupied
        used_faces  = face_ids[occ_mask]         # (V,) int32
        unique_fids = np.unique(used_faces)
        unique_fids = unique_fids[unique_fids >= 0]

        # ── 2. Look up RGB for each unique face ───────────────────────────────
        face_rgb = colour_map[unique_fids]        # (U, 3) uint8

        # ── 3. Convert face colours to LAB ────────────────────────────────────
        face_lab = _rgb_to_lab(face_rgb.astype(np.float32))

        # ── 4. Nearest-neighbour search in LAB space ──────────────────────────
        # For each face colour, find the closest palette block
        block_indices = self._nearest_block(face_lab)  # (U,) int

        # Build face_id → block_index lookup
        fid_to_block = {
            int(fid): block_indices[i]
            for i, fid in enumerate(unique_fids)
        }

        # ── 5. Fill output block grid ─────────────────────────────────────────
        block_grid = np.full((res, res, res), "air", dtype=object)

        xs, ys, zs = np.where(occ_mask)
        for x, y, z in zip(xs, ys, zs):
            fid = int(face_ids[x, y, z])
            if fid in fid_to_block:
                block_idx  = fid_to_block[fid]
                block_grid[x, y, z] = self._palette[block_idx].id

        logger.info("Colour mapping complete.")

        return MappedGrid(
            block_grid=block_grid,
            occupied=occupied,
            palette=self._palette,
        )

    def get_palette_blocks(self) -> list:
        """Return the list of active MinecraftBlock entries."""
        return list(self._palette)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_palette(self):
        """Load and merge all active palette group JSON files."""
        seen_ids = set()

        for group in self.active_groups:
            json_path = PALETTE_DIR / f"{group}.json"

            if not json_path.exists():
                logger.warning(
                    f"Palette file not found: {json_path}. "
                    f"Skipping group '{group}'."
                )
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load palette '{group}': {e}")
                continue

            for entry in data.get("blocks", []):
                block_id = entry.get("id", "")
                if not block_id or block_id in seen_ids:
                    continue

                try:
                    block = MinecraftBlock(
                        id=block_id,
                        display_name=entry.get("display_name", block_id),
                        rgb=tuple(entry["rgb"]),
                        groups=entry.get("groups", [group]),
                    )
                    self._palette.append(block)
                    seen_ids.add(block_id)
                except (KeyError, TypeError) as e:
                    logger.warning(f"Skipping malformed block entry {entry}: {e}")

        if not self._palette:
            logger.warning(
                "No palette blocks loaded. Ensure assets/block_palettes/ "
                "JSON files are populated."
            )
            return

        # Pre-compute LAB values for all palette blocks
        palette_rgb = np.array(
            [b.rgb for b in self._palette], dtype=np.float32
        )
        self._palette_lab = _rgb_to_lab(palette_rgb)

        logger.info(
            f"Loaded {len(self._palette)} blocks from groups: "
            f"{self.active_groups}"
        )

    def _nearest_block(self, query_lab: np.ndarray) -> np.ndarray:
        """
        For each query colour (N, 3 LAB), return the index of the closest
        palette block using Euclidean distance in LAB space.

        Args:
            query_lab: (N, 3) float32 LAB colours to match.

        Returns:
            (N,) int array of palette indices.
        """
        # Compute pairwise distances: (N, M)
        # Using broadcasting: expand dims to (N, 1, 3) - (1, M, 3) → (N, M, 3)
        diff = query_lab[:, np.newaxis, :] - self._palette_lab[np.newaxis, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))   # (N, M)
        return dist.argmin(axis=1)                # (N,)