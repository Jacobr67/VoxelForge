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
    id:           str    # e.g. "minecraft:stone"
    display_name: str    # e.g. "Stone"
    rgb:          tuple  # (R, G, B) 0-255
    groups:       list   # e.g. ["full_blocks", "survival_obtainable"]


@dataclass
class MappedGrid:
    """
    Output of ColourMapper.map() — a 3D grid of Minecraft block IDs.

    Attributes:
        block_grid:  (X, Y, Z) array of block ID strings. Empty = "air".
        occupied:    (X, Y, Z) bool — which voxels are filled.
        palette:     List of MinecraftBlock entries used.
    """
    block_grid: np.ndarray   # dtype object (str), shape (X, Y, Z)
    occupied:   np.ndarray   # bool, shape (X, Y, Z)
    palette:    list         # List[MinecraftBlock]


# ── Colour space helpers ───────────────────────────────────────────────────────

def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert (N, 3) float32 RGB array (values 0-255) to CIE LAB.
    LAB is perceptually uniform so nearest-neighbour gives better
    colour matches than raw Euclidean RGB distance.
    """
    rgb_f = rgb.astype(np.float32) / 255.0

    # Linearise sRGB gamma
    mask         = rgb_f > 0.04045
    rgb_f[mask]  = ((rgb_f[mask]  + 0.055) / 1.055) ** 2.4
    rgb_f[~mask] = rgb_f[~mask] / 12.92

    # RGB → XYZ (D65)
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
    eps   = 0.008856
    kappa = 903.3

    fx = np.where(xyz[:, 0] > eps, xyz[:, 0] ** (1/3), (kappa * xyz[:, 0] + 16) / 116)
    fy = np.where(xyz[:, 1] > eps, xyz[:, 1] ** (1/3), (kappa * xyz[:, 1] + 16) / 116)
    fz = np.where(xyz[:, 2] > eps, xyz[:, 2] ** (1/3), (kappa * xyz[:, 2] + 16) / 116)

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
            active_groups: Palette group names to include.
                           None = all groups. Active palette is their union.
        """
        if active_groups is None:
            active_groups = self.AVAILABLE_GROUPS

        invalid = [g for g in active_groups if g not in self.AVAILABLE_GROUPS]
        if invalid:
            raise ValueError(
                f"Unknown palette groups: {invalid}. "
                f"Valid: {self.AVAILABLE_GROUPS}"
            )

        self.active_groups = active_groups
        self._palette: list[MinecraftBlock] = []
        self._palette_lab: Optional[np.ndarray] = None   # (M, 3)
        self._palette_ids: Optional[np.ndarray] = None   # (M,) object array of str

        self._load_palette()

    # ── Public API ─────────────────────────────────────────────────────────────

    def map(self, voxel_grid, colour_map: np.ndarray) -> MappedGrid:
        """
        Assign a Minecraft block to every occupied voxel.

        Args:
            voxel_grid:  VoxelGrid from Voxeliser.voxelise().
            colour_map:  (N_faces, 3) uint8 array — face_id → RGB colour.

        Returns:
            MappedGrid with per-voxel block ID strings.
        """
        if not self._palette:
            raise RuntimeError(
                "No blocks loaded in palette. Check assets/block_palettes/."
            )

        res      = voxel_grid.resolution
        occupied = voxel_grid.occupied
        face_ids = voxel_grid.face_ids

        logger.info(
            f"Mapping {occupied.sum():,} voxels → "
            f"{len(self._palette)} palette blocks..."
        )

        # ── 1. Unique face IDs used by occupied voxels ────────────────────────
        used_faces  = face_ids[occupied]                      # (V,)
        unique_fids = np.unique(used_faces)
        unique_fids = unique_fids[unique_fids >= 0]

        block_grid = np.full((res, res, res), "air", dtype=object)

        if len(unique_fids) == 0:
            logger.warning(
                "No occupied voxels with valid face IDs — "
                "voxelisation may have failed (check rtree / pyembree install)."
            )
            return MappedGrid(
                block_grid=block_grid,
                occupied=occupied,
                palette=self._palette,
            )

        # ── 2. RGB → LAB for each unique face colour ──────────────────────────
        face_rgb = colour_map[unique_fids].astype(np.float32)  # (U, 3)
        face_lab = _rgb_to_lab(face_rgb)                       # (U, 3)

        # ── 3. Nearest-neighbour match in LAB space ───────────────────────────
        block_indices = self._nearest_block(face_lab)          # (U,) int

        # ── 4. Build face_id → block_id lookup array ──────────────────────────
        max_fid      = int(unique_fids.max()) + 1
        fid_to_block = np.full(max_fid, "air", dtype=object)
        fid_to_block[unique_fids] = self._palette_ids[block_indices]

        # ── 5. Fill block grid — fully vectorised ─────────────────────────────
        xs, ys, zs = np.where(occupied)
        fids       = face_ids[xs, ys, zs]                      # (V,)

        valid_mask = (fids >= 0) & (fids < max_fid)
        block_grid[xs[valid_mask], ys[valid_mask], zs[valid_mask]] = \
            fid_to_block[fids[valid_mask]]

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

    def _load_palette(self) -> None:
        """Load and merge all active palette group JSON files."""
        seen_ids: set[str] = set()

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
                    self._palette.append(MinecraftBlock(
                        id=block_id,
                        display_name=entry.get("display_name", block_id),
                        rgb=tuple(entry["rgb"]),
                        groups=entry.get("groups", [group]),
                    ))
                    seen_ids.add(block_id)
                except (KeyError, TypeError) as e:
                    logger.warning(f"Skipping malformed block entry {entry}: {e}")

        if not self._palette:
            logger.warning(
                "No palette blocks loaded. Ensure assets/block_palettes/ "
                "JSON files are populated."
            )
            return

        palette_rgb        = np.array([b.rgb for b in self._palette], dtype=np.float32)
        self._palette_lab  = _rgb_to_lab(palette_rgb)
        self._palette_ids  = np.array([b.id for b in self._palette], dtype=object)

        logger.info(
            f"Loaded {len(self._palette)} blocks from groups: {self.active_groups}"
        )

    def _nearest_block(self, query_lab: np.ndarray) -> np.ndarray:
        """
        For each query colour (N, 3 LAB), return the palette index of the
        closest block by Euclidean distance in LAB space.
        """
        # (N, 1, 3) - (1, M, 3) → (N, M, 3) → (N, M) → (N,)
        diff = query_lab[:, np.newaxis, :] - self._palette_lab[np.newaxis, :, :]
        dist = (diff ** 2).sum(axis=2)   # skip sqrt — argmin is the same
        return dist.argmin(axis=1)