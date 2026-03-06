"""
src/voxeliser/voxel_grid.py

Converts a ProcessedMesh into a 3D voxel grid using ray casting.
Each occupied voxel stores its face index so the colour mapper can
look up which Minecraft block to assign.

Ray casting approach:
  For each XZ column in the grid, fire a ray downward along Y.
  Count intersections with the mesh surface — odd count means inside,
  even means outside (Jordan curve theorem extended to 3D).
  This correctly fills solid interiors and handles complex geometry.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class VoxelisationError(Exception):
    """Raised when voxelisation fails."""


@dataclass
class VoxelGrid:
    """
    A 3D occupancy grid produced by the voxeliser.

    Attributes:
        occupied:    (X, Y, Z) bool array — True where a voxel is filled.
        face_ids:    (X, Y, Z) int32 array — face index of the mesh face
                     that determined this voxel's colour. -1 = empty.
        resolution:  The grid dimension (e.g. 64 → 64×64×64 grid).
        voxel_size:  World-space size of one voxel (after normalisation).
    """
    occupied:   np.ndarray          # (X, Y, Z) bool
    face_ids:   np.ndarray          # (X, Y, Z) int32
    resolution: int
    voxel_size: float

    @property
    def shape(self):
        return self.occupied.shape

    @property
    def voxel_count(self) -> int:
        return int(self.occupied.sum())


class Voxeliser:
    """
    Converts a ProcessedMesh into a VoxelGrid using ray casting.

    Usage:
        voxeliser = Voxeliser(resolution=64)
        grid = voxeliser.voxelise(processed_mesh)
    """

    # Sensible resolution limits to protect against accidental huge grids
    MIN_RESOLUTION = 16
    MAX_RESOLUTION = 256

    def __init__(self, resolution: int = 64):
        """
        Args:
            resolution: Number of voxels along each axis (cube grid).
                        E.g. 64 → 64×64×64 = 262,144 voxels max.
                        Maps directly to the Minecraft block count on
                        the longest axis of the structure.
        """
        if not (self.MIN_RESOLUTION <= resolution <= self.MAX_RESOLUTION):
            raise ValueError(
                f"Resolution must be between {self.MIN_RESOLUTION} and "
                f"{self.MAX_RESOLUTION}. Got: {resolution}"
            )
        self.resolution = resolution

    def voxelise(self, processed_mesh) -> VoxelGrid:
        """
        Voxelise a ProcessedMesh using ray casting.

        Args:
            processed_mesh: A ProcessedMesh from MeshProcessor.load_glb().

        Returns:
            VoxelGrid with occupied voxels and per-voxel face IDs.

        Raises:
            VoxelisationError: If voxelisation fails.
        """
        mesh = processed_mesh.mesh
        res  = self.resolution

        logger.info(
            f"Voxelising mesh at resolution {res}³ "
            f"({res**3:,} voxels max)..."
        )

        # ── 1. Build voxel grid coordinate system ─────────────────────────────
        # Mesh is normalised to unit cube [-0.5, 0.5]³
        # We map this to voxel indices [0, res)
        voxel_size = 1.0 / res

        # Voxel centre positions along each axis
        coords = np.linspace(-0.5 + voxel_size / 2,
                              0.5 - voxel_size / 2,
                              res)

        # ── 2. Initialise output arrays ───────────────────────────────────────
        occupied = np.zeros((res, res, res), dtype=bool)
        face_ids = np.full((res, res, res), -1, dtype=np.int32)

        # ── 3. Build ray intersection engine ─────────────────────────────────
        try:
            intersector = mesh.ray
        except Exception as e:
            raise VoxelisationError(
                f"Could not initialise ray intersector: {e}"
            ) from e

        # ── 4. Ray cast: for each XZ column, fire ray along +Y ───────────────
        # Ray origins: just below the mesh bottom (-0.5 - epsilon)
        ray_origin_y = -0.5 - 0.01
        ray_direction = np.array([0.0, 1.0, 0.0])

        total_columns = res * res
        logged_pct = 0

        for xi, x in enumerate(coords):
            for zi, z in enumerate(coords):

                # Progress logging every 25%
                col_idx = xi * res + zi
                pct = int(100 * col_idx / total_columns)
                if pct >= logged_pct + 25:
                    logger.info(f"  Voxelising... {pct}%")
                    logged_pct = pct

                ray_origin = np.array([[x, ray_origin_y, z]])
                ray_dir    = np.array([[0.0, 1.0, 0.0]])

                try:
                    locations, index_ray, index_tri = \
                        intersector.intersects_location(
                            ray_origins=ray_origin,
                            ray_directions=ray_dir,
                            multiple_hits=True,
                        )
                except Exception:
                    continue  # Skip degenerate columns

                if len(locations) == 0:
                    continue

                # Sort intersections by Y
                sort_order  = np.argsort(locations[:, 1])
                hit_ys      = locations[sort_order, 1]
                hit_faces   = index_tri[sort_order]

                # Jordan curve: pair up intersections → inside spans
                # Each pair (hit_ys[i], hit_ys[i+1]) is an inside region
                for pair_start in range(0, len(hit_ys) - 1, 2):
                    y_enter = hit_ys[pair_start]
                    y_exit  = hit_ys[pair_start + 1]
                    face_in = hit_faces[pair_start]

                    # Find voxel indices in this Y span
                    yi_enter = max(0, int((y_enter + 0.5) / voxel_size))
                    yi_exit  = min(res, int((y_exit  + 0.5) / voxel_size) + 1)

                    occupied[xi, yi_enter:yi_exit, zi] = True
                    face_ids[xi, yi_enter:yi_exit, zi] = face_in

                # Also mark surface voxels from all hits
                for hit_y, hit_face in zip(hit_ys, hit_faces):
                    yi = int((hit_y + 0.5) / voxel_size)
                    yi = max(0, min(res - 1, yi))
                    occupied[xi, yi, zi] = True
                    face_ids[xi, yi, zi] = hit_face

        logger.info(
            f"Voxelisation complete. "
            f"Filled voxels: {occupied.sum():,} / {res**3:,} "
            f"({100 * occupied.sum() / res**3:.1f}%)"
        )

        return VoxelGrid(
            occupied=occupied,
            face_ids=face_ids,
            resolution=res,
            voxel_size=voxel_size,
        )

    def hollow(self, grid: VoxelGrid, shell_thickness: int = 1) -> VoxelGrid:
        """
        Remove interior voxels, keeping only the outer shell.
        Useful for large structures to reduce block count in Minecraft.

        Args:
            grid:            The filled VoxelGrid to hollow.
            shell_thickness: How many voxels thick the shell should be.

        Returns:
            A new VoxelGrid with interior voxels removed.
        """
        from scipy.ndimage import binary_erosion

        logger.info(
            f"Hollowing grid (shell thickness: {shell_thickness})..."
        )

        # Erode the filled grid — anything that survives erosion is interior
        interior = binary_erosion(
            grid.occupied,
            iterations=shell_thickness,
        )

        # Shell = filled minus interior
        shell_occupied = grid.occupied & ~interior
        shell_face_ids = grid.face_ids.copy()
        shell_face_ids[interior] = -1

        removed = int(interior.sum())
        logger.info(
            f"Hollowing removed {removed:,} interior voxels. "
            f"Shell voxels: {shell_occupied.sum():,}"
        )

        return VoxelGrid(
            occupied=shell_occupied,
            face_ids=shell_face_ids,
            resolution=grid.resolution,
            voxel_size=grid.voxel_size,
        )