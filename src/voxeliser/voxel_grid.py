"""
src/voxeliser/voxel_grid.py

Converts a ProcessedMesh into a 3D voxel grid using trimesh's built-in
surface voxelization. This approach has no optional dependencies (no rtree,
no pyembree) and is reliable across all platforms.

Voxelisation approach:
  trimesh.voxel.creation.voxelize_surface() casts rays along all three
  axes and fills the surface shell. The interior is then flood-filled using
  scipy binary_fill_holes so solid meshes come out solid, not hollow shells.
  Per-voxel face IDs are assigned by nearest-face lookup after voxelisation.
"""

import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class VoxelisationError(Exception):
    """Raised when voxelisation fails."""


@dataclass
class VoxelGrid:
    """
    A 3D occupancy grid produced by the voxeliser.

    Attributes:
        occupied:    (X, Y, Z) bool array — True where a voxel is filled.
        face_ids:    (X, Y, Z) int32 array — nearest face index per voxel.
                     Used by ColourMapper to assign block colours. -1 = empty.
        resolution:  Grid dimension (e.g. 64 -> 64x64x64).
        voxel_size:  World-space size of one voxel.
    """
    occupied:   np.ndarray   # (X, Y, Z) bool
    face_ids:   np.ndarray   # (X, Y, Z) int32
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
    Converts a ProcessedMesh into a VoxelGrid using trimesh voxelization.

    Usage:
        voxeliser = Voxeliser(resolution=64)
        grid = voxeliser.voxelise(processed_mesh)
    """

    MIN_RESOLUTION = 16
    MAX_RESOLUTION = 256

    def __init__(self, resolution: int = 64):
        if not (self.MIN_RESOLUTION <= resolution <= self.MAX_RESOLUTION):
            raise ValueError(
                f"Resolution must be {self.MIN_RESOLUTION}-"
                f"{self.MAX_RESOLUTION}. Got: {resolution}"
            )
        self.resolution = resolution

    def voxelise(self, processed_mesh) -> VoxelGrid:
        """
        Voxelise a ProcessedMesh using trimesh surface voxelization.

        Args:
            processed_mesh: A ProcessedMesh from MeshProcessor.load_glb().
                            Mesh must be normalised to unit cube [-0.5, 0.5]^3.

        Returns:
            VoxelGrid with occupied voxels and per-voxel face IDs.

        Raises:
            VoxelisationError: If voxelisation fails or produces 0 voxels.
        """
        from scipy.ndimage import binary_fill_holes

        mesh = processed_mesh.mesh
        res  = self.resolution

        logger.info(f"Voxelising {res}^3 ({res**3:,} voxels max)...")

        voxel_size = 1.0 / res

        # ---- 1. Surface voxelization via trimesh ----------------------------
        # mesh.voxelized() uses trimesh's own ray casting with no rtree dep.
        # pitch = world-space edge length of one voxel.
        try:
            vox = mesh.voxelized(pitch=voxel_size)
        except Exception as e:
            raise VoxelisationError(f"trimesh voxelization failed: {e}") from e

        # ---- 2. Convert sparse result to dense boolean grid -----------------
        try:
            surface = vox.matrix.astype(bool)   # (Nx, Ny, Nz)
        except Exception as e:
            raise VoxelisationError(f"Could not extract voxel matrix: {e}") from e

        # ---- 3. Flood-fill interior ------------------------------------------
        # voxelized() gives surface shell only; fill_holes makes it solid.
        filled = binary_fill_holes(surface)   # (Nx, Ny, Nz) bool

        # ---- 4. Paste into target (res, res, res) grid -----------------------
        # trimesh may return a slightly different grid size depending on pitch
        # rounding. We align by extracting the translation from vox.transform.
        # In trimesh 4.x VoxelGrid.transform is a 4x4 matrix; the origin
        # (world coords of voxel index [0,0,0]) is its translation column.
        occupied = np.zeros((res, res, res), dtype=bool)

        try:
            origin = np.array(vox.transform[:3, 3])   # trimesh 4.x
        except AttributeError:
            origin = np.array(vox.origin)             # older trimesh fallback
        offset = np.round((origin + 0.5) / voxel_size).astype(int)  # (3,)

        nx, ny, nz = filled.shape
        x0 = max(0, offset[0]);  x1 = min(res, offset[0] + nx)
        y0 = max(0, offset[1]);  y1 = min(res, offset[1] + ny)
        z0 = max(0, offset[2]);  z1 = min(res, offset[2] + nz)

        sx0 = max(0, -offset[0]); sx1 = sx0 + (x1 - x0)
        sy0 = max(0, -offset[1]); sy1 = sy0 + (y1 - y0)
        sz0 = max(0, -offset[2]); sz1 = sz0 + (z1 - z0)

        if x1 > x0 and y1 > y0 and z1 > z0:
            occupied[x0:x1, y0:y1, z0:z1] = filled[sx0:sx1, sy0:sy1, sz0:sz1]

        n_filled = int(occupied.sum())
        logger.info(
            f"Voxelisation complete. "
            f"{n_filled:,} / {res**3:,} voxels filled "
            f"({100 * n_filled / res**3:.1f}%)"
        )

        if n_filled == 0:
            raise VoxelisationError(
                "Voxelisation produced 0 filled voxels. "
                "The mesh may be degenerate or too small for this resolution."
            )

        # ---- 5. Assign face IDs via KD-tree on face centroids ---------------
        face_ids = self._assign_face_ids(mesh, occupied, voxel_size)

        return VoxelGrid(
            occupied=occupied,
            face_ids=face_ids,
            resolution=res,
            voxel_size=voxel_size,
        )

    def hollow(self, grid: VoxelGrid, shell_thickness: int = 1) -> VoxelGrid:
        """
        Remove interior voxels, keeping only the outer shell.

        Args:
            grid:            The filled VoxelGrid to hollow.
            shell_thickness: How many voxels thick the shell should be.

        Returns:
            A new VoxelGrid with interior voxels removed.
        """
        from scipy.ndimage import binary_erosion

        logger.info(f"Hollowing (shell thickness: {shell_thickness})...")

        interior       = binary_erosion(grid.occupied, iterations=shell_thickness)
        shell_occupied = grid.occupied & ~interior
        shell_face_ids = grid.face_ids.copy()
        shell_face_ids[interior] = -1

        logger.info(
            f"Removed {int(interior.sum()):,} interior voxels. "
            f"Shell: {shell_occupied.sum():,} voxels."
        )

        return VoxelGrid(
            occupied=shell_occupied,
            face_ids=shell_face_ids,
            resolution=grid.resolution,
            voxel_size=grid.voxel_size,
        )

    # ---- Private -------------------------------------------------------------

    def _assign_face_ids(
        self,
        mesh,
        occupied: np.ndarray,
        voxel_size: float,
    ) -> np.ndarray:
        """
        For every occupied voxel, find the nearest mesh face centroid and
        store that face index. ColourMapper uses this to look up the colour.

        Uses scipy cKDTree over face centroids — no rtree required.
        """
        from scipy.spatial import cKDTree

        res      = occupied.shape[0]
        face_ids = np.full((res, res, res), -1, dtype=np.int32)

        xs, ys, zs = np.where(occupied)
        vox_centres = np.column_stack([
            (xs + 0.5) * voxel_size - 0.5,
            (ys + 0.5) * voxel_size - 0.5,
            (zs + 0.5) * voxel_size - 0.5,
        ])  # (V, 3)

        face_centroids = mesh.triangles_center   # (F, 3)

        logger.info(
            f"Assigning face IDs: {len(xs):,} voxels, "
            f"{len(face_centroids):,} faces via KD-tree..."
        )

        tree           = cKDTree(face_centroids)
        _, nearest_idx = tree.query(vox_centres, workers=-1)

        face_ids[xs, ys, zs] = nearest_idx.astype(np.int32)
        return face_ids