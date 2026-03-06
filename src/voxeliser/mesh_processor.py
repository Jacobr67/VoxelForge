"""
src/voxeliser/mesh_processor.py

Receives raw GLB bytes from TrellisClient and prepares the mesh for
voxelisation. Responsibilities:
  - Load GLB into a trimesh Scene or Mesh
  - Flatten multi-part scenes into a single merged mesh
  - Centre and normalise the mesh into a unit cube
  - Extract vertex colours / texture colours for the colour mapper
"""

import io
import logging
import numpy as np
import trimesh
from PIL import Image

logger = logging.getLogger(__name__)


class MeshProcessingError(Exception):
    """Raised when a mesh cannot be loaded or processed."""


class ProcessedMesh:
    """
    Container for a cleaned, normalised mesh ready for voxelisation.

    Attributes:
        mesh:        The merged trimesh.Trimesh object.
        colour_map:  (N, 3) uint8 array — one RGB colour per face centroid.
                     Used by the colour mapper to assign Minecraft blocks.
        bounds:      (2, 3) float array — [min_xyz, max_xyz] in world space
                     after normalisation.
        scale:       The uniform scale factor applied during normalisation.
    """

    def __init__(self, mesh: trimesh.Trimesh, colour_map: np.ndarray):
        self.mesh = mesh
        self.colour_map = colour_map          # shape (N, 3) uint8
        self.bounds = mesh.bounds             # shape (2, 3)
        self.scale = 1.0                      # set by MeshProcessor


class MeshProcessor:
    """
    Loads and prepares a GLB mesh for the voxelisation pipeline.

    Usage:
        processor = MeshProcessor()
        processed = processor.load_glb(glb_bytes)
        # processed.mesh     → trimesh.Trimesh
        # processed.colour_map → np.ndarray (N, 3) uint8
    """

    def load_glb(self, glb_bytes: bytes) -> ProcessedMesh:
        """
        Load raw GLB bytes and return a ProcessedMesh ready for voxelisation.

        Args:
            glb_bytes: Raw binary GLB data from TrellisClient.

        Returns:
            ProcessedMesh with merged geometry and per-face colour map.

        Raises:
            MeshProcessingError: If the GLB cannot be parsed or is empty.
        """
        logger.info(f"Loading GLB mesh ({len(glb_bytes) / 1024:.1f} KB)...")

        # ── 1. Parse GLB ──────────────────────────────────────────────────────
        try:
            scene_or_mesh = trimesh.load(
                io.BytesIO(glb_bytes),
                file_type="glb",
                force="scene",         # always load as scene for consistency
            )
        except Exception as e:
            raise MeshProcessingError(f"Failed to parse GLB data: {e}") from e

        # ── 2. Merge scene into a single mesh ─────────────────────────────────
        mesh = self._merge_scene(scene_or_mesh)

        if mesh is None or len(mesh.faces) == 0:
            raise MeshProcessingError(
                "GLB produced an empty mesh. The model may have no geometry."
            )

        logger.info(
            f"Merged mesh: {len(mesh.vertices)} vertices, "
            f"{len(mesh.faces)} faces."
        )

        # ── 3. Extract per-face colours before normalisation ──────────────────
        colour_map = self._extract_face_colours(mesh)

        # ── 4. Centre and normalise into unit cube ────────────────────────────
        scale = self._normalise(mesh)

        result = ProcessedMesh(mesh=mesh, colour_map=colour_map)
        result.scale = scale

        logger.info(
            f"Mesh normalised. Scale factor: {scale:.4f}. "
            f"Bounds: {mesh.bounds[0]} → {mesh.bounds[1]}"
        )

        return result

    # ── Private helpers ────────────────────────────────────────────────────────

    def _merge_scene(self, scene_or_mesh) -> trimesh.Trimesh:
        """
        Flatten a trimesh Scene (potentially many meshes) into one Trimesh.
        If it's already a Trimesh, return it directly.
        """
        if isinstance(scene_or_mesh, trimesh.Trimesh):
            return scene_or_mesh

        if isinstance(scene_or_mesh, trimesh.Scene):
            geometries = list(scene_or_mesh.geometry.values())

            if not geometries:
                raise MeshProcessingError("GLB scene contains no geometry.")

            if len(geometries) == 1:
                return geometries[0]

            # Concatenate all meshes, preserving per-mesh visuals
            merged = trimesh.util.concatenate(geometries)
            return merged

        raise MeshProcessingError(
            f"Unexpected trimesh type: {type(scene_or_mesh)}. "
            "Expected Trimesh or Scene."
        )

    def _extract_face_colours(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Build a (N_faces, 3) uint8 colour array from the mesh.

        Priority order:
          1. Per-vertex colours baked on the mesh (interpolated to face)
          2. Texture map sampled at UV face centroids
          3. Flat material diffuse colour
          4. Fallback: mid-grey (128, 128, 128)

        Returns:
            np.ndarray of shape (N_faces, 3), dtype uint8, RGB values 0-255.
        """
        n_faces = len(mesh.faces)

        # ── Option 1: Vertex colours ──────────────────────────────────────────
        if mesh.visual is not None and hasattr(mesh.visual, "vertex_colors"):
            try:
                vc = mesh.visual.vertex_colors  # (N_verts, 4) RGBA uint8
                if vc is not None and len(vc) == len(mesh.vertices):
                    # Average the 3 vertex colours per face → face colour
                    face_colours = vc[mesh.faces].mean(axis=1)[:, :3]
                    return face_colours.astype(np.uint8)
            except Exception:
                pass

        # ── Option 2: UV texture map ──────────────────────────────────────────
        if hasattr(mesh.visual, "to_color"):
            try:
                color_visual = mesh.visual.to_color()
                vc = color_visual.vertex_colors
                if vc is not None and len(vc) == len(mesh.vertices):
                    face_colours = vc[mesh.faces].mean(axis=1)[:, :3]
                    return face_colours.astype(np.uint8)
            except Exception:
                pass

        # ── Option 3: Flat material colour ────────────────────────────────────
        if hasattr(mesh.visual, "material"):
            try:
                mat = mesh.visual.material
                if hasattr(mat, "main_color"):
                    colour = np.array(mat.main_color[:3], dtype=np.uint8)
                    return np.tile(colour, (n_faces, 1))
            except Exception:
                pass

        # ── Option 4: Fallback grey ───────────────────────────────────────────
        logger.warning(
            "Could not extract colours from mesh. Using fallback grey."
        )
        return np.full((n_faces, 3), 128, dtype=np.uint8)

    def _normalise(self, mesh: trimesh.Trimesh) -> float:
        """
        Centre the mesh at the origin and scale it to fit within a unit cube.
        Modifies the mesh in-place.

        Returns:
            The uniform scale factor applied.
        """
        # Centre at origin
        centroid = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-centroid)

        # Scale longest axis to 1.0
        extents = mesh.bounds[1] - mesh.bounds[0]
        max_extent = extents.max()

        if max_extent < 1e-8:
            raise MeshProcessingError(
                "Mesh has near-zero extent — it may be degenerate."
            )

        scale = 1.0 / max_extent
        mesh.apply_scale(scale)

        return scale