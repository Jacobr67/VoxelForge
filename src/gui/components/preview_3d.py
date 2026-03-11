"""
src/gui/components/preview_3d.py

Interactive 3D preview using PyQt6 QOpenGLWidget + VBOs.
All geometry is uploaded to the GPU once and rendered in a single draw call.

Modes:
    "mesh"  — renders trimesh geometry with fixed-function lighting
    "voxel" — renders voxel grid as solid-colour unit cubes
    "empty" — reference grid only

Controls:
    Left-drag:   orbit
    Right-drag:  pan
    Scroll:      zoom
"""

import logging
import math
import ctypes
import numpy as np

from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QMouseEvent, QWheelEvent

try:
    from OpenGL import GL
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

logger = logging.getLogger(__name__)


# ── Cube geometry (module-level constant, built once) ──────────────────────────
# 6 faces × 4 vertices × (x, y, z,  nx, ny, nz)
# Normals are used for the mesh draw path; voxel path uses colour instead.

_HALF = 0.5

# Face vertex positions and normals, unit cube centred at origin
# Order: +Y, -Y, +X, -X, +Z, -Z
_FACE_NORMALS = np.array([
    [ 0,  1,  0],
    [ 0, -1,  0],
    [ 1,  0,  0],
    [-1,  0,  0],
    [ 0,  0,  1],
    [ 0,  0, -1],
], dtype=np.float32)

_FACE_VERTS = np.array([
    # +Y top
    [[-_HALF,  _HALF, -_HALF], [-_HALF,  _HALF,  _HALF],
     [ _HALF,  _HALF,  _HALF], [ _HALF,  _HALF, -_HALF]],
    # -Y bottom
    [[-_HALF, -_HALF, -_HALF], [ _HALF, -_HALF, -_HALF],
     [ _HALF, -_HALF,  _HALF], [-_HALF, -_HALF,  _HALF]],
    # +X right
    [[ _HALF, -_HALF, -_HALF], [ _HALF,  _HALF, -_HALF],
     [ _HALF,  _HALF,  _HALF], [ _HALF, -_HALF,  _HALF]],
    # -X left
    [[-_HALF, -_HALF, -_HALF], [-_HALF, -_HALF,  _HALF],
     [-_HALF,  _HALF,  _HALF], [-_HALF,  _HALF, -_HALF]],
    # +Z front
    [[-_HALF, -_HALF,  _HALF], [ _HALF, -_HALF,  _HALF],
     [ _HALF,  _HALF,  _HALF], [-_HALF,  _HALF,  _HALF]],
    # -Z back
    [[-_HALF, -_HALF, -_HALF], [-_HALF,  _HALF, -_HALF],
     [ _HALF,  _HALF, -_HALF], [ _HALF, -_HALF, -_HALF]],
], dtype=np.float32)  # shape (6, 4, 3)

# Quad → 2 triangles (indices into the 4 quad verts)
_QUAD_TRI_IDX = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

# Neighbour offsets for face culling, same order as _FACE_VERTS
_NEIGHBOURS = np.array([
    [ 0,  1,  0],
    [ 0, -1,  0],
    [ 1,  0,  0],
    [-1,  0,  0],
    [ 0,  0,  1],
    [ 0,  0, -1],
], dtype=np.int32)


def _ptr(byte_offset: int) -> ctypes.c_void_p:
    """Return a ctypes void pointer for VBO byte offsets."""
    return ctypes.c_void_p(byte_offset)


class Preview3D(QOpenGLWidget):
    """
    Interactive 3D OpenGL preview.

    Call load_mesh() after API returns, load_voxels() after voxelisation,
    clear() to reset between generations.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)

        # ── Camera ─────────────────────────────────────────────────────────────
        self._yaw   = 30.0
        self._pitch = 20.0
        self._zoom  = 3.0
        self._pan_x = 0.0
        self._pan_y = 0.0

        # ── Mouse ──────────────────────────────────────────────────────────────
        self._last_mouse: QPoint | None = None
        self._mouse_button = None

        # ── Render state ───────────────────────────────────────────────────────
        self._mode = "empty"

        # ── Mesh VBOs ──────────────────────────────────────────────────────────
        self._mesh_vbo: int | None = None
        self._mesh_ibo: int | None = None
        self._mesh_idx_count = 0

        # ── Voxel VBOs ─────────────────────────────────────────────────────────
        self._voxel_vbo: int | None = None
        self._voxel_ibo: int | None = None
        self._voxel_idx_count = 0

        # ── Pending data (queued before GL context is ready) ───────────────────
        self._pending_mesh   = None
        self._pending_voxels = None

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_mesh(self, processed_mesh) -> None:
        """Queue mesh for GPU upload. Safe to call from any thread."""
        self._pending_mesh = processed_mesh
        self._mode = "mesh"
        self.update()

    def load_voxels(self, voxel_grid, colour_map: np.ndarray) -> None:
        """
        Queue voxel grid for GPU upload.
        Mode switches to 'voxel' only after the VBO is successfully written,
        so the mesh preview stays visible until voxels are ready.
        """
        self._pending_voxels = (voxel_grid, colour_map)
        self.update()

    def clear(self) -> None:
        """Reset preview to empty state."""
        self._mode = "empty"
        self._pending_mesh   = None
        self._pending_voxels = None
        self._free_vbos()
        self.update()

    # ── OpenGL lifecycle ───────────────────────────────────────────────────────

    def initializeGL(self) -> None:
        if not OPENGL_AVAILABLE:
            return
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glClearColor(0.08, 0.08, 0.10, 1.0)
        logger.debug("Preview3D GL initialised.")

    def resizeGL(self, w: int, h: int) -> None:
        if not OPENGL_AVAILABLE:
            return
        GL.glViewport(0, 0, w, max(h, 1))
        self._set_projection(w, h)

    def paintGL(self) -> None:
        if not OPENGL_AVAILABLE:
            self._paint_fallback()
            return

        # Upload any pending data now that we have a valid GL context
        if self._pending_mesh is not None:
            self._upload_mesh(self._pending_mesh)
            self._pending_mesh = None

        if self._pending_voxels is not None:
            self._upload_voxels(*self._pending_voxels)
            self._pending_voxels = None
            self._mode = "voxel"

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glLoadIdentity()

        # Apply camera transform
        GL.glTranslatef(self._pan_x, self._pan_y, -self._zoom)
        GL.glRotatef(self._pitch, 1, 0, 0)
        GL.glRotatef(self._yaw,   0, 1, 0)

        self._draw_grid()

        if self._mode == "mesh" and self._mesh_vbo:
            self._draw_mesh()
        elif self._mode == "voxel" and self._voxel_vbo:
            self._draw_voxels()

    # ── GPU upload ─────────────────────────────────────────────────────────────

    def _upload_mesh(self, processed_mesh) -> None:
        """Pack trimesh geometry into VBOs. Interleaved: pos(3) + normal(3)."""
        self._free_mesh_vbos()

        mesh  = processed_mesh.mesh
        verts = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.uint32)

        try:
            norms = mesh.vertex_normals.astype(np.float32)
        except Exception:
            norms = np.zeros_like(verts)

        interleaved = np.ascontiguousarray(
            np.hstack([verts, norms]), dtype=np.float32
        )

        self._mesh_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._mesh_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, interleaved.nbytes,
                        interleaved, GL.GL_STATIC_DRAW)

        self._mesh_ibo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._mesh_ibo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, faces.nbytes,
                        faces, GL.GL_STATIC_DRAW)

        self._mesh_idx_count = faces.size
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

        logger.info(f"Mesh VBO: {len(verts):,} verts, {len(faces):,} faces.")

    def _upload_voxels(self, voxel_grid, colour_map: np.ndarray) -> None:
        """
        Build a single interleaved VBO for all visible voxel faces.

        Interleaved layout per vertex: pos(3) + colour(3) = 6 floats.

        Optimised with NumPy — no Python loops over individual voxels.
        Hidden faces (neighbour also occupied) are culled before upload.
        """
        self._free_voxel_vbos()

        occupied = voxel_grid.occupied          # (X, Y, Z) bool
        face_ids = voxel_grid.face_ids          # (X, Y, Z) int32
        res      = voxel_grid.resolution
        scale    = 1.0 / res

        xs, ys, zs = np.where(occupied)
        n_voxels   = len(xs)
        if n_voxels == 0:
            return

        # Voxel centre positions in [-0.5, 0.5] world space
        cx = (xs + 0.5) * scale - 0.5   # (V,)
        cy = (ys + 0.5) * scale - 0.5
        cz = (zs + 0.5) * scale - 0.5
        centres = np.stack([cx, cy, cz], axis=1)  # (V, 3)

        # Per-voxel colours from face_ids → colour_map
        fids = face_ids[xs, ys, zs]                            # (V,)
        valid = (fids >= 0) & (fids < len(colour_map))
        colours = np.full((n_voxels, 3), 0.5, dtype=np.float32)
        colours[valid] = colour_map[fids[valid]].astype(np.float32) / 255.0

        all_verts   = []
        all_indices = []
        idx_offset  = 0

        for fi, (dx, dy, dz) in enumerate(_NEIGHBOURS):
            # Compute neighbour indices for all voxels at once
            nx = xs + dx
            ny = ys + dy
            nz = zs + dz

            # A face is visible if the neighbour is out-of-bounds OR unoccupied
            in_bounds = (
                (nx >= 0) & (nx < res) &
                (ny >= 0) & (ny < res) &
                (nz >= 0) & (nz < res)
            )
            neighbour_occupied = np.zeros(n_voxels, dtype=bool)
            neighbour_occupied[in_bounds] = occupied[
                nx[in_bounds], ny[in_bounds], nz[in_bounds]
            ]
            visible = ~neighbour_occupied  # (V,) mask

            n_vis = visible.sum()
            if n_vis == 0:
                continue

            # Unit-cube face verts scaled to voxel size, shape (4, 3)
            face_verts_local = _FACE_VERTS[fi] * scale  # (4, 3)

            # Broadcast: (V_vis, 1, 3) + (1, 4, 3) → (V_vis, 4, 3)
            v_centres = centres[visible][:, np.newaxis, :]   # (V_vis, 1, 3)
            quad_pos  = v_centres + face_verts_local[np.newaxis, :, :]  # (V_vis, 4, 3)

            # Repeat colour for each of 4 vertices: (V_vis, 1, 3) → (V_vis, 4, 3)
            v_colours = colours[visible][:, np.newaxis, :].repeat(4, axis=1)

            # Interleave pos and colour: (V_vis, 4, 6)
            quad_data = np.concatenate([quad_pos, v_colours], axis=2)

            # Flatten to (V_vis * 4, 6)
            all_verts.append(quad_data.reshape(-1, 6))

            # Build index array: each quad → 2 triangles
            # _QUAD_TRI_IDX offsets per quad: (V_vis, 6)
            offsets = (np.arange(n_vis, dtype=np.uint32) * 4 + idx_offset)
            quad_idx = offsets[:, np.newaxis] + _QUAD_TRI_IDX[np.newaxis, :]
            all_indices.append(quad_idx.reshape(-1))

            idx_offset += n_vis * 4

        if not all_verts:
            return

        vert_array = np.ascontiguousarray(
            np.vstack(all_verts), dtype=np.float32
        )
        idx_array = np.ascontiguousarray(
            np.concatenate(all_indices), dtype=np.uint32
        )

        self._voxel_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._voxel_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vert_array.nbytes,
                        vert_array, GL.GL_STATIC_DRAW)

        self._voxel_ibo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._voxel_ibo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idx_array.nbytes,
                        idx_array, GL.GL_STATIC_DRAW)

        self._voxel_idx_count = len(idx_array)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

        logger.info(
            f"Voxel VBO: {n_voxels:,} voxels, "
            f"{self._voxel_idx_count // 3:,} triangles."
        )

    # ── Draw calls ─────────────────────────────────────────────────────────────

    def _draw_grid(self) -> None:
        """Draw a flat reference grid on the Y=-0.5 plane."""
        GL.glLineWidth(1.0)
        GL.glColor4f(0.18, 0.18, 0.22, 0.6)
        GL.glBegin(GL.GL_LINES)
        for i in range(-5, 6):
            t = i * 0.1
            GL.glVertex3f(t,   -0.5, -0.5); GL.glVertex3f(t,   -0.5,  0.5)
            GL.glVertex3f(-0.5, -0.5,  t ); GL.glVertex3f( 0.5, -0.5,  t )
        GL.glEnd()

    def _draw_mesh(self) -> None:
        """
        Draw the raw trimesh geometry from VBO.
        Uses fixed-function lighting in compatibility mode.
        Interleaved layout: pos(3 floats) + normal(3 floats), stride = 24 bytes.
        """
        stride = 6 * 4  # 6 floats × 4 bytes

        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, [1.0, 2.0, 1.5, 0.0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE,  [0.9, 0.9, 0.9, 1.0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT,  [0.25, 0.25, 0.25, 1.0])
        GL.glColor4f(0.35, 0.60, 0.85, 1.0)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._mesh_vbo)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._mesh_ibo)

        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
        GL.glVertexPointer(3, GL.GL_FLOAT, stride, _ptr(0))
        GL.glNormalPointer(GL.GL_FLOAT,    stride, _ptr(12))

        GL.glDrawElements(GL.GL_TRIANGLES, self._mesh_idx_count,
                          GL.GL_UNSIGNED_INT, None)

        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glDisableClientState(GL.GL_NORMAL_ARRAY)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_COLOR_MATERIAL)

    def _draw_voxels(self) -> None:
        """
        Draw all visible voxel faces in a single draw call.
        Interleaved layout: pos(3 floats) + colour(3 floats), stride = 24 bytes.
        """
        stride = 6 * 4  # 6 floats × 4 bytes

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._voxel_vbo)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._voxel_ibo)

        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_COLOR_ARRAY)
        GL.glVertexPointer(3, GL.GL_FLOAT, stride, _ptr(0))
        GL.glColorPointer (3, GL.GL_FLOAT, stride, _ptr(12))

        GL.glDrawElements(GL.GL_TRIANGLES, self._voxel_idx_count,
                          GL.GL_UNSIGNED_INT, None)

        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glDisableClientState(GL.GL_COLOR_ARRAY)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    # ── Projection ─────────────────────────────────────────────────────────────

    def _set_projection(self, w: int, h: int) -> None:
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        aspect    = w / max(h, 1)
        fov_y     = 45.0
        near, far = 0.01, 100.0
        f         = 1.0 / math.tan(math.radians(fov_y / 2))
        proj = [
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) / (near - far), -1,
            0, 0, (2 * far * near) / (near - far), 0,
        ]
        GL.glLoadMatrixf(proj)
        GL.glMatrixMode(GL.GL_MODELVIEW)

    # ── VBO cleanup ────────────────────────────────────────────────────────────

    def _free_mesh_vbos(self) -> None:
        if self._mesh_vbo is not None:
            GL.glDeleteBuffers(1, [self._mesh_vbo])
            self._mesh_vbo = None
        if self._mesh_ibo is not None:
            GL.glDeleteBuffers(1, [self._mesh_ibo])
            self._mesh_ibo = None
        self._mesh_idx_count = 0

    def _free_voxel_vbos(self) -> None:
        if self._voxel_vbo is not None:
            GL.glDeleteBuffers(1, [self._voxel_vbo])
            self._voxel_vbo = None
        if self._voxel_ibo is not None:
            GL.glDeleteBuffers(1, [self._voxel_ibo])
            self._voxel_ibo = None
        self._voxel_idx_count = 0

    def _free_vbos(self) -> None:
        self._free_mesh_vbos()
        self._free_voxel_vbos()

    def closeEvent(self, event) -> None:
        self.makeCurrent()
        self._free_vbos()
        self.doneCurrent()
        super().closeEvent(event)

    # ── Fallback (no OpenGL) ───────────────────────────────────────────────────

    def _paint_fallback(self) -> None:
        from PyQt6.QtGui import QPainter
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        painter.setPen(Qt.GlobalColor.darkGray)
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignCenter,
            "3D Preview\n(OpenGL unavailable)",
        )
        painter.end()

    # ── Mouse / scroll ─────────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._last_mouse   = event.pos()
        self._mouse_button = event.button()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._last_mouse   = None
        self._mouse_button = None

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_mouse is None:
            return
        dx = event.pos().x() - self._last_mouse.x()
        dy = event.pos().y() - self._last_mouse.y()
        self._last_mouse = event.pos()

        if self._mouse_button == Qt.MouseButton.LeftButton:
            self._yaw   += dx * 0.4
            self._pitch += dy * 0.4
            self._pitch  = max(-89.0, min(89.0, self._pitch))
        elif self._mouse_button == Qt.MouseButton.RightButton:
            self._pan_x += dx * 0.002
            self._pan_y -= dy * 0.002

        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        self._zoom -= event.angleDelta().y() * 0.002
        self._zoom  = max(0.2, min(10.0, self._zoom))
        self.update()