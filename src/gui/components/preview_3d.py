"""
src/gui/components/preview_3d.py

Embedded 3D preview widget using PyQt6's QOpenGLWidget.
Renders the Trellis mesh (wireframe + shaded) and the voxelised
block grid using OpenGL. Supports orbit (left-drag), pan (right-drag),
and scroll-to-zoom.

Two render modes toggled by the main window:
  - MESH:  Renders the trimesh as a shaded + wireframe overlay
  - VOXEL: Renders the voxel grid as coloured cubes
"""

import logging
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


class Preview3D(QOpenGLWidget):
    """
    Interactive 3D OpenGL preview.

    Modes:
        "mesh"  — renders trimesh geometry
        "voxel" — renders voxel grid as unit cubes
        "empty" — shows a placeholder grid
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)

        # Camera state
        self._yaw   = 30.0    # degrees
        self._pitch = 20.0    # degrees
        self._zoom  = 3.0     # distance from origin
        self._pan_x = 0.0
        self._pan_y = 0.0

        # Mouse tracking
        self._last_mouse: QPoint | None = None
        self._mouse_button: Qt.MouseButton | None = None

        # Render data
        self._mode       = "empty"
        self._mesh_verts  = None   # (N, 3) float32 — vertex positions
        self._mesh_norms  = None   # (N, 3) float32 — per-vertex normals
        self._mesh_faces  = None   # (M, 3) int32   — face indices
        self._voxel_data  = None   # list of (x, y, z, r, g, b) tuples

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_mesh(self, processed_mesh):
        """
        Load a ProcessedMesh for display in mesh mode.

        Args:
            processed_mesh: ProcessedMesh from MeshProcessor.
        """
        mesh = processed_mesh.mesh

        self._mesh_verts = mesh.vertices.astype(np.float32)
        self._mesh_faces = mesh.faces.astype(np.int32)

        # Compute smooth vertex normals
        try:
            self._mesh_norms = mesh.vertex_normals.astype(np.float32)
        except Exception:
            self._mesh_norms = np.zeros_like(self._mesh_verts)

        self._mode = "mesh"
        self.update()
        logger.info(
            f"Preview loaded mesh: {len(self._mesh_verts)} verts, "
            f"{len(self._mesh_faces)} faces."
        )

    def load_voxels(self, voxel_grid, colour_map: np.ndarray):
        """
        Load a VoxelGrid for display in voxel mode.

        Args:
            voxel_grid: VoxelGrid from Voxeliser.
            colour_map: (N_faces, 3) uint8 colour per face from ProcessedMesh.
        """
        occupied = voxel_grid.occupied
        face_ids = voxel_grid.face_ids
        res      = voxel_grid.resolution

        xs, ys, zs = np.where(occupied)
        voxel_list = []

        for x, y, z in zip(xs, ys, zs):
            fid = int(face_ids[x, y, z])
            if fid >= 0 and fid < len(colour_map):
                r, g, b = colour_map[fid]
            else:
                r, g, b = 128, 128, 128

            # Normalise position to [-0.5, 0.5]
            nx = x / res - 0.5
            ny = y / res - 0.5
            nz = z / res - 0.5
            voxel_list.append((nx, ny, nz, r / 255, g / 255, b / 255))

        self._voxel_data = voxel_list
        self._mode = "voxel"
        self.update()
        logger.info(f"Preview loaded {len(voxel_list)} voxels.")

    def clear(self):
        """Reset to empty state."""
        self._mode      = "empty"
        self._mesh_verts = None
        self._mesh_faces = None
        self._voxel_data = None
        self.update()

    # ── OpenGL lifecycle ───────────────────────────────────────────────────────

    def initializeGL(self):
        if not OPENGL_AVAILABLE:
            return
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glClearColor(0.08, 0.08, 0.10, 1.0)

    def resizeGL(self, w, h):
        if not OPENGL_AVAILABLE:
            return
        GL.glViewport(0, 0, w, max(h, 1))
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        aspect = w / max(h, 1)
        # Simple perspective
        import math
        fov_y   = 45.0
        near, far = 0.01, 100.0
        f = 1.0 / math.tan(math.radians(fov_y / 2))
        proj = [
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) / (near - far), -1,
            0, 0, (2 * far * near) / (near - far), 0,
        ]
        GL.glLoadMatrixf(proj)
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def paintGL(self):
        if not OPENGL_AVAILABLE:
            self._paint_fallback()
            return

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glLoadIdentity()

        # Apply camera transform
        import math
        GL.glTranslatef(self._pan_x, self._pan_y, -self._zoom)
        GL.glRotatef(self._pitch, 1, 0, 0)
        GL.glRotatef(self._yaw,   0, 1, 0)

        self._draw_grid()

        if self._mode == "mesh" and self._mesh_verts is not None:
            self._draw_mesh()
        elif self._mode == "voxel" and self._voxel_data:
            self._draw_voxels()

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _draw_grid(self):
        """Draw a subtle reference grid on the XZ plane."""
        GL.glLineWidth(1.0)
        GL.glColor4f(0.18, 0.18, 0.22, 0.6)
        GL.glBegin(GL.GL_LINES)
        for i in range(-5, 6):
            t = i * 0.1
            GL.glVertex3f(t, -0.5, -0.5)
            GL.glVertex3f(t, -0.5,  0.5)
            GL.glVertex3f(-0.5, -0.5, t)
            GL.glVertex3f( 0.5, -0.5, t)
        GL.glEnd()

    def _draw_mesh(self):
        """Render mesh as flat-shaded triangles + wireframe overlay."""
        verts = self._mesh_verts
        faces = self._mesh_faces
        norms = self._mesh_norms

        # Shaded pass
        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, [1.0, 2.0, 1.5, 0.0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE,  [0.9, 0.9, 0.9, 1.0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT,  [0.2, 0.2, 0.2, 1.0])
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glColor4f(0.35, 0.60, 0.85, 1.0)

        GL.glBegin(GL.GL_TRIANGLES)
        for face in faces:
            for vi in face:
                GL.glNormal3fv(norms[vi])
                GL.glVertex3fv(verts[vi])
        GL.glEnd()
        GL.glDisable(GL.GL_LIGHTING)

        # Wireframe overlay
        GL.glPolygonOffset(1.0, 1.0)
        GL.glEnable(GL.GL_POLYGON_OFFSET_LINE)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        GL.glLineWidth(0.5)
        GL.glColor4f(0.15, 0.35, 0.55, 0.4)
        GL.glBegin(GL.GL_TRIANGLES)
        for face in faces:
            for vi in face:
                GL.glVertex3fv(verts[vi])
        GL.glEnd()
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glDisable(GL.GL_POLYGON_OFFSET_LINE)

    def _draw_voxels(self):
        """Render each voxel as a coloured unit cube."""
        s = 0.9 / max(
            1,
            round(1.0 / self._voxel_data[0][0]) if self._voxel_data else 64
        ) if self._voxel_data else 0.01
        # Use a fixed small size per voxel
        s = 0.012

        GL.glBegin(GL.GL_QUADS)
        for (x, y, z, r, g, b) in self._voxel_data:
            GL.glColor3f(r, g, b)
            self._draw_cube(x, y, z, s)
        GL.glEnd()

    def _draw_cube(self, x, y, z, s):
        """Draw a single axis-aligned cube at (x, y, z) with half-size s."""
        # 6 faces × 4 vertices, using immediate mode for simplicity
        faces_verts = [
            # +Y top
            [(-1,-1, 1),( 1,-1, 1),( 1, 1, 1),(-1, 1, 1)],
            # -Y bottom
            [(-1,-1,-1),(-1, 1,-1),( 1, 1,-1),( 1,-1,-1)],
            # +X right
            [( 1,-1,-1),( 1, 1,-1),( 1, 1, 1),( 1,-1, 1)],
            # -X left
            [(-1,-1,-1),(-1,-1, 1),(-1, 1, 1),(-1, 1,-1)],
            # +Z front
            [(-1, 1,-1),(-1, 1, 1),( 1, 1, 1),( 1, 1,-1)],
            # -Z back
            [(-1,-1,-1),( 1,-1,-1),( 1,-1, 1),(-1,-1, 1)],
        ]
        for face in faces_verts:
            for (fx, fy, fz) in face:
                GL.glVertex3f(x + fx * s, y + fy * s, z + fz * s)

    def _paint_fallback(self):
        """Render a text placeholder if OpenGL is unavailable."""
        from PyQt6.QtGui import QPainter
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        painter.setPen(Qt.GlobalColor.darkGray)
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignCenter,
            "3D Preview\n(OpenGL unavailable)"
        )
        painter.end()

    # ── Mouse / scroll input ───────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent):
        self._last_mouse  = event.pos()
        self._mouse_button = event.button()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._last_mouse   = None
        self._mouse_button = None

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._last_mouse is None:
            return
        dx = event.pos().x() - self._last_mouse.x()
        dy = event.pos().y() - self._last_mouse.y()
        self._last_mouse = event.pos()

        if self._mouse_button == Qt.MouseButton.LeftButton:
            # Orbit
            self._yaw   += dx * 0.4
            self._pitch += dy * 0.4
            self._pitch  = max(-89, min(89, self._pitch))
        elif self._mouse_button == Qt.MouseButton.RightButton:
            # Pan
            self._pan_x += dx * 0.002
            self._pan_y -= dy * 0.002

        self.update()

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        self._zoom -= delta * 0.002
        self._zoom  = max(0.2, min(10.0, self._zoom))
        self.update()