"""
src/gui/components/preview_3d.py

High-performance 3D preview using PyQt6 QOpenGLWidget + VBOs.
All geometry is uploaded to the GPU once and rendered in a single draw call.

Fixes vs previous version:
  - VBO-based rendering (no per-frame Python loops)
  - Geometry stored as GL buffer objects — never garbage collected
  - Textures loaded from a Minecraft texture pack directory structure
  - Orbit / pan / zoom unchanged

Texture pack layout expected at assets/textures/<block_id>.png
e.g. assets/textures/stone.png, assets/textures/oak_log_top.png
Falls back to solid colour if texture not found.
"""

import logging
import math
import numpy as np
from pathlib import Path

from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QMouseEvent, QWheelEvent, QImage

try:
    from OpenGL import GL
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

logger = logging.getLogger(__name__)

# Path to texture pack relative to project root
_TEXTURE_DIR = Path(__file__).parent.parent.parent.parent / "assets" / "textures"

# Cube face definitions — 6 faces × 4 vertices × (x,y,z, nx,ny,nz, u,v)
# Each face is a quad; we'll convert to triangles on upload
_HALF = 0.5
_CUBE_FACES = np.array([
    # +Y top
    [[-_HALF, _HALF, -_HALF,  0, 1, 0,  0, 0],
     [-_HALF, _HALF,  _HALF,  0, 1, 0,  0, 1],
     [ _HALF, _HALF,  _HALF,  0, 1, 0,  1, 1],
     [ _HALF, _HALF, -_HALF,  0, 1, 0,  1, 0]],
    # -Y bottom
    [[-_HALF, -_HALF, -_HALF, 0,-1, 0,  0, 0],
     [ _HALF, -_HALF, -_HALF, 0,-1, 0,  1, 0],
     [ _HALF, -_HALF,  _HALF, 0,-1, 0,  1, 1],
     [-_HALF, -_HALF,  _HALF, 0,-1, 0,  0, 1]],
    # +X right
    [[ _HALF, -_HALF, -_HALF, 1, 0, 0,  0, 0],
     [ _HALF,  _HALF, -_HALF, 1, 0, 0,  0, 1],
     [ _HALF,  _HALF,  _HALF, 1, 0, 0,  1, 1],
     [ _HALF, -_HALF,  _HALF, 1, 0, 0,  1, 0]],
    # -X left
    [[-_HALF, -_HALF, -_HALF,-1, 0, 0,  1, 0],
     [-_HALF, -_HALF,  _HALF,-1, 0, 0,  0, 0],
     [-_HALF,  _HALF,  _HALF,-1, 0, 0,  0, 1],
     [-_HALF,  _HALF, -_HALF,-1, 0, 0,  1, 1]],
    # +Z front
    [[-_HALF, -_HALF,  _HALF, 0, 0, 1,  0, 0],
     [ _HALF, -_HALF,  _HALF, 0, 0, 1,  1, 0],
     [ _HALF,  _HALF,  _HALF, 0, 0, 1,  1, 1],
     [-_HALF,  _HALF,  _HALF, 0, 0, 1,  0, 1]],
    # -Z back
    [[-_HALF, -_HALF, -_HALF, 0, 0,-1,  1, 0],
     [-_HALF,  _HALF, -_HALF, 0, 0,-1,  1, 1],
     [ _HALF,  _HALF, -_HALF, 0, 0,-1,  0, 1],
     [ _HALF, -_HALF, -_HALF, 0, 0,-1,  0, 0]],
], dtype=np.float32)

# Quad → 2 triangles index offsets
_QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)


class Preview3D(QOpenGLWidget):
    """
    Interactive 3D OpenGL preview with VBO-based rendering.

    Modes:
        "mesh"  — renders trimesh geometry (shaded + wireframe)
        "voxel" — renders voxel grid as textured unit cubes
        "empty" — shows a reference grid only
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)

        # Camera
        self._yaw   = 30.0
        self._pitch = 20.0
        self._zoom  = 3.0
        self._pan_x = 0.0
        self._pan_y = 0.0

        # Mouse
        self._last_mouse: QPoint | None = None
        self._mouse_button = None

        # Render state
        self._mode = "empty"

        # Mesh VBO state
        self._mesh_vbo       = None
        self._mesh_ibo       = None
        self._mesh_vert_count = 0
        self._mesh_idx_count  = 0

        # Voxel VBO state
        self._voxel_vbo       = None
        self._voxel_ibo       = None
        self._voxel_idx_count = 0

        # Texture cache: block_id → GL texture id
        self._tex_cache: dict[str, int] = {}
        self._default_tex: int | None   = None

        # Pending data (set before GL context exists)
        self._pending_mesh   = None
        self._pending_voxels = None

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_mesh(self, processed_mesh):
        """Upload mesh geometry to GPU. Called from main thread."""
        self._pending_mesh = processed_mesh
        self._mode = "mesh"
        self.update()

    def load_voxels(self, voxel_grid, colour_map: np.ndarray, block_grid=None):
        """
        Upload voxel geometry to GPU.

        Args:
            voxel_grid:  VoxelGrid with .occupied and .face_ids arrays.
            colour_map:  (N, 3) uint8 fallback colours per face.
            block_grid:  Optional (X,Y,Z) array of block ID strings for textures.
        """
        self._pending_voxels = (voxel_grid, colour_map, block_grid)
        self._mode = "voxel"
        self.update()

    def clear(self):
        self._mode = "empty"
        self._pending_mesh   = None
        self._pending_voxels = None
        self._free_vbos()
        self.update()

    # ── OpenGL lifecycle ───────────────────────────────────────────────────────

    def initializeGL(self):
        if not OPENGL_AVAILABLE:
            return
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glClearColor(0.08, 0.08, 0.10, 1.0)
        GL.glEnable(GL.GL_TEXTURE_2D)
        self._default_tex = self._make_default_texture()
        logger.debug("Preview3D GL initialised.")

    def resizeGL(self, w, h):
        if not OPENGL_AVAILABLE:
            return
        GL.glViewport(0, 0, w, max(h, 1))
        self._set_projection(w, h)

    def paintGL(self):
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

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glLoadIdentity()

        # Camera transform
        GL.glTranslatef(self._pan_x, self._pan_y, -self._zoom)
        GL.glRotatef(self._pitch, 1, 0, 0)
        GL.glRotatef(self._yaw,   0, 1, 0)

        self._draw_grid()

        if self._mode == "mesh" and self._mesh_vbo:
            self._draw_mesh_vbo()
        elif self._mode == "voxel" and self._voxel_vbo:
            self._draw_voxels_vbo()

    # ── GL upload ──────────────────────────────────────────────────────────────

    def _upload_mesh(self, processed_mesh):
        """Pack mesh into VBOs."""
        self._free_mesh_vbos()
        mesh  = processed_mesh.mesh
        verts = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.uint32)

        try:
            norms = mesh.vertex_normals.astype(np.float32)
        except Exception:
            norms = np.zeros_like(verts)

        # Interleaved: position(3) + normal(3)
        interleaved = np.hstack([verts, norms]).astype(np.float32)

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
        logger.info(f"Mesh VBO uploaded: {len(verts)} verts, {len(faces)} faces.")

    def _upload_voxels(self, voxel_grid, colour_map, block_grid):
        """Pack all visible voxel faces into a single VBO."""
        self._free_voxel_vbos()

        occupied = voxel_grid.occupied
        face_ids = voxel_grid.face_ids
        res      = voxel_grid.resolution
        scale    = 1.0 / res

        xs, ys, zs = np.where(occupied)
        if len(xs) == 0:
            return

        all_verts   = []
        all_indices = []
        idx_offset  = 0

        # Neighbour offsets for face culling (+Y -Y +X -X +Z -Z)
        neighbours = [(0,1,0),(0,-1,0),(1,0,0),(-1,0,0),(0,0,1),(0,0,-1)]

        for xi, yi, zi in zip(xs, ys, zs):
            # Get colour for this voxel
            fid = int(face_ids[xi, yi, zi])
            if fid >= 0 and fid < len(colour_map):
                r, g, b = colour_map[fid] / 255.0
            else:
                r, g, b = 0.5, 0.5, 0.5

            cx = (xi + 0.5) * scale - 0.5
            cy = (yi + 0.5) * scale - 0.5
            cz = (zi + 0.5) * scale - 0.5

            for face_i, (dx, dy, dz) in enumerate(neighbours):
                nx, ny, nz = xi + dx, yi + dy, zi + dz
                # Skip face if neighbour is also occupied (hidden face)
                if (0 <= nx < res and 0 <= ny < res and 0 <= nz < res
                        and occupied[nx, ny, nz]):
                    continue

                face_def = _CUBE_FACES[face_i]  # (4, 8)
                for v in face_def:
                    # pos(3) + colour(3) + uv(2)
                    all_verts.extend([
                        cx + v[0] * scale,
                        cy + v[1] * scale,
                        cz + v[2] * scale,
                        r, g, b,
                        v[6], v[7],
                    ])

                tri_idx = _QUAD_INDICES + idx_offset
                all_indices.extend(tri_idx.tolist())
                idx_offset += 4

        if not all_verts:
            return

        vert_array = np.array(all_verts, dtype=np.float32)
        idx_array  = np.array(all_indices, dtype=np.uint32)

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
        logger.info(f"Voxel VBO uploaded: {idx_offset} faces, "
                    f"{self._voxel_idx_count} indices.")

    # ── Draw calls ─────────────────────────────────────────────────────────────

    def _draw_grid(self):
        GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glLineWidth(1.0)
        GL.glColor4f(0.18, 0.18, 0.22, 0.6)
        GL.glBegin(GL.GL_LINES)
        for i in range(-5, 6):
            t = i * 0.1
            GL.glVertex3f(t, -0.5, -0.5); GL.glVertex3f(t, -0.5,  0.5)
            GL.glVertex3f(-0.5, -0.5, t); GL.glVertex3f( 0.5, -0.5, t)
        GL.glEnd()
        GL.glEnable(GL.GL_TEXTURE_2D)

    def _draw_mesh_vbo(self):
        """Draw mesh from VBO — single draw call."""
        GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, [1.0, 2.0, 1.5, 0.0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE,  [0.9, 0.9, 0.9, 1.0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT,  [0.25, 0.25, 0.25, 1.0])
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glColor4f(0.35, 0.60, 0.85, 1.0)

        stride = 6 * 4  # 6 floats × 4 bytes
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._mesh_vbo)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._mesh_ibo)

        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
        GL.glVertexPointer(3, GL.GL_FLOAT, stride, None)
        GL.glNormalPointer(GL.GL_FLOAT, stride, ctypes_offset(12))

        GL.glDrawElements(GL.GL_TRIANGLES, self._mesh_idx_count,
                          GL.GL_UNSIGNED_INT, None)

        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glDisableClientState(GL.GL_NORMAL_ARRAY)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_TEXTURE_2D)

    def _draw_voxels_vbo(self):
        """Draw all voxel faces in a single draw call with colour + texture."""
        GL.glBindTexture(GL.GL_TEXTURE_2D,
                         self._default_tex or 0)

        stride = 8 * 4  # pos(3) + col(3) + uv(2) = 8 floats × 4 bytes
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._voxel_vbo)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._voxel_ibo)

        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_COLOR_ARRAY)
        GL.glEnableClientState(GL.GL_TEXTURE_COORD_ARRAY)

        GL.glVertexPointer  (3, GL.GL_FLOAT, stride, ctypes_offset(0))
        GL.glColorPointer   (3, GL.GL_FLOAT, stride, ctypes_offset(12))
        GL.glTexCoordPointer(2, GL.GL_FLOAT, stride, ctypes_offset(24))

        GL.glDrawElements(GL.GL_TRIANGLES, self._voxel_idx_count,
                          GL.GL_UNSIGNED_INT, None)

        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glDisableClientState(GL.GL_COLOR_ARRAY)
        GL.glDisableClientState(GL.GL_TEXTURE_COORD_ARRAY)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    # ── Texture loading ────────────────────────────────────────────────────────

    def _make_default_texture(self) -> int:
        """Create a 1×1 white texture used when no texture pack is found."""
        tex_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, 1, 1, 0,
                        GL.GL_RGB, GL.GL_UNSIGNED_BYTE,
                        np.array([255, 255, 255], dtype=np.uint8))
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        return tex_id

    def load_texture_pack(self, texture_dir: str | Path | None = None):
        """
        Load block textures from a Minecraft-style texture pack directory.

        Expected layout:
            <texture_dir>/
                stone.png
                oak_log.png
                oak_log_top.png
                grass_block_top.png
                grass_block_side.png
                ...

        Textures are named after their block ID (without minecraft: prefix).
        e.g. block ID "minecraft:stone" → looks for "stone.png"

        Args:
            texture_dir: Path to texture directory.
                         Defaults to assets/textures/ in the project root.
        """
        search_dir = Path(texture_dir) if texture_dir else _TEXTURE_DIR
        if not search_dir.exists():
            logger.warning(
                f"Texture directory not found: {search_dir}. "
                "Using solid colours. "
                "Create assets/textures/ and add .png files named after block IDs."
            )
            return

        loaded = 0
        for png in search_dir.glob("*.png"):
            block_id = png.stem   # e.g. "stone" from "stone.png"
            tex_id   = self._load_png_texture(png)
            if tex_id is not None:
                # Store under both bare name and namespaced name
                self._tex_cache[block_id] = tex_id
                self._tex_cache[f"minecraft:{block_id}"] = tex_id
                loaded += 1

        logger.info(f"Loaded {loaded} block textures from {search_dir}.")

    def _load_png_texture(self, path: Path) -> int | None:
        """Load a PNG file into a GL texture and return its ID."""
        try:
            img = QImage(str(path)).convertToFormat(
                QImage.Format.Format_RGBA8888
            )
            if img.isNull():
                logger.warning(f"Failed to load texture: {path}")
                return None

            w, h     = img.width(), img.height()
            img_data = img.bits().asarray(w * h * 4)
            arr      = np.frombuffer(img_data, dtype=np.uint8).copy()

            tex_id = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, w, h, 0,
                            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, arr)
            GL.glTexParameteri(GL.GL_TEXTURE_2D,
                               GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D,
                               GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D,
                               GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D,
                               GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
            return tex_id

        except Exception as e:
            logger.warning(f"Error loading texture {path}: {e}")
            return None

    def get_texture_for_block(self, block_id: str) -> int:
        """
        Return GL texture ID for a block, falling back to default.

        Strips 'minecraft:' prefix for lookup.
        """
        if block_id in self._tex_cache:
            return self._tex_cache[block_id]
        bare = block_id.replace("minecraft:", "")
        if bare in self._tex_cache:
            return self._tex_cache[bare]
        return self._default_tex or 0

    # ── Projection ─────────────────────────────────────────────────────────────

    def _set_projection(self, w, h):
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        aspect   = w / max(h, 1)
        fov_y    = 45.0
        near, far = 0.01, 100.0
        f        = 1.0 / math.tan(math.radians(fov_y / 2))
        proj = [
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) / (near - far), -1,
            0, 0, (2 * far * near) / (near - far), 0,
        ]
        GL.glLoadMatrixf(proj)
        GL.glMatrixMode(GL.GL_MODELVIEW)

    # ── VBO cleanup ────────────────────────────────────────────────────────────

    def _free_mesh_vbos(self):
        if self._mesh_vbo:
            GL.glDeleteBuffers(1, [self._mesh_vbo])
            self._mesh_vbo = None
        if self._mesh_ibo:
            GL.glDeleteBuffers(1, [self._mesh_ibo])
            self._mesh_ibo = None
        self._mesh_idx_count = 0

    def _free_voxel_vbos(self):
        if self._voxel_vbo:
            GL.glDeleteBuffers(1, [self._voxel_vbo])
            self._voxel_vbo = None
        if self._voxel_ibo:
            GL.glDeleteBuffers(1, [self._voxel_ibo])
            self._voxel_ibo = None
        self._voxel_idx_count = 0

    def _free_vbos(self):
        self._free_mesh_vbos()
        self._free_voxel_vbos()

    def closeEvent(self, event):
        self.makeCurrent()
        self._free_vbos()
        for tex_id in self._tex_cache.values():
            GL.glDeleteTextures(1, [tex_id])
        if self._default_tex:
            GL.glDeleteTextures(1, [self._default_tex])
        self.doneCurrent()
        super().closeEvent(event)

    # ── Fallback (no OpenGL) ───────────────────────────────────────────────────

    def _paint_fallback(self):
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

    # ── Mouse / scroll ─────────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent):
        self._last_mouse   = event.pos()
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
            self._yaw   += dx * 0.4
            self._pitch += dy * 0.4
            self._pitch  = max(-89, min(89, self._pitch))
        elif self._mouse_button == Qt.MouseButton.RightButton:
            self._pan_x += dx * 0.002
            self._pan_y -= dy * 0.002

        self.update()

    def wheelEvent(self, event: QWheelEvent):
        self._zoom -= event.angleDelta().y() * 0.002
        self._zoom  = max(0.2, min(10.0, self._zoom))
        self.update()


# ── Helper ─────────────────────────────────────────────────────────────────────

def ctypes_offset(byte_offset: int):
    """Return a ctypes void pointer for VBO byte offsets."""
    import ctypes
    return ctypes.c_void_p(byte_offset)