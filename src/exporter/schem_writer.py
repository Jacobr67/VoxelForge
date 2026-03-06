"""
src/exporter/schem_writer.py

Writes a MappedGrid to the modern .schem format (Sponge Schematic v2).
Uses NBT block states rather than numeric IDs — compatible with
WorldEdit, FAWE, and all Minecraft versions 1.13+.

Sponge Schematic v2 spec:
  https://github.com/SpongePowered/Schematic-Specification/blob/master/versions/schematic-2.md

Key structure:
  - Version: Int (2)
  - DataVersion: Int (Minecraft data version, e.g. 2730 for 1.18.2)
  - Width, Height, Length: Short
  - Palette: Compound — maps "minecraft:block_id[state=value]" → Int index
  - PaletteMax: Int
  - BlockData: ByteArray — varint-encoded palette indices, XZY order
  - BlockEntities: List (empty for our use case)
"""

import logging
from pathlib import Path
from typing import Union

import nbtlib
import numpy as np

logger = logging.getLogger(__name__)

# Minecraft data version for 1.20.1 — a widely-supported modern release
# Update this if targeting a different version
MC_DATA_VERSION = 3465


class SchemWriter:
    """
    Writes a MappedGrid to a .schem file (Sponge Schematic v2).

    Usage:
        writer = SchemWriter()
        writer.write(mapped_grid, Path("output/my_structure.schem"))
    """

    def write(self, mapped_grid, output_path: Union[str, Path]) -> Path:
        """
        Write a MappedGrid to a .schem file.

        Args:
            mapped_grid:  MappedGrid from ColourMapper.map().
            output_path:  Destination file path (will be created/overwritten).

        Returns:
            The resolved output path.

        Raises:
            OSError: If the file cannot be written.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        grid = mapped_grid.block_grid   # (X, Y, Z) object array
        sx, sy, sz = grid.shape

        logger.info(
            f"Writing .schem: {sx}×{sy}×{sz} blocks → {output_path}"
        )

        # ── 1. Build palette ──────────────────────────────────────────────────
        # Collect all unique block IDs, assign integer indices.
        # "minecraft:air" must be index 0 by convention.
        unique_blocks = set(grid.flatten())
        unique_blocks.discard("air")
        unique_blocks.discard("minecraft:air")

        palette_list = ["minecraft:air"] + sorted(unique_blocks)
        palette_map  = {block: idx for idx, block in enumerate(palette_list)}

        palette_nbt = nbtlib.Compound({
            block: nbtlib.Int(idx)
            for block, idx in palette_map.items()
        })

        # ── 2. Build BlockData varint array (XZY order) ───────────────────────
        # Sponge spec iterates: for x in X, for z in Z, for y in Y
        block_data_bytes = bytearray()

        for x in range(sx):
            for z in range(sz):
                for y in range(sy):
                    block_id = grid[x, y, z]
                    # Normalise bare "air" to namespaced form
                    if block_id == "air":
                        block_id = "minecraft:air"
                    idx = palette_map.get(block_id, 0)
                    block_data_bytes.extend(_encode_varint(idx))

        # ── 3. Build NBT structure ────────────────────────────────────────────
        schem_nbt = nbtlib.File({
            "Schematic": nbtlib.Compound({
                "Version":     nbtlib.Int(2),
                "DataVersion": nbtlib.Int(MC_DATA_VERSION),
                "Width":       nbtlib.Short(sx),
                "Height":      nbtlib.Short(sy),
                "Length":      nbtlib.Short(sz),
                "Offset":      nbtlib.IntArray(
                                   np.array([0, 0, 0], dtype=np.int32)
                               ),
                "PaletteMax":  nbtlib.Int(len(palette_list)),
                "Palette":     palette_nbt,
                "BlockData":   nbtlib.ByteArray(
                                   np.frombuffer(
                                       bytes(block_data_bytes),
                                       dtype=np.int8
                                   )
                               ),
                "BlockEntities": nbtlib.List[nbtlib.Compound]([]),
            })
        })

        # ── 4. Write gzip-compressed NBT ──────────────────────────────────────
        schem_nbt.save(str(output_path), gzipped=True)

        size_kb = output_path.stat().st_size / 1024
        logger.info(
            f".schem written: {output_path.name} "
            f"({size_kb:.1f} KB, palette: {len(palette_list)} block types)"
        )

        return output_path


# ── Varint encoder ─────────────────────────────────────────────────────────────

def _encode_varint(value: int) -> bytes:
    """
    Encode an integer as a variable-length integer (varint).
    Used by the Sponge Schematic format for the BlockData array.

    Each byte uses 7 bits of data; the MSB signals whether more bytes follow.
    """
    result = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value != 0:
            byte |= 0x80
        result.append(byte)
        if value == 0:
            break
    return bytes(result)