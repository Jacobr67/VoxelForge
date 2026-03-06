"""
src/exporter/litematica_writer.py

Writes a MappedGrid to the .litematica format used by the Litematica
client-side mod. Allows players to load the structure as a holographic
overlay in-game for guided manual building.

Litematica format overview:
  - A gzip-compressed NBT file
  - Root compound contains "MinecraftDataVersion", "Version", "Metadata",
    and "Regions" (a compound of named region compounds)
  - Each region contains:
      - "BlockStatePalette": List of block state compounds
      - "BlockStates": LongArray — packed bit array of palette indices
      - "Position" and "Size": position/size compounds
      - "TileEntities", "Entities", "PendingBlockTicks": empty lists

Litematica version: 5 (current as of mod version 0.12.x)
"""

import logging
import math
from pathlib import Path
from typing import Union

import nbtlib
import numpy as np

logger = logging.getLogger(__name__)

LITEMATICA_VERSION   = 5
MC_DATA_VERSION      = 3465   # Minecraft 1.20.1


class LitematicaWriter:
    """
    Writes a MappedGrid to a .litematica file.

    Usage:
        writer = LitematicaWriter()
        writer.write(mapped_grid, Path("output/my_structure.litematica"))
    """

    def write(
        self,
        mapped_grid,
        output_path: Union[str, Path],
        region_name: str = "VoxelForge",
    ) -> Path:
        """
        Write a MappedGrid to a .litematica file.

        Args:
            mapped_grid:  MappedGrid from ColourMapper.map().
            output_path:  Destination file path (will be created/overwritten).
            region_name:  Name for the single region inside the schematic.

        Returns:
            The resolved output path.

        Raises:
            OSError: If the file cannot be written.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        grid = mapped_grid.block_grid   # (X, Y, Z) object array
        sx, sy, sz = grid.shape
        total_blocks = sx * sy * sz

        logger.info(
            f"Writing .litematica: {sx}×{sy}×{sz} blocks → {output_path}"
        )

        # ── 1. Build block state palette ──────────────────────────────────────
        # Litematica uses a list of block state compounds, each with "Name"
        # and optionally "Properties". We use simple name-only entries.
        unique_blocks = set(grid.flatten())
        unique_blocks.discard("air")

        # air must be index 0
        palette_list = ["minecraft:air"] + sorted(
            (b if ":" in b else f"minecraft:{b}")
            for b in unique_blocks
            if b != "minecraft:air"
        )

        palette_map = {block: idx for idx, block in enumerate(palette_list)}

        palette_nbt = nbtlib.List[nbtlib.Compound]([
            nbtlib.Compound({"Name": nbtlib.String(block_id)})
            for block_id in palette_list
        ])

        # ── 2. Pack block indices into LongArray ──────────────────────────────
        # Litematica packs palette indices into 64-bit longs.
        # Bits per entry = max(4, ceil(log2(palette_size)))
        # Indices are packed YZX order (y is innermost loop).
        bits_per_entry = max(4, math.ceil(math.log2(max(len(palette_list), 2))))
        entries_per_long = 64 // bits_per_entry
        num_longs = math.ceil(total_blocks / entries_per_long)

        long_array = np.zeros(num_longs, dtype=np.int64)

        block_index = 0
        for x in range(sx):
            for z in range(sz):
                for y in range(sy):
                    raw_id = grid[x, y, z]
                    if raw_id == "air":
                        raw_id = "minecraft:air"
                    elif ":" not in raw_id:
                        raw_id = f"minecraft:{raw_id}"

                    palette_idx = palette_map.get(raw_id, 0)

                    long_idx  = block_index // entries_per_long
                    bit_offset = (block_index % entries_per_long) * bits_per_entry

                    long_array[long_idx] |= np.int64(palette_idx) << bit_offset
                    block_index += 1

        # ── 3. Build region NBT ───────────────────────────────────────────────
        region_nbt = nbtlib.Compound({
            "BlockStatePalette": palette_nbt,
            "BlockStates": nbtlib.LongArray(long_array),
            "TileEntities":    nbtlib.List[nbtlib.Compound]([]),
            "Entities":        nbtlib.List[nbtlib.Compound]([]),
            "PendingBlockTicks": nbtlib.List[nbtlib.Compound]([]),
            "Position": nbtlib.Compound({
                "x": nbtlib.Int(0),
                "y": nbtlib.Int(0),
                "z": nbtlib.Int(0),
            }),
            "Size": nbtlib.Compound({
                "x": nbtlib.Int(sx),
                "y": nbtlib.Int(sy),
                "z": nbtlib.Int(sz),
            }),
        })

        # ── 4. Build metadata ─────────────────────────────────────────────────
        metadata_nbt = nbtlib.Compound({
            "Name":        nbtlib.String(region_name),
            "Author":      nbtlib.String("VoxelForge"),
            "Description": nbtlib.String("Generated by VoxelForge"),
            "TimeCreated":  nbtlib.Long(0),
            "TimeModified": nbtlib.Long(0),
            "EnclosingSize": nbtlib.Compound({
                "x": nbtlib.Int(sx),
                "y": nbtlib.Int(sy),
                "z": nbtlib.Int(sz),
            }),
            "RegionCount": nbtlib.Int(1),
        })

        # ── 5. Assemble root NBT ──────────────────────────────────────────────
        root_nbt = nbtlib.File({
            "": nbtlib.Compound({
                "MinecraftDataVersion": nbtlib.Int(MC_DATA_VERSION),
                "Version":              nbtlib.Int(LITEMATICA_VERSION),
                "Metadata":             metadata_nbt,
                "Regions": nbtlib.Compound({
                    region_name: region_nbt,
                }),
            })
        })

        # ── 6. Write gzip-compressed NBT ──────────────────────────────────────
        root_nbt.save(str(output_path), gzipped=True)

        size_kb = output_path.stat().st_size / 1024
        logger.info(
            f".litematica written: {output_path.name} "
            f"({size_kb:.1f} KB, {bits_per_entry} bits/entry, "
            f"{len(palette_list)} block types)"
        )

        return output_path