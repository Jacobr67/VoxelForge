"""
src/exporter/schematic_writer.py

Writes a MappedGrid to the legacy .schematic format (MCEdit / WorldEdit).
Uses numeric block IDs and data values — compatible with Minecraft 1.12
and older, and still accepted by many classic modding tools.

Format spec:
  - Root NBT tag: "Schematic"
  - Required tags: Width, Height, Length (short), Materials (string),
    Blocks (byte array), Data (byte array)
  - Block array is indexed as: (y * length + z) * width + x
  - Materials tag must be "Alpha" for standard Minecraft blocks
"""

import logging
import struct
from pathlib import Path
from typing import Union

import nbtlib
import numpy as np

logger = logging.getLogger(__name__)


# ── Legacy block ID map ────────────────────────────────────────────────────────
# Maps modern Minecraft namespaced IDs → (legacy_id, data_value) tuples
# for Minecraft 1.12 and below.
# Extend this as more blocks are added to the palettes.

LEGACY_BLOCK_IDS: dict[str, tuple[int, int]] = {
    "air":                          (0,   0),
    "minecraft:air":                (0,   0),
    "minecraft:stone":              (1,   0),
    "minecraft:granite":            (1,   1),
    "minecraft:polished_granite":   (1,   2),
    "minecraft:diorite":            (1,   3),
    "minecraft:polished_diorite":   (1,   4),
    "minecraft:andesite":           (1,   5),
    "minecraft:polished_andesite":  (1,   6),
    "minecraft:grass_block":        (2,   0),
    "minecraft:dirt":               (3,   0),
    "minecraft:cobblestone":        (4,   0),
    "minecraft:oak_planks":         (5,   0),
    "minecraft:spruce_planks":      (5,   1),
    "minecraft:birch_planks":       (5,   2),
    "minecraft:jungle_planks":      (5,   3),
    "minecraft:acacia_planks":      (5,   4),
    "minecraft:dark_oak_planks":    (5,   5),
    "minecraft:sand":               (12,  0),
    "minecraft:red_sand":           (12,  1),
    "minecraft:gravel":             (13,  0),
    "minecraft:gold_ore":           (14,  0),
    "minecraft:iron_ore":           (15,  0),
    "minecraft:coal_ore":           (16,  0),
    "minecraft:oak_log":            (17,  0),
    "minecraft:spruce_log":         (17,  1),
    "minecraft:birch_log":          (17,  2),
    "minecraft:jungle_log":         (17,  3),
    "minecraft:oak_leaves":         (18,  0),
    "minecraft:spruce_leaves":      (18,  1),
    "minecraft:birch_leaves":       (18,  2),
    "minecraft:jungle_leaves":      (18,  3),
    "minecraft:glass":              (20,  0),
    "minecraft:lapis_ore":          (21,  0),
    "minecraft:lapis_block":        (22,  0),
    "minecraft:sandstone":          (24,  0),
    "minecraft:chiseled_sandstone": (24,  1),
    "minecraft:smooth_sandstone":   (24,  2),
    "minecraft:white_wool":         (35,  0),
    "minecraft:orange_wool":        (35,  1),
    "minecraft:magenta_wool":       (35,  2),
    "minecraft:light_blue_wool":    (35,  3),
    "minecraft:yellow_wool":        (35,  4),
    "minecraft:lime_wool":          (35,  5),
    "minecraft:pink_wool":          (35,  6),
    "minecraft:gray_wool":          (35,  7),
    "minecraft:light_gray_wool":    (35,  8),
    "minecraft:cyan_wool":          (35,  9),
    "minecraft:purple_wool":        (35, 10),
    "minecraft:blue_wool":          (35, 11),
    "minecraft:brown_wool":         (35, 12),
    "minecraft:green_wool":         (35, 13),
    "minecraft:red_wool":           (35, 14),
    "minecraft:black_wool":         (35, 15),
    "minecraft:gold_block":         (41,  0),
    "minecraft:iron_block":         (42,  0),
    "minecraft:brick":              (45,  0),
    "minecraft:mossy_cobblestone":  (48,  0),
    "minecraft:obsidian":           (49,  0),
    "minecraft:diamond_ore":        (56,  0),
    "minecraft:diamond_block":      (57,  0),
    "minecraft:crafting_table":     (58,  0),
    "minecraft:farmland":           (60,  0),
    "minecraft:netherrack":         (87,  0),
    "minecraft:soul_sand":          (88,  0),
    "minecraft:glowstone":          (89,  0),
    "minecraft:white_stained_glass": (95, 0),
    "minecraft:orange_stained_glass": (95, 1),
    "minecraft:magenta_stained_glass": (95, 2),
    "minecraft:light_blue_stained_glass": (95, 3),
    "minecraft:yellow_stained_glass": (95, 4),
    "minecraft:lime_stained_glass": (95, 5),
    "minecraft:pink_stained_glass": (95, 6),
    "minecraft:gray_stained_glass": (95, 7),
    "minecraft:light_gray_stained_glass": (95, 8),
    "minecraft:cyan_stained_glass": (95, 9),
    "minecraft:purple_stained_glass": (95, 10),
    "minecraft:blue_stained_glass": (95, 11),
    "minecraft:brown_stained_glass": (95, 12),
    "minecraft:green_stained_glass": (95, 13),
    "minecraft:red_stained_glass":  (95, 14),
    "minecraft:black_stained_glass": (95, 15),
    "minecraft:emerald_ore":        (129, 0),
    "minecraft:emerald_block":      (133, 0),
    "minecraft:quartz_block":       (155, 0),
    "minecraft:white_terracotta":   (159, 0),
    "minecraft:orange_terracotta":  (159, 1),
    "minecraft:magenta_terracotta": (159, 2),
    "minecraft:light_blue_terracotta": (159, 3),
    "minecraft:yellow_terracotta":  (159, 4),
    "minecraft:lime_terracotta":    (159, 5),
    "minecraft:pink_terracotta":    (159, 6),
    "minecraft:gray_terracotta":    (159, 7),
    "minecraft:light_gray_terracotta": (159, 8),
    "minecraft:cyan_terracotta":    (159, 9),
    "minecraft:purple_terracotta":  (159, 10),
    "minecraft:blue_terracotta":    (159, 11),
    "minecraft:brown_terracotta":   (159, 12),
    "minecraft:green_terracotta":   (159, 13),
    "minecraft:red_terracotta":     (159, 14),
    "minecraft:black_terracotta":   (159, 15),
    "minecraft:white_concrete":     (251, 0),
    "minecraft:orange_concrete":    (251, 1),
    "minecraft:magenta_concrete":   (251, 2),
    "minecraft:light_blue_concrete": (251, 3),
    "minecraft:yellow_concrete":    (251, 4),
    "minecraft:lime_concrete":      (251, 5),
    "minecraft:pink_concrete":      (251, 6),
    "minecraft:gray_concrete":      (251, 7),
    "minecraft:light_gray_concrete": (251, 8),
    "minecraft:cyan_concrete":      (251, 9),
    "minecraft:purple_concrete":    (251, 10),
    "minecraft:blue_concrete":      (251, 11),
    "minecraft:brown_concrete":     (251, 12),
    "minecraft:green_concrete":     (251, 13),
    "minecraft:red_concrete":       (251, 14),
    "minecraft:black_concrete":     (251, 15),
}

# Fallback block for any unmapped modern IDs
FALLBACK_BLOCK_ID   = 1   # Stone
FALLBACK_BLOCK_DATA = 0


class SchematicWriter:
    """
    Writes a MappedGrid to a .schematic file.

    Usage:
        writer = SchematicWriter()
        writer.write(mapped_grid, Path("output/my_structure.schematic"))
    """

    def write(self, mapped_grid, output_path: Union[str, Path]) -> Path:
        """
        Write a MappedGrid to a .schematic file.

        Args:
            mapped_grid:  MappedGrid from ColourMapper.map().
            output_path:  Destination file path (will be created/overwritten).

        Returns:
            The resolved output path.

        Raises:
            OSError: If the file cannot be written.
            ValueError: If the grid dimensions exceed schematic limits (32767).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        grid = mapped_grid.block_grid   # (X, Y, Z) object array of block IDs
        sx, sy, sz = grid.shape

        if max(sx, sy, sz) > 32767:
            raise ValueError(
                f"Grid dimensions ({sx}, {sy}, {sz}) exceed the .schematic "
                f"maximum of 32767 per axis."
            )

        logger.info(
            f"Writing .schematic: {sx}×{sy}×{sz} blocks → {output_path}"
        )

        # ── Build flat block/data arrays ──────────────────────────────────────
        # .schematic index order: (y * length + z) * width + x
        # Our grid is (X=width, Y=height, Z=length)
        total = sx * sy * sz
        blocks_arr = np.zeros(total, dtype=np.uint8)
        data_arr   = np.zeros(total, dtype=np.uint8)

        unmapped_ids = set()

        for y in range(sy):
            for z in range(sz):
                for x in range(sx):
                    idx = (y * sz + z) * sx + x
                    block_id = grid[x, y, z]

                    legacy_id, legacy_data = LEGACY_BLOCK_IDS.get(
                        block_id,
                        (None, None)
                    )

                    if legacy_id is None:
                        unmapped_ids.add(block_id)
                        legacy_id   = FALLBACK_BLOCK_ID
                        legacy_data = FALLBACK_BLOCK_DATA

                    blocks_arr[idx] = legacy_id & 0xFF
                    data_arr[idx]   = legacy_data & 0x0F

        if unmapped_ids:
            logger.warning(
                f"{len(unmapped_ids)} block IDs had no legacy mapping and "
                f"were replaced with stone: {unmapped_ids}"
            )

        # ── Build NBT structure ───────────────────────────────────────────────
        schematic = nbtlib.Compound({
            "Schematic": nbtlib.Compound({
                "Width":     nbtlib.Short(sx),
                "Height":    nbtlib.Short(sy),
                "Length":    nbtlib.Short(sz),
                "Materials": nbtlib.String("Alpha"),
                "Blocks":    nbtlib.ByteArray(blocks_arr.astype(np.int8)),
                "Data":      nbtlib.ByteArray(data_arr.astype(np.int8)),
                "Entities":  nbtlib.List[nbtlib.Compound]([]),
                "TileEntities": nbtlib.List[nbtlib.Compound]([]),
            })
        })

        # ── Write compressed NBT ──────────────────────────────────────────────
        schematic.save(str(output_path), gzipped=True)

        size_kb = output_path.stat().st_size / 1024
        logger.info(
            f".schematic written: {output_path.name} "
            f"({size_kb:.1f} KB, {total:,} blocks)"
        )

        return output_path