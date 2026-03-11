# VoxelForge

> AI-powered text-to-3D Minecraft structure generator

---

## 📜 License

© 2026 VoxelForge — All rights reserved.

This software and its source code are proprietary.
No permission is granted to use, copy, modify, or distribute this software without explicit permission.

---

## 📖 About

VoxelForge is a desktop application that converts natural language prompts into Minecraft-compatible 3D structures using generative AI.

Users can describe a structure in plain text, and the application generates a 3D model, converts it into Minecraft-style voxels, and exports it as a structure file ready to import into Minecraft building tools.

The goal of VoxelForge is to make advanced structure creation accessible to everyone by removing the need for manual 3D modelling.

---

## ✨ Features

* 🧠 **AI-generated 3D meshes** from text prompts via NVIDIA Trellis
* 🧱 **Automatic voxelisation** into Minecraft blocks
* 🎚 **Adjustable output resolution** (16–256 blocks per axis)
* 🎨 **Block palette filtering** — toggle natural, stone, wood, coloured, and survival-only block sets
* 🔢 **Reproducible generation** via seed control
* 🫙 **Hollow interior** option to reduce block count on large structures
* 📦 **Multiple export formats**
* 🖥 **Live 3D preview** — orbit, pan, and zoom the mesh and voxel result before exporting

Supported export formats:

* `.schematic` — Legacy MCEdit / WorldEdit (Minecraft 1.12 and older)
* `.schem` — Modern WorldEdit / FAWE (Minecraft 1.13+) ✓ Recommended
* `.litematica` — Litematica mod, in-game holographic overlay

---

## ⚙️ How It Works

1. Enter a text prompt describing the structure.
2. NVIDIA Trellis generates a 3D mesh from the prompt.
3. The mesh is voxelised into a block grid at your chosen resolution.
4. Each voxel is colour-matched to the closest Minecraft block using perceptual LAB colour distance.
5. The structure is exported to your chosen Minecraft-compatible format.

---

## 🚀 Getting Started

### Requirements

* Python 3.11+
* An [NVIDIA NGC API key](https://build.nvidia.com/microsoft/trellis) for the Trellis endpoint

### Installation

```bash
git clone https://github.com/your-username/VoxelForge.git
cd VoxelForge
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and add your API key:

```
NVIDIA_API_KEY=nvapi-your-key-here
```

### Running

```bash
python main.py
```

Add `--debug` for verbose console logging:

```bash
python main.py --debug
```

---

## 🖥 Platform Support

VoxelForge is designed as a cross-platform desktop application.

| Platform | Status |
|----------|--------|
| Windows  | ✅ Tested |
| Linux    | 🔄 Planned |
| macOS    | 🔄 Planned |

---

## 🚧 Project Status

VoxelForge is currently under active development. The core pipeline is functional end-to-end:

* ✅ Text prompt → NVIDIA Trellis API → GLB mesh
* ✅ Mesh processing and normalisation
* ✅ Surface voxelisation with interior fill
* ✅ Perceptual colour mapping to Minecraft blocks
* ✅ Export to `.schematic`, `.schem`, `.litematica`
* ✅ Live OpenGL 3D preview (mesh + voxel modes)

Features and architecture may change as development progresses.

---

## 💬 Contributing

This repository is public for visibility, feedback, and issue tracking.

Code contributions are not currently accepted without explicit permission from the project owner.

Community members are welcome to:

* 🐛 Open issue reports
* 💡 Suggest features
* 💬 Provide feedback and discussion

---

## ⚠ Disclaimer

VoxelForge is an independent project and is **not affiliated with Mojang or Minecraft**.
