# 🎭 Meme Matcher

Match your live facial expression to the closest meme in your collection, in real time, using your webcam.

Meme Matcher uses [MediaPipe](https://developers.google.com/mediapipe) face and hand landmark detection to score expressions (surprise, smile, concern, "cheers", hand raised, etc.) and finds the meme image whose expression profile is the closest match to yours.

## How it works

1. On startup, Meme Matcher downloads the MediaPipe `face_landmarker.task` and `hand_landmarker.task` models (only once — they're cached locally).
2. It scans your `assets/` folder for meme images (`.jpg`, `.jpeg`, `.png`), runs face/hand landmark detection on each, and extracts a feature vector (eye openness, mouth shape, eyebrow height, hand-raised, etc.).
3. Extracted meme features are cached to `meme_features_cache.pkl` so future runs start instantly.
4. Your webcam feed is processed frame-by-frame (with configurable frame skipping for performance), features are extracted from your face, and compared against every cached meme using a weighted exponential similarity score.
5. The best-matching meme is displayed side-by-side with your webcam feed, along with its name and match score.

## Requirements

- Python 3.9+
- A webcam
- `curl` available on your system PATH (used to download the MediaPipe models)
- Dependencies listed in `requirements.txt`:
  ```
  opencv-python>=4.8.0
  numpy>=1.24.0
  mediapipe>=0.10.0
  ```

## Installation

### Windows

1. Install [Python 3.9+](https://www.python.org/downloads/windows/) (check "Add python.exe to PATH" during install).
2. `curl` ships by default on Windows 10/11. Verify with:
   ```powershell
   curl --version
   ```
3. Clone or download this project, then open a terminal (PowerShell or Command Prompt) in the project folder.
4. (Recommended) create and activate a virtual environment:
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```
5. Install the required packages:
   ```powershell
   pip install -r requirements.txt
   ```

### Linux

1. Install Python 3.9+ and pip (usually preinstalled; if not):
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv curl
   ```
   (use your distro's package manager if not Debian/Ubuntu-based)
2. Clone or download this project and open a terminal in the project folder.
3. (Recommended) create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
5. Make sure your user has permission to access the webcam device (`/dev/video0`); on most distros this works out of the box.

### macOS

1. Install Python 3.9+, e.g. via [Homebrew](https://brew.sh/):
   ```bash
   brew install python
   ```
   `curl` is preinstalled on macOS.
2. Clone or download this project and open a terminal in the project folder.
3. (Recommended) create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
5. The first time you run the script, macOS will prompt you to grant camera access to your terminal app — allow it, or the webcam feed won't open.

## Setup

1. Make sure you've completed the installation steps for your OS above.
2. Create an `assets/` folder in the project directory and fill it with meme images (`.jpg`, `.jpeg`, or `.png`). Each image should contain a clearly visible face.

```
project/
├── main.py
└── assets/
    ├── surprised_pikachu.jpg
    ├── side_eye.png
    └── crying_laughing.jpg
```

3. Run the script:

```bash
python main.py
```

The first run will:
- Download the required MediaPipe model files (`face_landmarker.task`, `hand_landmarker.task`)
- Process every image in `assets/` and cache the extracted features

Subsequent runs will load instantly from the cache.

## Usage

- A window will open showing your webcam feed next to the best-matching meme.
- The matched meme's name and similarity score are shown above it.
- Press **`q`** to quit.

If you add or change images in `assets/`, delete `meme_features_cache.pkl` so features are recomputed on the next run.

## Configuration

`MemeMatcher` accepts a few constructor parameters:

| Parameter | Default | Description |
|---|---|---|
| `assets_folder` | `"assets"` | Folder containing meme images |
| `frame_skip` | `2` | Only run full landmark detection every N frames (higher = faster, less responsive) |
| `meme_height` | `480` | Height (px) memes are resized to when loaded |

Example:

```python
matcher = MemeMatcher(assets_folder="my_memes", frame_skip=1, meme_height=600)
matcher.run()
```

## Notes & Troubleshooting

- **"No face detected in `<file>`"**: the image was skipped because MediaPipe couldn't find a face in it. Try a clearer, more front-facing image.
- **Model download fails**: if `curl` isn't available or the download fails, manually download the models from the URLs printed in the error message and place them in the project directory as `face_landmarker.task` and `hand_landmarker.task`.
- **Slow performance**: increase `frame_skip` to process fewer frames per second, or reduce webcam resolution.
- **Stale matches after updating memes**: delete `meme_features_cache.pkl` to force re-extraction.

## License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2026 Meme Matcher Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```