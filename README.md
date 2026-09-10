# Meme Matcher

Match your live facial expression to the closest meme in your collection, in real time, using your webcam.

Meme Matcher uses [MediaPipe](https://developers.google.com/mediapipe) face and hand landmark detection to score expressions (surprise, smile, concern, "cheers", pointing, thumbs-up, victory sign, etc.) and finds the meme image whose expression/gesture profile is the closest match to yours.

## How it works

1. On startup, Meme Matcher downloads the MediaPipe `face_landmarker.task` and `hand_landmarker.task` models (only once — they're cached locally).
2. It scans your `assets/` folder for meme images (`.jpg`, `.jpeg`, `.png`), runs face/hand landmark detection on each, and extracts a feature vector: eye openness, mouth shape, eyebrow height, plus hand gestures (pointing, thumbs-up, victory, hand raised, hand near face) and where the hand sits relative to the face.
3. Extracted meme features are cached to `meme_features_cache.pkl`, keyed by each image's modification time and size, so future runs skip re-processing images that haven't changed and start almost instantly.
4. Your webcam feed is processed frame-by-frame (with configurable frame skipping for performance) and compared against every cached meme using a weighted exponential similarity score. Memes that involve a hand gesture only match when your hand gesture, side (left/right), and rough position relative to your face all agree with the meme's; face-only memes are skipped while you're gesturing.
5. A match only counts once its score clears `match_threshold` (default `120`) — below that, an animated loading indicator is shown in place of a meme.
6. The best-matching meme is displayed side-by-side with your webcam feed.

## Requirements

- Python 3.9+
- A webcam
- `curl` available on your system PATH (used to download the MediaPipe models)
- Third-party packages, installed via `requirements.txt`:
  - `opencv-python`
  - `numpy`
  - `mediapipe`
  - `Pillow` (used to render the on-screen status pill and loading animation)

  > **Note:** `Pillow` isn't currently listed in `requirements.txt` even though `main.py` imports it. Until that's added, install it manually with `pip install Pillow` alongside the other packages.


## Installation

### Windows

1. Install [Python 3.9+](https://www.python.org/downloads/windows/) (check "Add python.exe to PATH" during install).
2. `curl` ships by default on Windows 10/11. Verify with:
   ```powershell
   curl --version
   ```
3. Clone this project, then open a terminal (PowerShell or Command Prompt) in the project folder:
   ```powershell
   git clone https://github.com/luoijin/Meme-matcher.git
   cd Meme-matcher 
   ```
4. Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```
5. Install the required packages:
   ```powershell
   pip install -r requirements.txt
   ```
6. Run it:
   ```powershell
   python main.py
   ```

### Linux

1. Install Python 3.9+ and pip (usually preinstalled; if not):
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv curl
   ```
   (use your distro's package manager if not Debian/Ubuntu-based)
2. Clone this project and open a terminal in the project folder:
   ```bash
   git clone https://github.com/luoijin/Meme-matcher.git
   cd Meme-matcher
   ```
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
6. Run it:
   ```bash
   python main.py
   ```

### macOS

1. Install Python 3.9+, e.g. via [Homebrew](https://brew.sh/):
   ```bash
   brew install python
   ```
   `curl` is preinstalled on macOS.
2. Clone this project and open a terminal in the project folder:
   ```bash
   git clone https://github.com/luoijin/Meme-matcher.git
   cd Meme-matcher
   ```
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
6. Run it:
   ```bash
   python main.py
   ```


## Usage

- A window titled "Meme Matcher" opens showing your webcam feed next to a meme panel, split by a thin divider line.
- A status pill in the top-left corner shows whether your face is being tracked: green "Tracking active" when a face is detected, red "No face detected" when it isn't.
- Until your expression/gesture clears the match threshold, the meme panel shows a centered, animated 3-dot loading wave instead of a meme.
- Once matched, the meme itself is shown in the panel.
- Press **`q`** to quit.

Meme features are cached automatically: on each run, Meme Matcher only re-extracts features for images in `assets/` that are new or have changed (by modification time and file size), and drops entries for memes you've removed. You don't need to delete `meme_features_cache.pkl` by hand for day-to-day additions or edits — see Troubleshooting below for when you still might.

## Configuration

`MemeMatcher` accepts a few constructor parameters:

| Parameter | Default | Description |
|---|---|---|
| `assets_folder` | `"assets"` | Folder containing meme images |
| `frame_skip` | `2` | Only run full landmark detection every N frames (higher = faster, less responsive) |
| `meme_height` | `480` | Height (px) memes are resized to when loaded |
| `match_threshold` | `120` | Minimum similarity score required before a meme counts as a match; below this, the loading animation is shown |

Example:

```python
matcher = MemeMatcher(assets_folder="my_memes", frame_skip=1, meme_height=600, match_threshold=100)
matcher.run()
```

## Notes & Troubleshooting

- **"No face detected in `<file>`"**: the image was skipped because MediaPipe couldn't find a face in it. Try a clearer, more front-facing image.
- **Model download fails**: if `curl` isn't available or the download fails, manually download the models from the URLs printed in the error message and place them in the project directory as `face_landmarker.task` and `hand_landmarker.task`.
- **Slow performance**: increase `frame_skip` to process fewer frames per second, or reduce webcam resolution.
- **A meme never seems to match, even with the right expression**: if the meme involves a visible hand gesture, your hand also needs to be on the same side and in roughly the same position relative to your face as in the meme image — try mirroring the pose more closely, or lower `match_threshold`.
- **Cache seems stuck / stale after bulk changes**: the cache is refreshed automatically per-image, but if you change `meme_height` or otherwise want a full rebuild, delete `meme_features_cache.pkl` to force re-extraction of every image.

## License

This project is licensed under the [MIT License](LICENSE).