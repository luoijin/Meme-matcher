import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pickle
import os
import subprocess
import time
import math
from PIL import Image, ImageDraw, ImageFont

class MemeMatcher:
    # MediaPipe landmark indices for facial features
    LEFT_EYE_UPPER = [150, 145, 158]
    LEFT_EYE_LOWER = [23, 27, 133]
    RIGHT_EYE_UPPER = [386, 374, 385]
    RIGHT_EYE_LOWER = [253, 257, 362]
    LEFT_EYEBROW = [70, 63, 105, 66, 107]
    RIGHT_EYEBROW = [300, 293, 334, 296, 336]
    MOUTH_OUTER = [61, 291, 39, 181, 0, 17, 269, 405]
    NOSE_TIP = 4

    # Hand Landmark Indices
    WRIST = 0
    THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
    INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
    MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP = 12, 10, 9
    RING_TIP, RING_PIP, RING_MCP = 16, 14, 13
    PINKY_TIP, PINKY_PIP, PINKY_MCP = 20, 18, 17

    CACHE_FILE = "meme_features_cache.pkl"

    def __init__(self, assets_folder="assets", frame_skip=2, meme_height=480, match_threshold=120):
        self.last_features = None
        self.frame_counter = 0
        self.frame_skip = frame_skip
        self.meme_height = meme_height
        self.match_threshold = match_threshold

        self.face_model_path = self._download_model(
            "face_landmarker.task",
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
        self.hand_model_path = self._download_model(
            "hand_landmarker.task",
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )

        self.face_mesh_video = self._init_face_landmarker(video_mode=True)
        self.hand_detector_video = self._init_hand_landmarker(video_mode=True)
        self.face_mesh_image = self._init_face_landmarker(video_mode=False)
        self.hand_detector_image = self._init_hand_landmarker(video_mode=False)

        self.memes = []
        self.meme_features = []

        self.feature_keys = [
            'surprise_score', 'smile_score', 'concern_score', 'cheers_score', 
            'hand_raised', 'num_hands', 'eye_openness', 'eyes_symmetry', 
            'mouth_openness', 'mouth_width_ratio', 'mouth_elevation', 
            'eyebrow_height', 'brow_symmetry',
            'pointing_score', 'thumbs_up_score', 'victory_score', 'hand_near_face',
            'rel_hand_x', 'rel_hand_y', 'active_hand_side'
        ]

        self.feature_weights = np.array([25, 20, 20, 30, 25, 15, 20, 10, 25, 20, 15, 20, 10, 30, 30, 30, 25, 25, 25, 20])
        self.feature_factors = np.array([10, 10, 10, 10, 15, 15, 5, 5, 5, 5, 5, 5, 5, 12, 12, 12, 12, 4, 4, 10])

        self.load_memes(assets_folder)

        if self.memes:
            aspects = [m['image'].shape[1] / m['image'].shape[0] for m in self.memes]
            self.meme_aspect_ratio = float(np.mean(aspects))
        else:
            self.meme_aspect_ratio = 1.0

        # Load system font for modern UI overlay
        self.font_title = self._get_font(22, bold=True)
        self.font_subtitle = self._get_font(15, bold=False)
        self.font_status = self._get_font(14, bold=True)

    def _get_font(self, size, bold=False):
        """Attempts to load common system fonts or falls back to default PIL font."""
        font_names = [
            "arial.ttf", "segoeui.ttf", "SF-Pro-Display-Regular.otf", 
            "Helvetica.ttf", "DejaVuSans.ttf"
        ]
        if bold:
            font_names = ["arialbd.ttf", "segoeuib.ttf", "SF-Pro-Display-Bold.otf", "DejaVuSans-Bold.ttf"] + font_names

        for name in font_names:
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _download_model(self, model_path, url):
        if not os.path.exists(model_path):
            print(f"Downloading {model_path}...")
            try:
                subprocess.run(['curl', '-L', url, '-o', model_path], check=True, capture_output=True)
                print(f"{model_path} downloaded successfully.")
            except subprocess.CalledProcessError:
                raise RuntimeError(f"Failed to download model. Please download manually from {url}")
        return model_path

    def _init_face_landmarker(self, video_mode=True):
        mode = mp.tasks.vision.RunningMode.VIDEO if video_mode else mp.tasks.vision.RunningMode.IMAGE
        return mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=self.face_model_path),
                running_mode=mode,
                num_faces=1,
                min_face_detection_confidence=0.5 if video_mode else 0.3,
                min_face_presence_confidence=0.5 if video_mode else 0.3,
                min_tracking_confidence=0.5 if video_mode else 0.0
            )
        )

    def _init_hand_landmarker(self, video_mode=True):
        mode = mp.tasks.vision.RunningMode.VIDEO if video_mode else mp.tasks.vision.RunningMode.IMAGE
        return mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=self.hand_model_path),
                running_mode=mode,
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3 if video_mode else 0.0
            )
        )

    def _file_signature(self, img_file):
        stat = img_file.stat()
        return (stat.st_mtime, stat.st_size, self.meme_height)

    def load_memes(self, folder):
        assets_path = Path(folder)
        image_files = sorted(
            list(assets_path.glob("*.jpg"))
            + list(assets_path.glob("*.jpeg"))
            + list(assets_path.glob("*.png"))
        )
        print(f"Found {len(image_files)} meme image(s) in '{folder}'.")

        cache = {}
        if os.path.exists(self.CACHE_FILE):
            try:
                with open(self.CACHE_FILE, "rb") as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    cache = loaded
                else:
                    print("Cache is in an old format, rebuilding it.")
            except (pickle.UnpicklingError, EOFError, AttributeError, ValueError):
                print("Cache file could not be read, rebuilding it.")

        to_process = []
        reused = {}
        for img_file in image_files:
            key = str(img_file)
            entry = cache.get(key)
            if entry and entry.get("signature") == self._file_signature(img_file):
                reused[key] = entry
            else:
                to_process.append(img_file)

        removed = set(cache.keys()) - {str(f) for f in image_files}
        if removed:
            names = ", ".join(Path(p).name for p in removed)
            print(f"Dropping {len(removed)} meme(s) no longer in '{folder}': {names}")

        if to_process:
            print(f"Extracting features for {len(to_process)} new/changed image(s)...")

            def process_meme(img_file):
                img = cv2.imread(str(img_file))
                if img is None:
                    print(f"Could not read image: {img_file.name}")
                    return None
                h, w = img.shape[:2]
                scale = self.meme_height / h
                img_resized = cv2.resize(img, (int(w * scale), self.meme_height))
                features = self.extract_face_features(img_resized, is_static=True)
                if features is None:
                    print(f"No face detected in {img_file.name} - skipping.")
                    return None
                meme = {
                    'image': img_resized,
                    'name': img_file.stem.replace('_', ' ').title(),
                    'path': str(img_file),
                }
                return str(img_file), {
                    "signature": self._file_signature(img_file),
                    "meme": meme,
                    "features": features,
                }

            with ThreadPoolExecutor() as executor:
                results = list(executor.map(process_meme, to_process))

            for r in results:
                if r:
                    key, entry = r
                    reused[key] = entry
                    print(f"Loaded: {entry['meme']['name']}")
        else:
            print("No new or changed images - using cached features for all of them.")

        self.memes = []
        self.meme_features = []
        for img_file in image_files:
            entry = reused.get(str(img_file))
            if entry:
                self.memes.append(entry["meme"])
                self.meme_features.append(entry["features"])

        with open(self.CACHE_FILE, "wb") as f:
            pickle.dump(reused, f)

        print(f"Memes loaded: {len(self.memes)}\n")

    def extract_face_features(self, image, is_static=False):
        if is_static:
            face_landmarker = self.face_mesh_image
            hand_landmarker = self.hand_detector_image
        else:
            face_landmarker = self.face_mesh_video
            hand_landmarker = self.hand_detector_video

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if is_static:
            face_result = face_landmarker.detect(mp_image)
            hand_result = hand_landmarker.detect(mp_image)
        else:
            self.frame_counter += 1
            if self.frame_counter % self.frame_skip != 0:
                return getattr(self, "last_features", None)
            timestamp = int(self.frame_counter * 33)
            face_result = face_landmarker.detect_for_video(mp_image, timestamp)
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp)

        if not face_result.face_landmarks:
            return None

        landmarks = face_result.face_landmarks[0]
        landmark_array = np.array([[l.x, l.y] for l in landmarks])
        features = self._compute_features(landmark_array, hand_result)
        self.last_features = features
        return features

    def _compute_features(self, landmark_array, hand_result):
        def ear(upper, lower):
            vertical = np.linalg.norm(landmark_array[upper] - landmark_array[lower], axis=1).mean()
            horizontal = np.linalg.norm(landmark_array[upper[0]] - landmark_array[upper[-1]])
            return vertical / (horizontal + 1e-6)

        left_ear = ear(self.LEFT_EYE_UPPER, self.LEFT_EYE_LOWER)
        right_ear = ear(self.RIGHT_EYE_UPPER, self.RIGHT_EYE_LOWER)
        avg_ear = (left_ear + right_ear) / 2.0

        mouth_top, mouth_bottom = landmark_array[13], landmark_array[14]
        mouth_height = np.linalg.norm(mouth_top - mouth_bottom)
        mouth_left, mouth_right = landmark_array[61], landmark_array[291]
        mouth_width = np.linalg.norm(mouth_left - mouth_right)
        mouth_ar = mouth_height / (mouth_width + 1e-6)
        inner_width = np.linalg.norm(landmark_array[78] - landmark_array[308])
        mouth_width_ratio = inner_width / (mouth_width + 1e-6)

        left_brow_y = landmark_array[self.LEFT_EYEBROW][:, 1].mean()
        right_brow_y = landmark_array[self.RIGHT_EYEBROW][:, 1].mean()
        left_eye_center = landmark_array[self.LEFT_EYE_UPPER + self.LEFT_EYE_LOWER][:, 1].mean()
        right_eye_center = landmark_array[self.RIGHT_EYE_UPPER + self.RIGHT_EYE_LOWER][:, 1].mean()
        left_brow_h = left_eye_center - left_brow_y
        right_brow_h = right_eye_center - right_brow_y
        avg_brow_h = (left_brow_h + right_brow_h) / 2.0

        mouth_center_y = (mouth_left[1] + mouth_right[1]) / 2.0
        nose_tip = landmark_array[self.NOSE_TIP]
        mouth_elev = nose_tip[1] - mouth_center_y

        num_hands = len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0
        hand_raised = 0.0
        pointing_score = 0.0
        thumbs_up_score = 0.0
        victory_score = 0.0
        hand_near_face = 0.0
        
        rel_hand_x = 0.0
        rel_hand_y = 0.0
        active_hand_side = 0.0

        face_center = landmark_array.mean(axis=0)
        face_width = np.linalg.norm(landmark_array[454] - landmark_array[234]) + 1e-6

        if num_hands > 0 and hand_result.handedness:
            sides = []
            for hand_landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                pts = np.array([[lm.x, lm.y] for lm in hand_landmarks])
                label = handedness[0].category_name
                sides.append(1.0 if label == "Left" else 2.0)

                def get_finger_angle(p1, p2, p3):
                    v1 = p1 - p2
                    v2 = p3 - p2
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

                index_ext = get_finger_angle(pts[self.INDEX_MCP], pts[self.INDEX_PIP], pts[self.INDEX_TIP]) > 140
                middle_ext = get_finger_angle(pts[self.MIDDLE_MCP], pts[self.MIDDLE_PIP], pts[self.MIDDLE_TIP]) > 140
                ring_ext = get_finger_angle(pts[self.RING_MCP], pts[self.RING_PIP], pts[self.RING_TIP]) > 140
                pinky_ext = get_finger_angle(pts[self.PINKY_MCP], pts[self.PINKY_PIP], pts[self.PINKY_TIP]) > 140
                thumb_ext = get_finger_angle(pts[self.THUMB_MCP], pts[self.THUMB_IP], pts[self.THUMB_TIP]) > 130

                wrist_pt = pts[self.WRIST]
                offset = (wrist_pt - face_center) / face_width
                rel_hand_x = float(np.clip(offset[0], -2.0, 2.0))
                rel_hand_y = float(np.clip(offset[1], -2.0, 2.0))

                min_dist_to_face = np.min(np.linalg.norm(pts - face_center, axis=1)) / face_width
                if min_dist_to_face < 0.8:
                    hand_near_face = 1.0

                if wrist_pt[1] < face_center[1] + 0.2:
                    hand_raised = 1.0

                if index_ext and not middle_ext and not ring_ext and not pinky_ext:
                    pointing_score = 1.0
                elif thumb_ext and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
                    thumbs_up_score = 1.0
                elif index_ext and middle_ext and not ring_ext and not pinky_ext:
                    victory_score = 1.0

            active_hand_side = float(sides[0]) if len(sides) == 1 else 3.0

        return {
            'eye_openness': avg_ear,
            'left_eye_open': left_ear,
            'right_eye_open': right_ear,
            'eyes_symmetry': abs(left_ear - right_ear),
            'mouth_openness': mouth_ar,
            'mouth_width': mouth_width,
            'mouth_width_ratio': mouth_width_ratio,
            'mouth_elevation': mouth_elev,
            'eyebrow_height': avg_brow_h,
            'left_brow_height': left_brow_h,
            'right_brow_height': right_brow_h,
            'brow_symmetry': abs(left_brow_h - right_brow_h),
            'num_hands': num_hands,
            'hand_raised': hand_raised,
            'surprise_score': avg_ear * avg_brow_h * mouth_ar,
            'smile_score': mouth_width_ratio * (1.0 - mouth_ar),
            'concern_score': avg_brow_h * (1.0 - mouth_elev),
            'cheers_score': mouth_width_ratio * (1.0 - mouth_ar) * hand_raised,
            'pointing_score': pointing_score,
            'thumbs_up_score': thumbs_up_score,
            'victory_score': victory_score,
            'hand_near_face': hand_near_face,
            'rel_hand_x': rel_hand_x,
            'rel_hand_y': rel_hand_y,
            'active_hand_side': active_hand_side
        }

    def compute_similarity(self, features1, features2):
        if features1 is None or features2 is None:
            return 0.0
        vector1 = np.array([features1[k] for k in self.feature_keys])
        vector2 = np.array([features2[k] for k in self.feature_keys])
        diff = np.abs(vector1 - vector2)
        similarity = np.exp(-diff * self.feature_factors)
        return float(np.sum(self.feature_weights * similarity))

    def find_best_match(self, user_features):
        if user_features is None:
            return None, 0.0

        best_meme = None
        best_score = 0.0

        face_keys = [
            'surprise_score', 'smile_score', 'concern_score', 'eye_openness', 
            'eyes_symmetry', 'mouth_openness', 'mouth_width_ratio', 
            'mouth_elevation', 'eyebrow_height', 'brow_symmetry'
        ]
        hand_keys = [
            'hand_raised', 'num_hands', 'cheers_score', 'pointing_score', 
            'thumbs_up_score', 'victory_score', 'hand_near_face'
        ]

        key_to_weight = dict(zip(self.feature_keys, self.feature_weights))
        face_weight_sum = sum(key_to_weight[k] for k in face_keys)
        hand_weight_sum = sum(key_to_weight[k] for k in hand_keys)

        for meme, mf in zip(self.memes, self.meme_features):
            face_sim = self._sub_similarity(user_features, mf, face_keys)
            hand_sim = self._sub_similarity(user_features, mf, hand_keys)

            face_ratio = face_sim / (face_weight_sum + 1e-6)
            hand_ratio = hand_sim / (hand_weight_sum + 1e-6)

            meme_has_gesture = (
                mf['hand_raised'] > 0.5 or 
                mf['pointing_score'] > 0.5 or 
                mf['thumbs_up_score'] > 0.5 or 
                mf['victory_score'] > 0.5 or 
                mf['hand_near_face'] > 0.5
            )

            if meme_has_gesture:
                if abs(mf['active_hand_side'] - user_features['active_hand_side']) > 0.5:
                    continue
                if abs(mf['rel_hand_x'] - user_features['rel_hand_x']) > 0.8:
                    continue
                if face_ratio < 0.5 or hand_ratio < 0.5:
                    continue
                if mf['pointing_score'] > 0.5 and user_features['pointing_score'] < 0.5:
                    continue
                if mf['thumbs_up_score'] > 0.5 and user_features['thumbs_up_score'] < 0.5:
                    continue
                if mf['victory_score'] > 0.5 and user_features['victory_score'] < 0.5:
                    continue
            else:
                if user_features['hand_raised'] > 0.5 or user_features['pointing_score'] > 0.5:
                    continue
                if face_ratio < 0.5:
                    continue

            total_score = face_sim + hand_sim
            if total_score > best_score:
                best_score = total_score
                best_meme = meme

        return best_meme, best_score

    def _sub_similarity(self, f1, f2, keys):
        key_map = {k: i for i, k in enumerate(self.feature_keys)}
        indices = [key_map[k] for k in keys]
        
        v1 = np.array([f1[k] for k in keys])
        v2 = np.array([f2[k] for k in keys])
        weights = self.feature_weights[indices]
        factors = self.feature_factors[indices]
        
        diff = np.abs(v1 - v2)
        sim = np.exp(-diff * factors)
        return float(np.sum(weights * sim))

    def _draw_dots_loader(self, draw, panel_x, panel_w, panel_h):
        """Renders an animated 3-dot loading indicator at the center of the meme panel."""
        t = time.time() * 5  # Speed of wave effect
        
        dot_radius = 8
        spacing = 28
        center_x = panel_x + (panel_w // 2)
        center_y = panel_h // 2

        # Draw 3 dots with staggered phase shifts for a smooth wave motion
        for i in range(3):
            # Calculate horizontal offset (-spacing, 0, +spacing)
            offset_x = (i - 1) * spacing
            cx = center_x + offset_x
            
            # Sinusoidal offset for vertical bounce + opacity pulse
            phase = t - (i * 0.5)
            bounce_y = int(6 * math.sin(phase))
            cy = center_y + bounce_y
            
            # Alpha pulses between 70 and 230
            alpha = int(150 + 80 * math.sin(phase))
            dot_color = (255, 255, 255, max(40, alpha))

            # Draw dot
            draw.ellipse(
                [cx - dot_radius, cy - dot_radius, cx + dot_radius, cy + dot_radius],
                fill=dot_color
            )

    def _draw_modern_ui(self, display_img, w, h, panel_w, best_meme, score, matched, user_features):
        """Renders an overlay with semi-translucent status pill and centered 3-dot loading animation."""
        img_pil = Image.fromarray(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)).convert("RGBA")
        overlay = Image.new("RGBA", img_pil.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        # 1. Floating User Status Pill (Top Left)
        if user_features is None:
            pill_text = "No face detected"
            pill_color = (239, 68, 68, 200)  # Red
        else:
            pill_text = "Tracking active"
            pill_color = (16, 185, 129, 180)  # Green

        tb = draw.textbbox((0, 0), pill_text, font=self.font_status)
        pw, ph = tb[2] - tb[0] + 24, tb[3] - tb[1] + 12
        draw.rounded_rectangle([16, 16, 16 + pw, 16 + ph], radius=12, fill=pill_color)
        draw.text((28, 20), pill_text, font=self.font_status, fill=(255, 255, 255, 255))

        # 2. Meme Area UI
        if not matched:
            # Display centered 3-dot animated wave loader while searching
            self._draw_dots_loader(draw, w, panel_w, h)

        # Composite overlay onto main image
        combined = Image.alpha_composite(img_pil, overlay)
        return cv2.cvtColor(np.array(combined.convert("RGB")), cv2.COLOR_RGB2BGR)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("\n Camera started! Press 'q' to quit\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            user_features = self.extract_face_features(frame)
            best_meme, score = self.find_best_match(user_features)
            matched = best_meme is not None and score >= self.match_threshold

            if matched:
                meme_img = best_meme['image']
                meme_h, meme_w = meme_img.shape[:2]
                scale = h / meme_h
                panel_w = int(meme_w * scale)
                meme_panel = cv2.resize(meme_img, (panel_w, h))
            else:
                panel_w = max(1, int(h * self.meme_aspect_ratio))
                # Dark slate modern background for placeholder
                meme_panel = np.full((h, panel_w, 3), (30, 27, 24), dtype=np.uint8)

            # Assemble split screen
            display = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
            display[:, :w] = frame
            display[:, w:w + panel_w] = meme_panel

            # Draw subtle vertical divider line between video feeds
            cv2.line(display, (w, 0), (w, h), (40, 40, 40), 2)

            # Apply UI rendering
            display = self._draw_modern_ui(display, w, h, panel_w, best_meme, score, matched, user_features)

            cv2.imshow("Meme Matcher", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Starting...\n")
    matcher = MemeMatcher(assets_folder="assets")
    matcher.run()