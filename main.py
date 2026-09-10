import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pickle
import os
import subprocess

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

        # Expanded feature set including granular hand/gesture metrics
        self.feature_keys = [
            'surprise_score', 'smile_score', 'concern_score', 'cheers_score', 
            'hand_raised', 'num_hands', 'eye_openness', 'eyes_symmetry', 
            'mouth_openness', 'mouth_width_ratio', 'mouth_elevation', 
            'eyebrow_height', 'brow_symmetry',
            # New gesture features:
            'pointing_score', 'thumbs_up_score', 'victory_score', 'hand_near_face'
        ]

        # Configured weights and scaling factors for similarity matching
        self.feature_weights = np.array([25, 20, 20, 30, 25, 15, 20, 10, 25, 20, 15, 20, 10, 30, 30, 30, 25])
        self.feature_factors = np.array([10, 10, 10, 10, 15, 15, 5, 5, 5, 5, 5, 5, 5, 12, 12, 12, 12])

        self.load_memes(assets_folder)

        if self.memes:
            aspects = [m['image'].shape[1] / m['image'].shape[0] for m in self.memes]
            self.meme_aspect_ratio = float(np.mean(aspects))
        else:
            self.meme_aspect_ratio = 1.0

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
        # Eye Aspect Ratio
        def ear(upper, lower):
            vertical = np.linalg.norm(landmark_array[upper] - landmark_array[lower], axis=1).mean()
            horizontal = np.linalg.norm(landmark_array[upper[0]] - landmark_array[upper[-1]])
            return vertical / (horizontal + 1e-6)

        left_ear = ear(self.LEFT_EYE_UPPER, self.LEFT_EYE_LOWER)
        right_ear = ear(self.RIGHT_EYE_UPPER, self.RIGHT_EYE_LOWER)
        avg_ear = (left_ear + right_ear) / 2.0

        # Mouth
        mouth_top, mouth_bottom = landmark_array[13], landmark_array[14]
        mouth_height = np.linalg.norm(mouth_top - mouth_bottom)
        mouth_left, mouth_right = landmark_array[61], landmark_array[291]
        mouth_width = np.linalg.norm(mouth_left - mouth_right)
        mouth_ar = mouth_height / (mouth_width + 1e-6)
        inner_width = np.linalg.norm(landmark_array[78] - landmark_array[308])
        mouth_width_ratio = inner_width / (mouth_width + 1e-6)

        # Eyebrows
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

        # Process Hand Landmarks & Specific Gestures
        num_hands = len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0
        hand_raised = 0.0
        pointing_score = 0.0
        thumbs_up_score = 0.0
        victory_score = 0.0
        hand_near_face = 0.0

        face_center = landmark_array.mean(axis=0)

        if num_hands > 0:
            for hand_landmarks in hand_result.hand_landmarks:
                # Convert normalized landmarks to numpy array (N x 2)
                pts = np.array([[lm.x, lm.y] for lm in hand_landmarks])

                # Determine extended state for each finger
                index_extended = pts[self.INDEX_TIP, 1] < pts[self.INDEX_PIP, 1]
                middle_extended = pts[self.MIDDLE_TIP, 1] < pts[self.MIDDLE_PIP, 1]
                ring_extended = pts[self.RING_TIP, 1] < pts[self.RING_PIP, 1]
                pinky_extended = pts[self.PINKY_TIP, 1] < pts[self.PINKY_PIP, 1]
                thumb_extended = pts[self.THUMB_TIP, 1] < pts[self.THUMB_IP, 1]

                # Check general hand elevation relative to face
                face_top = landmark_array[:, 1].min()
                if pts[self.WRIST, 1] < face_center[1] + 0.15 or pts[self.MIDDLE_TIP, 1] < face_top + 0.1:
                    hand_raised = 1.0

                # Measure hand proximity to face (for facepalm / thinking memes)
                min_dist_to_face = np.min(np.linalg.norm(pts - face_center, axis=1))
                if min_dist_to_face < 0.25:
                    hand_near_face = 1.0

                # Pointing gesture detection (Index extended, others folded)
                if index_extended and not middle_extended and not ring_extended and not pinky_extended:
                    pointing_score = 1.0

                # Thumbs-up detection (Thumb extended upward, all other fingers folded)
                if thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
                    thumbs_up_score = 1.0

                # Victory/Peace sign detection (Index and Middle extended)
                if index_extended and middle_extended and not ring_extended and not pinky_extended:
                    victory_score = 1.0

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
            # Gesture features
            'pointing_score': pointing_score,
            'thumbs_up_score': thumbs_up_score,
            'victory_score': victory_score,
            'hand_near_face': hand_near_face
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
        scores = np.array([self.compute_similarity(user_features, mf) for mf in self.meme_features])
        if len(scores) == 0:
            return None, 0.0
        best_match_idx = np.argmax(scores)
        return self.memes[best_match_idx], scores[best_match_idx]

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("\n🎥 Camera started! Press 'q' to quit\n")

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
                meme_panel = np.full((h, panel_w, 3), 25, dtype=np.uint8)
                placeholder = "..."
                (tw, th), _ = cv2.getTextSize(placeholder, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
                cv2.putText(
                    meme_panel, placeholder,
                    ((panel_w - tw) // 2, (h + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (90, 90, 90), 3
                )

            display = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
            display[:, :w] = frame
            display[:, w:w + panel_w] = meme_panel

            cv2.rectangle(display, (5, 5), (200, 45), (0, 0, 0), -1)
            cv2.putText(display, "YOU", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            if user_features is None:
                cv2.putText(display, "No face detected", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if matched:
                cv2.rectangle(display, (w + 5, 5), (w + panel_w - 5, 75), (0, 0, 0), -1)
                cv2.putText(display, best_meme['name'], (w + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display, f"Match: {score:.1f}", (w + 10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.rectangle(display, (w + 5, 5), (w + panel_w - 5, 45), (0, 0, 0), -1)
                readout = f"Waiting... ({score:.0f}/{self.match_threshold:.0f})" if best_meme else "Waiting for face..."
                cv2.putText(display, readout, (w + 10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

            cv2.imshow("Meme Matcher - Press Q to quit ^^", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Starting...\n")
    matcher = MemeMatcher(assets_folder="assets")
    matcher.run()