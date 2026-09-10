import time
import logging
import sys
from typing import Dict, Any, Tuple, Optional
import numpy as np

# Ensure UTF-8 output encoding for console safety on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = logging.getLogger(__name__)

# Canonical mapping from DeepFace raw 7 emotions -> 4 EduMotion states
DEEPFACE_MAPPING = {
    "happy":    {"target": "engaged",    "base_weight": 0.90},
    "surprise": {"target": "engaged",    "base_weight": 0.70},
    "neutral":  {"target": "bored",      "base_weight": 0.60},  # Ambiguous: learning stats resolve Bored vs Engaged
    "sad":      {"target": "bored",      "base_weight": 0.55},
    "fear":     {"target": "confused",   "base_weight": 0.75},
    "disgust":  {"target": "frustrated", "base_weight": 0.80},
    "angry":    {"target": "frustrated", "base_weight": 0.90},
}

TARGET_STATES = ["engaged", "bored", "confused", "frustrated"]


class DeepFaceEmotionService:
    """
    Primary Emotion Detector powered by Python DeepFace.
    Combines 70% DeepFace facial emotion + 30% learning/activity statistics.
    Includes grace period fallback for when no face is detected.
    """

    def __init__(self, grace_period_seconds: float = 8.0):
        self.grace_period_seconds = grace_period_seconds
        self.last_valid_emotion: Optional[str] = "engaged"
        self.last_valid_confidence: float = 0.85
        self.last_valid_prob_dist: Dict[str, float] = {"happy": 0.80, "neutral": 0.20}
        self.last_valid_time: float = 0.0
        self.consecutive_no_face: int = 0
        self._deepface_ready = False
        self._init_deepface()

    def _init_deepface(self):
        try:
            from deepface import DeepFace
            # Preload emotion model in memory using facial_attribute task
            DeepFace.build_model(task="facial_attribute", model_name="Emotion")
            self._deepface_ready = True
            logger.info("[DeepFace] Emotion model preloaded successfully.")
        except Exception as e:
            logger.warning(f"[DeepFace] Model preload note: {e}")
            self._deepface_ready = True  # Will lazily build on first analyze

    def analyze_face(self, cv_image: np.ndarray) -> Tuple[Optional[str], float, Dict[str, float], bool]:
        """
        Runs DeepFace facial emotion analysis on the CV2 image frame.
        Returns: (raw_emotion, raw_confidence, emotion_distribution, face_detected)
        """
        if cv_image is None or cv_image.size == 0:
            return None, 0.0, {}, False

        try:
            from deepface import DeepFace

            # analyze with opencv detector backend; enforce_detection=False avoids exceptions
            results = DeepFace.analyze(
                img_path=cv_image,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv",
                silent=True,
            )

            if not results:
                return None, 0.0, {}, False

            data = results[0] if isinstance(results, list) else results

            region = data.get("region", {})
            face_w = region.get("w", 0)
            face_h = region.get("h", 0)
            face_conf = data.get("face_confidence", None)

            img_h, img_w = cv_image.shape[:2]
            # When opencv backend fails to find a face, enforce_detection=False returns region covering ~whole frame with face_confidence=0.0
            is_full_frame = (face_w >= img_w * 0.95 and face_h >= img_h * 0.95)
            
            if face_conf is not None and face_conf <= 0.0:
                face_detected = False
            elif is_full_frame:
                face_detected = False
            else:
                face_detected = (face_w >= 20 and face_h >= 20)

            dominant_emotion = str(data.get("dominant_emotion", "neutral")).lower()
            raw_emotions = data.get("emotion", {})

            # Normalize raw probabilities (percentages 0-100 -> fractions 0-1)
            total_prob = sum(float(v) for v in raw_emotions.values()) if raw_emotions else 0.0
            if total_prob > 0:
                prob_dist = {k.lower(): float(v) / total_prob for k, v in raw_emotions.items()}
            else:
                prob_dist = {dominant_emotion: 1.0}

            dominant_conf = float(prob_dist.get(dominant_emotion, 0.60))

            return dominant_emotion, dominant_conf, prob_dist, face_detected

        except Exception as e:
            logger.debug(f"[DeepFace] Face analysis note: {e}")
            return None, 0.0, {}, False

    def calculate_learning_scores(
        self,
        posture: str,
        posture_confidence: float = 1.0,
    ) -> Dict[str, float]:
        """
        Calculates learning statistics score vector across the 4 target states (30% weight).
        Incorporates ergonomic posture signals and study activity statistics.
        """
        posture = (posture or "upright").lower().strip()
        conf = max(0.2, min(1.0, float(posture_confidence or 0.8)))

        # Base learning score priors by posture
        if posture == "upright":
            # Upright posture is a strong signal of active, alert study engagement
            scores = {"engaged": 0.85 * conf, "bored": 0.10, "confused": 0.03, "frustrated": 0.02}
        elif posture in ("leaning_forward", "lean_forward"):
            # Leaning forward indicates high cognitive focus / reading
            scores = {"engaged": 0.50 * conf, "confused": 0.40 * conf, "bored": 0.05, "frustrated": 0.05}
        elif posture in ("leaning_backward", "lean_backward"):
            # Leaning back can indicate passive reviewing or drifting
            scores = {"engaged": 0.35 * conf, "bored": 0.50 * conf, "confused": 0.10, "frustrated": 0.05}
        elif posture in ("slouched", "slouch", "hunched"):
            # Slouched posture strongly correlates with fatigue / boredom
            scores = {"bored": 0.70 * conf, "frustrated": 0.20 * conf, "engaged": 0.05, "confused": 0.05}
        elif posture in ("leaning_left", "leaning_right"):
            # Side leaning indicates fatigue or restlessness
            scores = {"bored": 0.50 * conf, "confused": 0.30 * conf, "frustrated": 0.10, "engaged": 0.10}
        else:
            scores = {"engaged": 0.50, "bored": 0.30, "confused": 0.10, "frustrated": 0.10}

        # Normalize so sum(scores) == 1.0
        total = sum(scores.values())
        return {k: round(v / total, 4) for k, v in scores.items()}

    def evaluate_combined(
        self,
        cv_image: np.ndarray,
        posture: str = "upright",
        posture_confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Full evaluation pipeline:
        1. DeepFace facial emotion detection.
        2. Grace period / no face handling.
        3. 70% DeepFace + 30% Learning Stats scoring.
        4. Neutral ambiguity resolution via learning stats (Bored vs Engaged).
        """
        now = time.time()
        raw_df_emotion, df_conf, prob_dist, face_detected = self.analyze_face(cv_image)

        # ----------------------------------------------------
        # 1. Compute 30% Learning Statistics Scores first
        # ----------------------------------------------------
        learning_scores = self.calculate_learning_scores(posture, posture_confidence)

        # ----------------------------------------------------
        # 2. Face Detection & Grace Period Handling
        # ----------------------------------------------------
        is_grace_period = False
        face_status_log = "DETECTED"

        if face_detected and raw_df_emotion:
            # Valid face detected
            self.last_valid_emotion = raw_df_emotion
            self.last_valid_confidence = df_conf
            self.last_valid_prob_dist = prob_dist
            self.last_valid_time = now
            self.consecutive_no_face = 0

            dominant_face_emotion = raw_df_emotion
            face_confidence = df_conf
            active_prob_dist = prob_dist
            face_status_log = "DETECTED"
        else:
            self.consecutive_no_face += 1
            # Check grace period against last valid detection
            elapsed_since_valid = (now - self.last_valid_time) if self.last_valid_time > 0 else 999.0

            if elapsed_since_valid <= self.grace_period_seconds and self.last_valid_time > 0:
                is_grace_period = True
                dominant_face_emotion = self.last_valid_emotion or "engaged"
                face_confidence = max(0.35, round(self.last_valid_confidence * 0.85, 4))
                active_prob_dist = getattr(self, "last_valid_prob_dist", {dominant_face_emotion: 1.0})
                face_status_log = f"GRACE_PERIOD ({dominant_face_emotion})"
            else:
                # No face detected & grace period expired -> do not force facial emotion
                dominant_face_emotion = None
                face_confidence = 0.0
                active_prob_dist = {}
                face_status_log = "NO_FACE_DETECTED"

        # ----------------------------------------------------
        # 3. Compute 70% Facial Scores from DeepFace
        # ----------------------------------------------------
        facial_scores = {"engaged": 0.0, "bored": 0.0, "confused": 0.0, "frustrated": 0.0}

        if face_status_log in ("DETECTED",) or is_grace_period:
            if active_prob_dist:
                for emo_name, prob in active_prob_dist.items():
                    mapping = DEEPFACE_MAPPING.get(emo_name)
                    if not mapping:
                        continue
                    target = mapping["target"]
                    weight = mapping["base_weight"]

                    # Neutral ambiguity resolution: learning stats resolve Bored vs Engaged
                    if emo_name == "neutral":
                        if learning_scores.get("engaged", 0.0) >= learning_scores.get("bored", 0.0):
                            target = "engaged"
                        else:
                            target = "bored"

                    facial_scores[target] += prob * weight
            elif dominant_face_emotion:
                mapping = DEEPFACE_MAPPING.get(dominant_face_emotion, {"target": "engaged", "base_weight": 0.70})
                target = mapping["target"]
                if dominant_face_emotion == "neutral":
                    target = "engaged" if learning_scores.get("engaged", 0.0) >= learning_scores.get("bored", 0.0) else "bored"
                facial_scores[target] = face_confidence * mapping["base_weight"]

            # Normalize facial scores across the 4 states
            facial_total = sum(facial_scores.values())
            if facial_total > 0:
                facial_scores = {k: round(v / facial_total, 4) for k, v in facial_scores.items()}
            else:
                facial_scores = {"engaged": 0.25, "bored": 0.25, "confused": 0.25, "frustrated": 0.25}
        else:
            # NO_FACE_DETECTED -> do not force emotion, balanced neutral prior
            facial_scores = {"engaged": 0.25, "bored": 0.25, "confused": 0.25, "frustrated": 0.25}

        # ----------------------------------------------------
        # 4. Final Combined Scores: 70% DeepFace + 30% Learning Stats
        # ----------------------------------------------------
        final_scores = {}
        if face_status_log == "NO_FACE_DETECTED":
            # When no face is detected and grace period expired, fall back to default state driven by learning stats
            final_scores = {k: round(v, 4) for k, v in learning_scores.items()}
            final_emotion = max(final_scores, key=final_scores.get)
            final_confidence = final_scores[final_emotion]
        else:
            for state in TARGET_STATES:
                score = 0.70 * facial_scores.get(state, 0.0) + 0.30 * learning_scores.get(state, 0.0)
                final_scores[state] = round(score, 4)

            # Pick winning final emotion
            final_emotion = max(final_scores, key=final_scores.get)
            final_confidence = final_scores[final_emotion]

        return {
            "deepface_raw_emotion": dominant_face_emotion,
            "deepface_confidence": round(face_confidence, 4),
            "face_detected": face_detected,
            "is_grace_period": is_grace_period,
            "face_status_log": face_status_log,
            "facial_scores": facial_scores,
            "learning_scores": learning_scores,
            "final_scores": final_scores,
            "final_emotion": final_emotion,
            "final_confidence": round(final_confidence, 4),
            "posture": posture,
            "posture_confidence": round(float(posture_confidence), 4),
        }
