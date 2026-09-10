import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Config:
    # Project Info
    PROJECT_NAME = os.getenv("PROJECT_NAME", "EduMotion")
    VERSION = os.getenv("VERSION", "1.0.0")
    DEBUG = os.getenv("DEBUG_MODE", "true").lower() == "true"

    # Emotion Service Mode
    # Possible values: "cloud", "local", "auto"
    EMOTION_MODE = os.getenv("EMOTION_MODE", "auto")

    # --------------------------------------------------------
    # Roboflow Posture Detection API (cloud service)
    # --------------------------------------------------------
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "YClVCZs2pjD71cFXfsNm")
    ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "sachin-singh-g50va")
    ROBOFLOW_WORKFLOW_ID = os.getenv(
        "ROBOFLOW_WORKFLOW_ID",
        "sitting-posture-classification-6vwq1-kbzhk"
    )
    ROBOFLOW_API_URL = (
        f"https://serverless.roboflow.com/infer/workflows/"
        f"{os.getenv('ROBOFLOW_WORKSPACE', 'sachin-singh-g50va')}/"
        f"{os.getenv('ROBOFLOW_WORKFLOW_ID', 'sitting-posture-classification-6vwq1-kbzhk')}"
    )

    # Legacy cloud API (kept for compatibility, unused when Roboflow is active)
    CLOUD_API_URL = os.getenv("EMOTION_API_URL", "")
    CLOUD_API_KEY = os.getenv("EMOTION_API_KEY", "")
    CLOUD_API_METHOD = os.getenv("EMOTION_API_METHOD", "POST")
    CLOUD_API_IMAGE_FIELD = os.getenv("EMOTION_API_IMAGE_FIELD", "image")

    # Local Model Configuration
    LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "emotion_cnn.pkl")

    # Processing Settings
    PROCESSING_INTERVAL_MS = int(os.getenv("PROCESSING_INTERVAL_MS", 1500))
    MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", 0.40))
    STABLE_PREDICTIONS_REQUIRED = int(os.getenv("STABLE_PREDICTIONS_REQUIRED", 3))
    TEMPORAL_WINDOW_SIZE = int(os.getenv("TEMPORAL_WINDOW_SIZE", 5))

    # --------------------------------------------------------
    # Posture → Learning State Mapping
    # (used when source == "cloud", i.e. Roboflow posture model)
    # --------------------------------------------------------
    POSTURE_MAPPING = {
        "upright":          "FOCUSED",    # Good posture → focused & attentive
        "leaning_forward":  "CONFUSED",   # Leaning into screen → confused/straining
        "leaning_backward": "TIRED",      # Slouching back → disengaged/tired
        "leaning_left":     "TIRED",      # Sideways lean → uncomfortable/tired
        "leaning_right":    "TIRED",      # Sideways lean → uncomfortable/tired
    }

    # Emotion Mappings for local model (Raw Model Output → Learning State)
    CANONICAL_MAPPING = {
        "neutral":  "CALM",
        "happy":    "HAPPY",
        "sad":      "TIRED",
        "angry":    "STRESSED",
        "fear":     "STRESSED",
        "disgust":  "STRESSED",
        "surprise": "CONFUSED",
    }

    # Mapping for local model specific emotions (from notebook)
    LOCAL_MODEL_MAPPING = {
        "engaged":    "FOCUSED",
        "bored":      "TIRED",
        "confused":   "CONFUSED",
        "frustrated": "STRESSED",
    }

    # All possible learning states
    LEARNING_STATES = [
        "CALM", "HAPPY", "FOCUSED", "STRESSED", "TIRED", "CONFUSED"
    ]

    # Course Metadata (Configurable)
    COURSE_NAME = "Introduction to Machine Learning"
    LESSON_TITLE = "Foundations of AI & ML"
    LESSON_DESCRIPTION = "Explore the fundamental concepts of machine learning, from supervised learning to neural networks."
    YOUTUBE_VIDEO_URL = "https://www.youtube.com/embed/HcqpanDadyQ"
