from ..config import Config


class AdaptationService:
    @staticmethod
    def map_to_learning_state(raw_emotion: str, source: str):
        """
        Maps a raw model output to a learning state.

        Routing:
          source == "cloud"  → POSTURE_MAPPING  (Roboflow posture classes)
          source == "local"  → LOCAL_MODEL_MAPPING
          else               → CANONICAL_MAPPING (generic emotion labels)
        """
        raw_emotion = raw_emotion.lower()

        # Select correct mapping based on source
        if source == "cloud":
            mapping = Config.POSTURE_MAPPING
        elif source == "local":
            mapping = Config.LOCAL_MODEL_MAPPING
        else:
            mapping = Config.CANONICAL_MAPPING

        # Direct lookup
        state = mapping.get(raw_emotion)

        # Fallback: if the label is already a known learning state
        if not state:
            if raw_emotion.upper() in Config.LEARNING_STATES:
                return raw_emotion.upper()
            return "CALM"

        return state

    @staticmethod
    def get_adaptation_ui_changes(state: str, confidence: float):
        """
        Returns UI adaptation instructions for the given learning state.
        """
        if confidence < Config.MIN_CONFIDENCE:
            return {
                "message": "Maintaining current state (low confidence)",
                "action": "none",
                "ui_overlay": None,
            }

        adaptations = {
            "CALM": {
                "message": "You're in a steady learning state.",
                "theme": "normal",
                "density": "standard",
                "assistance": "low",
                "posture_tip": None,
            },
            "HAPPY": {
                "message": "Great! You're engaged with the lesson.",
                "theme": "positive",
                "density": "standard",
                "assistance": "low",
                "animation": "gentle",
                "posture_tip": None,
            },
            "FOCUSED": {
                "message": "Great posture! You're upright and focused - keep it up.",
                "theme": "minimal",
                "density": "high",
                "assistance": "none",
                "distractions": "hidden",
                "posture_tip": "Upright posture detected",
            },
            "STRESSED": {
                "message": "Take a moment. You can review this concept at your own pace.",
                "theme": "calm",
                "density": "low",
                "assistance": "high",
                "offer_break": True,
                "simplified_view": True,
                "posture_tip": None,
            },
            "TIRED": {
                "message": "Poor posture detected - try sitting upright for better focus and energy.",
                "theme": "soothing",
                "density": "low",
                "assistance": "medium",
                "offer_break": True,
                "summary_only": True,
                "posture_tip": "Sit upright: back straight, feet flat on the floor.",
            },
            "CONFUSED": {
                "message": "Leaning forward detected - adjust your screen distance and sit back.",
                "theme": "help",
                "density": "standard",
                "assistance": "high",
                "show_help_card": True,
                "offer_simplify": True,
                "offer_example": True,
                "posture_tip": "Ideal screen distance: arm's length away.",
            },
        }

        return adaptations.get(state, adaptations["CALM"])

    @staticmethod
    def handle_combinations(current_states: list):
        """
        Handles combinations of states for advanced adaptation.
        """
        if not current_states:
            return "CALM"

        combo_key = "+".join(sorted(list(set(current_states))))

        combinations = {
            "CONFUSED+STRESSED":  "MAX_ASSISTANCE",
            "CONFUSED+TIRED":     "SIMPLIFIED_SUMMARY",
            "FOCUSED+HAPPY":      "DEEP_LEARNING",
            "CALM+FOCUSED":       "NORMAL_FOCUSED",
            "TIRED+CONFUSED":     "SIMPLIFIED_SUMMARY",
        }

        return combinations.get(combo_key, current_states[0])