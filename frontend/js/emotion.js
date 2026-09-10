/**
 * EduMotion - Emotion Processing Module
 * 
 * Handles emotion detection, normalization, and state mapping.
 * Communication with backend via REST API.
 */

// Emotion-to-learning-state mapping configuration
// Enforcing strictly 4 states: ENGAGED, BORED, CONFUSED, FRUSTRATED
const EMOTION_MAP = {
    // ── Canonical face-emotion labels (fallback) ──────────────
    neutral:  "ENGAGED",
    happy:    "ENGAGED",
    sad:      "BORED",
    angry:    "FRUSTRATED",
    fear:     "FRUSTRATED",
    disgust:  "FRUSTRATED",
    surprise: "CONFUSED",

    // ── Local model emotions ───────────────────────────────────
    engaged:    "ENGAGED",
    bored:      "BORED",
    confused:   "CONFUSED",
    frustrated: "FRUSTRATED",

    // ── Pass-through (model already outputs our states) ───────
    ENGAGED:    "ENGAGED",
    BORED:      "BORED",
    CONFUSED:   "CONFUSED",
    FRUSTRATED: "FRUSTRATED"
};

/**
 * Normalize emotion response from either cloud API or local model
 * to a canonical format.
 *
 * @param {object} response - Raw response from backend API
 * @returns {object} Normalized emotion object
 */
export function normalizeEmotionResponse(response) {
    if (!response) {
        return getDefaultEmotion();
    }

    const rawEmotion = (response.emotion || response.label || "neutral").toLowerCase();
    const confidence = Number(response.confidence || response.score || 0);
    const source = response.source || "unknown";
    const processingTime = Number(response.processing_time_ms || 0);

    // Pass raw_posture through if backend returns it
    const rawPosture = response.posture || null;

    // Determine learning state through mapping
    let learningState = mapRawEmotionToLearningState(rawEmotion, source);

    // If confidence is too low, default to ENGAGED
    if (confidence < 0.55) {
        learningState = "ENGAGED";
    }

    return {
        raw_emotion: rawEmotion,
        raw_posture: rawPosture,
        learning_state: learningState,
        confidence: confidence,
        source: source || "unknown",
        timestamp: response.timestamp || new Date().toISOString(),
        processing_time_ms: processingTime,
    };
}

/**
 * Map a raw emotion from the model to a learning state.
 *
 * @param {string} rawEmotion - Raw emotion from model
 * @param {string} source - Inference source (cloud, local)
 * @returns {string} Mapped learning state
 */
export function mapRawEmotionToLearningState(rawEmotion, source) {
    // Try direct match first
    if (EMOTION_MAP[rawEmotion]) {
        return EMOTION_MAP[rawEmotion];
    }

    // Try capitalized version
    const capitalized = rawEmotion.charAt(0).toUpperCase() + rawEmotion.slice(1);
    if (EMOTION_MAP[capitalized]) {
        return EMOTION_MAP[capitalized];
    }

    // Fallback
    return "ENGAGED";
}

/**
 * Get default emotion object when no response available
 * @returns {object} Default emotion object
 */
export function getDefaultEmotion() {
    return {
        raw_emotion: "neutral",
        raw_posture: null,
        learning_state: "ENGAGED",
        confidence: 0,
        source: "none",
        timestamp: new Date().toISOString(),
        processing_time_ms: 0,
    };
}

/**
 * Predict emotion from base64 image by calling the backend API.
 * 
 * @param {string} base64Image - Base64 encoded JPEG image
 * @returns {Promise<object>} Normalized emotion prediction
 */
export async function predictEmotionFromBase64(base64Image) {
    try {
        const response = await fetch("/api/emotion/predict-base64", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                image: base64Image,
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const result = await response.json();
        return normalizeEmotionResponse(result);
    } catch (err) {
        console.error("Error predicting emotion:", err);
        return getDefaultEmotion();
    }
}

/**
 * Predict emotion state manually (for demo/manual mode).
 * 
 * @param {string} state - Emotion state to set
 * @returns {Promise<object>} Normalized emotion object
 */
export async function predictEmotionManual(state) {
    try {
        const response = await fetch("/api/emotion/mode", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ mode: "manual", state: state }),
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const result = await response.json();
        return normalizeEmotionResponse({
            emotion: state,
            confidence: 1.0,
            source: "manual",
            ...result,
        });
    } catch (err) {
        console.error("Error setting manual emotion:", err);
        return {
            raw_emotion: state,
            learning_state: mapRawEmotionToLearningState(state, "manual"),
            confidence: 1.0,
            source: "manual",
            timestamp: new Date().toISOString(),
            processing_time_ms: 0,
        };
    }
}