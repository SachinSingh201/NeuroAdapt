/**
 * EduMotion - Emotion Processing Module
 * 
 * Handles emotion normalization to strictly 4 classes:
 * engaged, bored, confused, frustrated.
 */

const EMOTION_MAP = {
    engaged:    "ENGAGED",
    bored:      "BORED",
    confused:   "CONFUSED",
    frustrated: "FRUSTRATED",

    // Aliases / Fallbacks
    happy:      "ENGAGED",
    focused:    "ENGAGED",
    calm:       "ENGAGED",
    neutral:    "ENGAGED",
    sad:        "BORED",
    tired:      "BORED",
    surprise:   "CONFUSED",
    angry:      "FRUSTRATED",
    stressed:   "FRUSTRATED",
    fear:       "FRUSTRATED",
    disgust:    "FRUSTRATED",
};

export function normalizeEmotionResponse(response) {
    if (!response) {
        return getDefaultEmotion();
    }

    const rawEmotion = (response.emotion || "engaged").toLowerCase();
    const confidence = Number(response.confidence || 0);
    const source = response.source || "unknown";
    const processingTime = Number(response.processing_time_ms || 0);
    const posture = response.posture || "upright";
    const postureConfidence = Number(response.posture_confidence || 0);

    const learningState = (response.learning_state || EMOTION_MAP[rawEmotion] || "ENGAGED").toUpperCase();

    return {
        emotion: rawEmotion,
        posture: posture,
        learning_state: learningState,
        confidence: confidence,
        posture_confidence: postureConfidence,
        theme: response.theme || "positive",
        background: response.background || rawEmotion,
        message: response.message || "",
        posture_tip: response.posture_tip || "",
        action: response.action || "maintain",
        assistance: response.assistance || "none",
        source: source,
        processing_time_ms: processingTime,
    };
}

export function getDefaultEmotion() {
    return {
        emotion: "engaged",
        posture: "upright",
        learning_state: "ENGAGED",
        confidence: 0,
        posture_confidence: 0,
        theme: "positive",
        background: "engaged",
        message: "Maintaining current state.",
        posture_tip: "Sit comfortably upright.",
        action: "maintain",
        assistance: "none",
        source: "none",
        processing_time_ms: 0,
    };
}

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