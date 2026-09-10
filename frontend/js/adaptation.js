/**
 * EduMotion - Adaptation Module
 * 
 * Handles UI adaptation based on emotion states,
 * including combination handling and priority systems.
 */

// Emotion combination configuration
// Keys are normalized (sorted alphabetically) so "focused+happy" and "happy+focused" match
const EMOTION_COMBINATIONS = {
    "confused+stressed": {
        assistance: "max",
        message: "Would you like help simplifying the content?",
        action: "show_help_card",
        density: "low",
        offer_simplify: true,
        offer_example: true,
        offer_pause: true,
    },
    "tired+confused": {
        assistance: "summary",
        message: "Would you like a shorter summary and a break?",
        action: "show_summary",
        density: "low",
        offer_break: true,
        summary_only: true,
    },
    "happy+focused": {
        assistance: "positive",
        message: "Excellent engagement! Keep going.",
        action: "none",
        density: "standard",
        animation: "gentle",
    },
    "focused+happy": {
        assistance: "positive",
        message: "Excellent engagement! Keep going.",
        action: "none",
        density: "standard",
        animation: "gentle",
    },
    "calm+focused": {
        assistance: "normal",
        message: "Focused learning mode active.",
        action: "none",
        density: "high",
    },
};

// Application state tracking for combinations
const AdaptationState = {
    // Track recent states for combination detection
    stateHistory: [],

    // Add a state to history
    addState: function (state, confidence) {
        this.stateHistory.push({
            state: state,
            confidence: confidence,
            timestamp: Date.now(),
        });

        // Keep only recent entries (window of 5)
        const windowSize = 5;
        if (this.stateHistory.length > windowSize) {
            this.stateHistory = this.stateHistory.slice(-windowSize);
        }
    },

    // Get the current adaptation mode based on all factors
    getCurrentAdaptation: function () {
        if (this.stateHistory.length === 0) {
            return this.getDefaultAdaptation();
        }

        // Get the most recent high-confidence state
        const highConfidenceStates = this.stateHistory.filter(
            (s) => s.confidence >= 0.55
        );

        if (highConfidenceStates.length === 0) {
            // Low confidence - maintain current/stable state
            return this.getDefaultAdaptation();
        }

        // Use the most recent high-confidence state
        const latest = highConfidenceStates[highConfidenceStates.length - 1];
        const state = latest.state;

        // Check for combinations
        const comboKey = this.getCombinationKey([state]);
        const combo = EMOTION_COMBINATIONS[comboKey];

        if (combo) {
            return {
                ...combo,
                primaryState: state,
                confidence: latest.confidence,
            };
        }

        // Single state adaptation
        return this.getSingleStateAdaptation(state);
    },

    // Get combination key (normalized/sorted)
    getCombinationKey: function (states) {
        if (!states || states.length === 0) {
            return "";
        }
        // Sort alphabetically so focused+happy === happy+focused
        const sorted = [...states].sort();
        return sorted.join("+");
    },

    // Get adaptation for a single state
    getSingleStateAdaptation: function (state) {
        const adaptations = {
            CALM: {
                message: "You're in a steady learning state.",
                theme: "normal",
                density: "standard",
                assistance: "low",
            },
            HAPPY: {
                message: "Great! You're engaged with the lesson.",
                theme: "positive",
                density: "standard",
                assistance: "low",
                animation: "gentle",
            },
            FOCUSED: {
                message: "You're focused. Keep going.",
                theme: "minimal",
                density: "high",
                assistance: "none",
                distractions: "hidden",
            },
            STRESSED: {
                message: "Take a moment. You can review this concept at your own pace.",
                theme: "calm",
                density: "low",
                assistance: "high",
                offer_break: true,
                simplified_view: true,
            },
            TIRED: {
                message: "You may want to take a short break before continuing.",
                theme: "soothing",
                density: "low",
                assistance: "medium",
                offer_break: true,
                summary_only: true,
            },
            CONFUSED: {
                message: "Would you like a simpler explanation?",
                theme: "help",
                density: "standard",
                assistance: "high",
                show_help_card: true,
                offer_simplify: true,
                offer_example: true,
            },
        };

        return adaptations[state] || adaptations.CALM;
    },

    // Get default adaptation when no states available
    getDefaultAdaptation: function () {
        return {
            message: "Maintaining current learning state.",
            theme: "normal",
            density: "standard",
            assistance: "low",
        };
    },
};

/**
 * Process adaptation based on emotion state and confidence.
 * 
 * @param {string} state - Learning state (CALM, HAPPY, etc.)
 * @param {number} confidence - Prediction confidence (0-1)
 * @param {array} recentStates - Array of recent states from temporal smoothing
 * @returns {object} Adaptation instructions for UI
 */
export function processAdaptation(state, confidence, recentStates) {
    // Add to history
    AdaptationState.addState(state, confidence);

    // Get current adaptation mode
    return AdaptationState.getCurrentAdaptation();
}

/**
 * Handle emotion combinations.
 * Multiple states can be active simultaneously.
 * 
 * @param {array} states - Array of learning states
 * @param {array} confidences - Corresponding confidence values
 * @returns {object} Combined adaptation instructions
 */
export function handleEmotionCombinations(states, confidences) {
    // Filter states by confidence
    const activeStates = states.filter(
        (_, i) => confidences[i] >= Config.MIN_CONFIDENCE
    );

    if (activeStates.length === 0) {
        return AdaptationState.getDefaultAdaptation();
    }

    // Get combination key (normalized)
    const comboKey = AdaptationState.getCombinationKey(activeStates);
    const combo = EMOTION_COMBINATIONS[comboKey];

    if (combo) {
        return {
            ...combo,
            primaryStates: activeStates,
        };
    }

    // No specific combination found, return individual state adaptations
    const adaptations = activeStates.map((state, i) => ({
        state,
        confidence: confidences[i],
        adaptation: AdaptationState.getSingleStateAdaptation(state),
    }));

    return {
        mode: "individual",
        adaptations,
    };
}

/**
 * Priority-based adaptation when multiple states are detected.
 * Uses priority system to determine which adaptation takes precedence.
 * 
 * @param {array} states - Array of detected learning states
 * @param {array} confidences - Corresponding confidence values
 * @returns {object} Priority-based adaptation result
 */
export function priorityBasedAdaptation(states, confidences) {
    // Priority order (highest to lowest)
    const priorityOrder = [
        "confused+stressed",
        "stressed",
        "tired+confused",
        "tired",
        "confused",
        "focused",
        "happy",
        "calm",
        "neutral",
    ];

    // Filter by confidence
    const activeStates = states.filter(
        (_, i) => confidences[i] >= Config.MIN_CONFIDENCE
    );

    if (activeStates.length === 0) {
        return AdaptationState.getDefaultAdaptation();
    }

    // Sort states by priority
    let primaryState = activeStates[0];

    for (const priority of priorityOrder) {
        // Check if this priority matches current states
        const priorityParts = priority.split("+");
        let matches = true;

        if (priorityParts.length === 1) {
            // Single state priority
            matches = activeStates.includes(priority);
        } else {
            // Combination priority
            matches = priorityParts.every((p) => activeStates.includes(p));
        }

        if (matches) {
            primaryState = priorityParts.length === 1 ? priority : priorityParts;
            break;
        }
    }

    // Get adaptation for the primary state
    return AdaptationState.getSingleStateAdaptation(primaryState);
}

// Make Config accessible (will be set from main app)
window.Config = {
    MIN_CONFIDENCE: 0.55,
};