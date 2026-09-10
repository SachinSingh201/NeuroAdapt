/**
 * EduMotion - Learning Page Module (Posture Detection)
 *
 * LAYOUT:
 *  - #youtube-iframe  → LEFT section (ML video, always visible, never touched)
 *  - #webcam-video    → RIGHT panel  (independent webcam stream)
 *
 * FLOW:
 *  Clicking "Start Camera" opens the webcam AND auto-starts posture detection
 *  after a 1-second warmup (gives the browser time to set videoWidth).
 *  Every 5 seconds a JPEG frame is captured and POSTed to /api/emotion/predict-base64.
 */

// ─────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────

const LessonConfig = {
    processingInterval:  5000,   // ms between posture checks (5 seconds)
    videoWarmupDelay:    1200,   // ms to wait after camera opens before first capture
    maxVideoWaitMs:      8000,   // max ms to wait for videoWidth > 0
};

// ─────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────

const LearningState = {
    isCameraActive:       false,
    isProcessing:         false,
    isProcessingRequest:  false,   // prevent concurrent API calls
    videoStream:          null,
    processingIntervalId: null,
    postureHistory:       [],      // rolling window of {state, confidence}
    frameCount:           0,       // for debug logging
};

// ─────────────────────────────────────────────────────────────
// Posture display config
// ─────────────────────────────────────────────────────────────

const POSTURE_ICONS = {
    ENGAGED:  "🎯",
    BORED:    "😑",
    CONFUSED: "🤔",
    FRUSTRATED: "😤",
};

const POSTURE_RAW_LABELS = {
    upright:          "Upright",
    leaning_forward:  "Leaning Forward",
    leaning_backward: "Leaning Backward",
    leaning_left:     "Leaning Left",
    leaning_right:    "Leaning Right",
    slouched:         "Slouched",
};

const POSTURE_ICONS_BY_NAME = {
    upright:          "🧍",
    leaning_forward:  "📐",
    leaning_backward: "🪑",
    leaning_left:     "↙️",
    leaning_right:    "↘️",
    slouched:         "🛋️",
};

const POSTURE_ALERTS = {
    ENGAGED:  { msg: "✓ Engaged — keep it up!",                                           cls: "alert-focused"  },
    BORED:    { msg: "⚠ Looks like you're losing interest — want to take a break?",               cls: "alert-tired"    },
    CONFUSED: { msg: "🤔 Confused — let's review the current topic.",           cls: "alert-confused" },
    FRUSTRATED: { msg: "😟 Frustrated — take a deep breath and relax.",        cls: "alert-stressed" },
};

// ─────────────────────────────────────────────────────────────
// Emotion Theme System
// Maps backend learning states → 4 frontend display emotions.
// Only these 4 exist per the backend: engaged, bored, confused, frustrated.
// ─────────────────────────────────────────────────────────────

/**
 * Maps backend learning states to frontend emotion keys.
 * Each state gets its own distinct color theme for maximum visibility.
 * Keys drive body[data-emotion] CSS variable and dropdown indicator.
 */
const LEARNING_STATE_TO_EMOTION = {
    ENGAGED:    "engaged",
    BORED:      "bored",
    CONFUSED:   "confused",
    FRUSTRATED: "frustrated",
};

// Icons & display names for all emotion keys
const EMOTION_DISPLAY = {
    engaged:    { icon: "🎯", label: "Engaged" },
    bored:      { icon: "😑", label: "Bored" },
    confused:   { icon: "🤔", label: "Confused" },
    frustrated: { icon: "😤", label: "Frustrated" },
};

/** State for the manual emotion override (null = auto mode) */
let manualEmotionOverride = null;

/**
 * Apply an emotion theme to the whole page.
 * Sets body[data-emotion] which is picked up by CSS.
 * If manualEmotionOverride is set, it takes precedence.
 * @param {string|null} autoEmotion - derived from backend learning state, or null to clear
 */
function applyEmotionTheme(autoEmotion) {
    const emotion = manualEmotionOverride || autoEmotion || "none";
    document.body.setAttribute("data-emotion", emotion);
    updateEmotionButtonIndicator(emotion, manualEmotionOverride !== null);
}

/**
 * Update the navbar Emotion button to show the active emotion.
 */
function updateEmotionButtonIndicator(emotion, isManual) {
    const btn      = document.getElementById("emotion-toggle-btn");
    const iconEl   = document.getElementById("emotion-btn-icon");
    const labelEl  = document.getElementById("emotion-btn-label");
    if (!btn) return;

    const info = EMOTION_DISPLAY[emotion];
    if (info) {
        if (iconEl)  iconEl.textContent  = info.icon;
        if (labelEl) labelEl.textContent = info.label;
        if (isManual) {
            btn.classList.add("has-manual");
            btn.title = `Manual override: ${info.label}`;
        } else {
            btn.classList.remove("has-manual");
            btn.title = `Detected emotion: ${info.label}`;
        }
    } else {
        if (iconEl)  iconEl.textContent  = "🎭";
        if (labelEl) labelEl.textContent = "Emotion";
        btn.classList.remove("has-manual");
        btn.title = "Emotion selector";
    }
}

// ─────────────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    setupCameraControls();
    setupEventListeners();
    setupEmotionDropdown();
    setupYouTubePlayer();
    setupFocusMode();
    console.log("[EduMotion] Dashboard ready. Click 'Start Camera' to begin posture detection.");
});

// ─────────────────────────────────────────────────────────────
// Wire buttons
// ─────────────────────────────────────────────────────────────

function setupCameraControls() {
    // "Start Camera" button → starts camera AND auto detection
    const startCameraBtn = document.getElementById("start-processing");
    const stopCameraBtn  = document.getElementById("stop-processing");
    if (startCameraBtn) startCameraBtn.addEventListener("click", startCameraAndDetection);
    if (stopCameraBtn)  stopCameraBtn.addEventListener("click",  stopCamera);

    // "START AUTO DETECTION" button (posture monitor card)
    const startAutoBtn = document.getElementById("start-auto");
    const stopAutoBtn  = document.getElementById("stop-auto");
    if (startAutoBtn) startAutoBtn.addEventListener("click", startAutoDetection);
    if (stopAutoBtn)  stopAutoBtn.addEventListener("click",  stopAutoDetection);

    // Duplicate buttons (manual card)
    const startAuto2 = document.getElementById("start-auto-2");
    const stopAuto2  = document.getElementById("stop-auto-2");
    if (startAuto2) startAuto2.addEventListener("click", startAutoDetection);
    if (stopAuto2)  stopAuto2.addEventListener("click",  stopAutoDetection);
}

// ─────────────────────────────────────────────────────────────
// startCameraAndDetection
// Opens webcam AND auto-starts detection after warmup.
// This is the MAIN entry point for the "Start Camera" button.
// ─────────────────────────────────────────────────────────────

async function startCameraAndDetection() {
    setStatus("Opening camera…");
    await startCamera();
    if (!LearningState.isCameraActive) return;

    // Wait for the video to actually report dimensions (videoWidth > 0)
    setStatus("Warming up camera…");
    const ready = await waitForVideoReady();
    if (!ready) {
        setStatus("Camera stream not ready. Try stopping and restarting.");
        console.warn("[EduMotion] Video stream never became ready.");
        return;
    }

    setStatus("Camera ready ✓ — starting posture detection…");
    await startAutoDetection();
}

// ─────────────────────────────────────────────────────────────
// waitForVideoReady — polls until videoWidth > 0 or timeout
// ─────────────────────────────────────────────────────────────

function waitForVideoReady() {
    return new Promise((resolve) => {
        const video = document.getElementById("webcam-video");
        if (!video) { resolve(false); return; }
        if (video.videoWidth > 0) { resolve(true); return; }

        const deadline = Date.now() + LessonConfig.maxVideoWaitMs;
        const poll = setInterval(() => {
            if (video.videoWidth > 0) {
                clearInterval(poll);
                console.log(`[EduMotion] Video ready: ${video.videoWidth}×${video.videoHeight}`);
                resolve(true);
            } else if (Date.now() > deadline) {
                clearInterval(poll);
                resolve(false);
            }
        }, 100);
    });
}

// ─────────────────────────────────────────────────────────────
// Start Camera — opens webcam stream only
// ─────────────────────────────────────────────────────────────

async function startCamera() {
    if (LearningState.isCameraActive) return;

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        });

        LearningState.videoStream    = stream;
        LearningState.isCameraActive = true;

        const webcamVideo = document.getElementById("webcam-video");
        const placeholder = document.getElementById("camera-placeholder");
        const badge       = document.getElementById("camera-badge");

        if (webcamVideo) {
            webcamVideo.srcObject = stream;
            webcamVideo.style.display = "block";
        }
        if (placeholder) placeholder.style.display = "none";
        if (badge)       badge.style.display        = "inline-block";

        setElDisplay("start-processing", false);
        setElDisplay("stop-processing",  true);

        console.log("[EduMotion] Camera opened successfully.");

    } catch (err) {
        console.error("[EduMotion] Camera access denied:", err);
        setStatus("❌ Camera permission denied. Allow access in browser settings.");
        alert(
            "Camera permission is required for posture detection.\n" +
            "Please allow camera access in your browser settings and refresh."
        );
    }
}

// ─────────────────────────────────────────────────────────────
// Stop Camera — stops webcam AND processing
// ─────────────────────────────────────────────────────────────

function stopCamera() {
    stopAutoDetection();   // stop AI first

    if (LearningState.videoStream) {
        LearningState.videoStream.getTracks().forEach((t) => t.stop());
        LearningState.videoStream = null;
    }
    LearningState.isCameraActive = false;
    LearningState.frameCount     = 0;

    const webcamVideo = document.getElementById("webcam-video");
    const placeholder = document.getElementById("camera-placeholder");
    const badge       = document.getElementById("camera-badge");

    if (webcamVideo) { webcamVideo.srcObject = null; webcamVideo.style.display = "none"; }
    if (placeholder) placeholder.style.display = "flex";
    if (badge)       badge.style.display        = "none";

    setElDisplay("start-processing", true);
    setElDisplay("stop-processing",  false);

    resetPostureDisplay();
    setStatus("");
    console.log("[EduMotion] Camera stopped.");
}

// ─────────────────────────────────────────────────────────────
// Start Auto Detection
// ─────────────────────────────────────────────────────────────

async function startAutoDetection() {
    if (LearningState.isProcessing) return;

    // If camera not active, open it first
    if (!LearningState.isCameraActive) {
        await startCamera();
        if (!LearningState.isCameraActive) return;
        await waitForVideoReady();
    }

    LearningState.isProcessing = true;

    setElDisplay("start-auto",   false);
    setElDisplay("stop-auto",    true);
    setElDisplay("start-auto-2", false);
    setElDisplay("stop-auto-2",  true);

    setStatus("🔄 Posture detection active — capturing every 5 seconds…");
    console.log(`[EduMotion] Auto detection started. Interval: ${LessonConfig.processingInterval}ms`);

    startProcessingLoop();
}

// ─────────────────────────────────────────────────────────────
// Stop Auto Detection — keeps camera ON
// ─────────────────────────────────────────────────────────────

function stopAutoDetection() {
    if (!LearningState.isProcessing && !LearningState.processingIntervalId) return;

    LearningState.isProcessing = false;

    if (LearningState.processingIntervalId) {
        clearInterval(LearningState.processingIntervalId);
        LearningState.processingIntervalId = null;
    }

    setElDisplay("start-auto",   true);
    setElDisplay("stop-auto",    false);
    setElDisplay("start-auto-2", true);
    setElDisplay("stop-auto-2",  false);

    if (LearningState.isCameraActive) {
        setStatus("⏸ Detection paused. Camera still active.");
    }
    console.log("[EduMotion] Auto detection stopped.");
}

// ─────────────────────────────────────────────────────────────
// Processing loop — fires every 5 seconds
// ─────────────────────────────────────────────────────────────

function startProcessingLoop() {
    // First capture after warmup delay, then every interval
    setTimeout(() => {
        if (!LearningState.isProcessing) return;
        processNextFrame();   // immediate first capture

        LearningState.processingIntervalId = setInterval(
            processNextFrame,
            LessonConfig.processingInterval
        );
    }, LessonConfig.videoWarmupDelay);
}

// ─────────────────────────────────────────────────────────────
// Capture frame and send to backend
// ─────────────────────────────────────────────────────────────

async function processNextFrame() {
    if (!LearningState.isProcessing)       return;
    if (LearningState.isProcessingRequest) return;   // skip if previous request still in flight

    const video = document.getElementById("webcam-video");
    if (!video) { console.warn("[EduMotion] #webcam-video not found"); return; }
    if (!video.srcObject) { console.warn("[EduMotion] No srcObject on webcam"); return; }
    if (video.videoWidth === 0) {
        console.warn("[EduMotion] videoWidth still 0, skipping frame");
        return;
    }

    LearningState.isProcessingRequest = true;
    LearningState.frameCount++;
    const frameNum = LearningState.frameCount;

    console.log(`[EduMotion] ▶ Capturing frame #${frameNum} (${video.videoWidth}×${video.videoHeight}) → sending to backend…`);
    setStatus(`📷 Capturing frame #${frameNum}…`);

    try {
        // Draw frame to hidden canvas
        const canvas = document.getElementById("capture-canvas");
        if (!canvas) { console.error("[EduMotion] #capture-canvas not found"); return; }

        canvas.width  = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        if (!ctx) { console.error("[EduMotion] Could not get canvas context"); return; }
        ctx.drawImage(video, 0, 0);

        // JPEG at 80% quality — good balance of size vs quality
        const base64Image = canvas.toDataURL("image/jpeg", 0.80);
        const approxKB    = Math.round(base64Image.length * 0.75 / 1024);
        console.log(`[EduMotion]   Image size: ~${approxKB} KB — POSTing to /api/emotion/predict-base64`);

        await sendFrameToBackend(base64Image, frameNum);

    } catch (err) {
        console.error(`[EduMotion] Frame #${frameNum} error:`, err);
    } finally {
        LearningState.isProcessingRequest = false;
    }
}

// ─────────────────────────────────────────────────────────────
// POST to backend
// ─────────────────────────────────────────────────────────────

async function sendFrameToBackend(base64Image, frameNum) {
    const t0 = performance.now();

    try {
        const response = await fetch("/api/emotion/predict-base64", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ image: base64Image }),
        });

        const elapsed = Math.round(performance.now() - t0);

        if (!response.ok) {
            const errText = await response.text();
            console.error(`[EduMotion] Backend returned HTTP ${response.status}: ${errText}`);
            setStatus(`❌ Backend error: HTTP ${response.status}`);
            return;
        }

        const result = await response.json();
        console.log(`[EduMotion] ✅ Frame #${frameNum} response in ${elapsed}ms:`, result);
        setStatus(`✅ Frame #${frameNum} processed (${elapsed}ms) — ${result.learning_state || result.emotion || "?"}`);

        updatePostureUI(result);

    } catch (err) {
        console.error(`[EduMotion] Network error on frame #${frameNum}:`, err);
        setStatus(`❌ Network error — is the backend running?`);
    }
}

// ─────────────────────────────────────────────────────────────
// UI updates
// ─────────────────────────────────────────────────────────────

function updatePostureUI(result) {
    const rawPosture    = result.emotion        || "upright";
    const confidence    = result.confidence     || 0;
    const learningState = (result.learning_state || "CALM").toUpperCase();
    const source        = result.source         || "none";

    updatePostureHistory(learningState, confidence);
    const stableState = getStablePosture();

    updateStateDisplay(stableState, rawPosture, confidence, source);
    applyPostureAdaptation(stableState, confidence);
    updateEngagementBar(stableState, confidence);
    showPostureAlert(stableState);

    console.log(`[EduMotion]   Stable state: ${stableState} | raw: ${rawPosture} | conf: ${Math.round(confidence * 100)}% | src: ${source}`);
}

function updateStateDisplay(state, rawPosture, confidence, source) {
    const stateEl         = document.getElementById("current-state");
    const confidenceEl    = document.getElementById("confidence-value");
    const rawLabelEl      = document.getElementById("raw-posture-label");
    const modeEl          = document.getElementById("mode-indicator");
    const postureTextEl   = document.getElementById("posture-text");
    const postureBadgeEl  = document.getElementById("posture-badge");
    const postureIconEl   = document.getElementById("posture-icon");

    const icon = POSTURE_ICONS[state] || "😐";
    const rawClean = (rawPosture || "upright").toLowerCase().trim();
    const formattedPosture = POSTURE_RAW_LABELS[rawClean] || rawClean.replace(/_/g, " ");
    const postureIcon = POSTURE_ICONS_BY_NAME[rawClean] || POSTURE_ICONS[state] || "🧍";

    if (stateEl)      stateEl.textContent = `${icon} ${state}`;
    if (confidenceEl) {
        confidenceEl.textContent = `${Math.round(confidence * 100)}%`;
        confidenceEl.style.color = confidence < 0.40 ? "#ef4444" : "#22c55e";
    }
    if (rawLabelEl)    rawLabelEl.textContent = formattedPosture;
    if (postureTextEl) postureTextEl.textContent = formattedPosture;
    if (postureBadgeEl) {
        postureBadgeEl.textContent = rawClean.toUpperCase().replace(/_/g, " ");
        postureBadgeEl.className = `posture-badge badge-${rawClean.replace(/_/g, "-")}`;
    }
    if (postureIconEl) postureIconEl.textContent = postureIcon;

    if (modeEl) {
        modeEl.textContent = source === "cloud" ? "CLOUD AI" : source === "local" ? "LOCAL CNN" : source.toUpperCase();
        modeEl.className   = "mode-indicator auto";
    }
}

function applyPostureAdaptation(state, confidence) {
    const panel = document.getElementById("emotion-panel");
    if (!panel) return;
    panel.classList.remove("state-engaged","state-bored","state-confused","state-frustrated");
    if (state && confidence >= 0.30) {
        panel.classList.add(`state-${state.toLowerCase()}`);
    }
    // Apply full-page emotion background (only when not manually overridden)
    if (!manualEmotionOverride) {
        const autoEmotion = LEARNING_STATE_TO_EMOTION[state] || null;
        applyEmotionTheme(autoEmotion);
    }
}

function updateEngagementBar(state, confidence) {
    const bar = document.getElementById("engagement-fill");
    const pct = document.getElementById("engagement-pct");

    const qualityMap = { ENGAGED:90, BORED:35, CONFUSED:50, FRUSTRATED:20 };
    const colorMap   = { ENGAGED:"#22c55e", BORED:"#fb923c", CONFUSED:"#f59e0b", FRUSTRATED:"#ef4444" };

    const quality = Math.round((qualityMap[state] || 60) * (confidence > 0.30 ? 1 : 0.6));
    if (bar) { bar.style.width = `${quality}%`; bar.style.backgroundColor = colorMap[state] || "#60a5fa"; }
    if (pct) pct.textContent = `${quality}%`;
}

function showPostureAlert(state) {
    const alertEl = document.getElementById("posture-alert");
    if (!alertEl) return;
    const cfg = POSTURE_ALERTS[state] || POSTURE_ALERTS.CALM;
    alertEl.textContent  = cfg.msg;
    alertEl.className    = "";
    alertEl.classList.add(cfg.cls);
    alertEl.style.display = "block";
}

function resetPostureDisplay() {
    const stateEl         = document.getElementById("current-state");
    const confidenceEl    = document.getElementById("confidence-value");
    const rawLabelEl      = document.getElementById("raw-posture-label");
    const alertEl         = document.getElementById("posture-alert");
    const bar             = document.getElementById("engagement-fill");
    const pct             = document.getElementById("engagement-pct");
    const panel           = document.getElementById("emotion-panel");
    const postureTextEl   = document.getElementById("posture-text");
    const postureBadgeEl  = document.getElementById("posture-badge");
    const postureIconEl   = document.getElementById("posture-icon");

    if (stateEl)         stateEl.textContent      = "—";
    if (confidenceEl)    confidenceEl.textContent  = "—";
    if (rawLabelEl)      rawLabelEl.textContent    = "";
    if (postureTextEl)   postureTextEl.textContent = "—";
    if (postureBadgeEl) {
        postureBadgeEl.textContent = "INACTIVE";
        postureBadgeEl.className = "posture-badge";
    }
    if (postureIconEl)   postureIconEl.textContent = "🧍";
    if (alertEl)         alertEl.style.display     = "none";
    if (bar)             bar.style.width           = "72%";
    if (pct)             pct.textContent           = "72%";
    if (panel)           panel.classList.remove("state-engaged","state-bored","state-confused","state-frustrated");

    // Reset emotion theme to none when camera stopped (unless manual override active)
    if (!manualEmotionOverride) {
        document.body.setAttribute("data-emotion", "none");
        updateEmotionButtonIndicator("none", false);
    }

    LearningState.postureHistory = [];
}

// ─────────────────────────────────────────────────────────────
// Temporal smoothing
// ─────────────────────────────────────────────────────────────

function updatePostureHistory(state, confidence) {
    LearningState.postureHistory.push({ state, confidence, ts: Date.now() });
    if (LearningState.postureHistory.length > 5)
        LearningState.postureHistory = LearningState.postureHistory.slice(-5);
}

function getStablePosture() {
    const history = LearningState.postureHistory;
    if (!history.length) return "ENGAGED";
    const filtered = history.filter((p) => p.confidence >= 0.30);
    const src      = filtered.length ? filtered : history;
    const counts   = {};
    src.forEach((p) => { counts[p.state] = (counts[p.state] || 0) + 1; });
    return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
}

// ─────────────────────────────────────────────────────────────
// Event listeners — manual mode, lesson buttons
// ─────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────
// Navbar Emotion Dropdown
// ─────────────────────────────────────────────────────────────

function setupEmotionDropdown() {
    const toggleBtn  = document.getElementById("emotion-toggle-btn");
    const dropdown   = document.getElementById("emotion-dropdown");
    const resetBtn   = document.getElementById("btn-reset-emotion");
    const optionBtns = document.querySelectorAll(".emotion-option[data-emotion]");

    if (!toggleBtn || !dropdown) return;

    // Toggle dropdown open/close
    toggleBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const isOpen = !dropdown.hidden;
        dropdown.hidden = isOpen;
        toggleBtn.setAttribute("aria-expanded", String(!isOpen));
    });

    // Close when clicking outside
    document.addEventListener("click", (e) => {
        if (!dropdown.hidden && !dropdown.contains(e.target) && !toggleBtn.contains(e.target)) {
            dropdown.hidden = true;
            toggleBtn.setAttribute("aria-expanded", "false");
        }
    });

    // Emotion option click
    optionBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            const rawVal = btn.dataset.emotion;  // engaged | bored | confused | frustrated
            const state = rawVal.toUpperCase();

            // Set manual override
            manualEmotionOverride = rawVal;
            applyEmotionTheme(rawVal);

            // Apply panel border theme
            const panel = document.getElementById("emotion-panel");
            if (panel) {
                panel.classList.remove("state-engaged","state-bored","state-confused","state-frustrated");
                panel.classList.add(`state-${state.toLowerCase()}`);
            }

            // Mark active option
            optionBtns.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");

            // Close dropdown
            dropdown.hidden = true;
            toggleBtn.setAttribute("aria-expanded", "false");

            // Update UI elements to match manual choice
            showPostureAlert(state);
            updateEngagementBar(state, 1.0);

            const stateEl = document.getElementById("current-state");
            if (stateEl) stateEl.textContent = `${POSTURE_ICONS[state] || "😐"} ${rawVal}`;
            const confEl = document.getElementById("confidence-value");
            if (confEl) { confEl.textContent = "Manual"; confEl.style.color = "#818cf8"; }
            const modeEl = document.getElementById("mode-indicator");
            if (modeEl) { modeEl.textContent = "MANUAL"; modeEl.className = "mode-indicator manual"; }

            // Notify backend of manual mode
            fetch("/api/emotion/mode", {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify({ mode: "manual", state: state }),
            }).catch((err) => console.warn("[EduMotion] Backend mode notify failed:", err));

            console.log(`[EduMotion] Emotion manually set to: ${rawVal}`);
        });
    });

    // Reset to auto detection
    if (resetBtn) {
        resetBtn.addEventListener("click", () => {
            manualEmotionOverride = null;
            optionBtns.forEach((b) => b.classList.remove("active"));
            document.body.setAttribute("data-emotion", "none");
            updateEmotionButtonIndicator("none", false);
            dropdown.hidden = true;
            toggleBtn.setAttribute("aria-expanded", "false");
            console.log("[EduMotion] Emotion override cleared — back to auto detection.");

            // Notify backend
            fetch("/api/emotion/mode", {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify({ mode: "auto" }),
            }).catch((err) => console.warn("[EduMotion] Backend mode notify failed:", err));
        });
    }
}

function setupEventListeners() {
    document.querySelector(".btn-lesson.more-videos")?.addEventListener("click", () => {
        alert("More videos — shows related content.");
    });
    document.querySelector(".btn-lesson.next-lesson")?.addEventListener("click", () => {
        alert("Navigating to next lesson…");
    });
}

// ─────────────────────────────────────────────────────────────
// YouTube Video Loader
// ─────────────────────────────────────────────────────────────

function setupYouTubePlayer() {
    const input = document.getElementById("youtube-url-input");
    const loadBtn = document.getElementById("btn-load-video");
    const iframe = document.getElementById("youtube-iframe");
    const titleDisplay = document.getElementById("course-title-display");

    if (!input || !loadBtn || !iframe) return;

    function extractYouTubeId(urlOrId) {
        if (!urlOrId) return null;
        const trimmed = urlOrId.trim();
        // Direct 11 char ID
        if (/^[a-zA-Z0-9_-]{11}$/.test(trimmed)) {
            return trimmed;
        }
        // Match various youtube URL formats: watch?v=, youtu.be/, embed/, live/, shorts/
        const match = trimmed.match(/(?:youtu\.be\/|youtube\.com\/(?:embed\/|v\/|watch\?v=|watch\?.+&v=|live\/|shorts\/))([\w-]{11})/i);
        return match ? match[1] : null;
    }

    function loadVideoFromInput() {
        const url = input.value.trim();
        const videoId = extractYouTubeId(url);
        if (!videoId) {
            alert("Please enter a valid YouTube Video URL or Video ID (e.g., https://www.youtube.com/watch?v=HcqpanDadyQ)");
            return;
        }
        iframe.src = `https://www.youtube.com/embed/${videoId}?autoplay=1&rel=0&showinfo=0&enablejsapi=1`;
        if (titleDisplay) {
            titleDisplay.textContent = `Lesson Video (ID: ${videoId})`;
        }
        console.log(`[EduMotion] Loaded YouTube Video: ${videoId}`);
    }

    loadBtn.addEventListener("click", loadVideoFromInput);
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            loadVideoFromInput();
        }
    });
}

// ─────────────────────────────────────────────────────────────
// Focus Mode Toggle
// ─────────────────────────────────────────────────────────────

function setupFocusMode() {
    const focusBtn = document.getElementById("btn-focus-mode");
    if (!focusBtn) return;

    focusBtn.addEventListener("click", () => {
        document.body.classList.toggle("focus-mode");
        const isFocus = document.body.classList.contains("focus-mode");
        focusBtn.innerHTML = isFocus ? "⬛ Exit Focus" : "🌑 Focus Mode";
        focusBtn.classList.toggle("active", isFocus);
        focusBtn.setAttribute("aria-pressed", String(isFocus));
        console.log(`[EduMotion] Focus mode: ${isFocus ? "ON — pitch black background" : "OFF"}`);
    });
}

// ─────────────────────────────────────────────────────────────
// Utility helpers
// ─────────────────────────────────────────────────────────────

function setElDisplay(id, visible) {
    const el = document.getElementById(id);
    if (el) el.style.display = visible ? "inline-block" : "none";
}

/**
 * Show a small status line below the camera controls.
 * Uses #camera-status if present, otherwise uses posture-alert.
 */
function setStatus(msg) {
    const el = document.getElementById("camera-status");
    if (el) { el.textContent = msg; el.style.display = msg ? "block" : "none"; }
}