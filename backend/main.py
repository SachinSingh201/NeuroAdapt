import time
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import Config

from .services.emotion_service import EmotionService
from .services.adaptation_service import AdaptationService

from .models.schemas import (
    EmotionPredictionRequest,
    EmotionPredictionResponse,
    SystemStatus,
    LessonContent,
    HealthCheck,
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# ============================================================
# SERVICES
# ============================================================

emotion_service = EmotionService()


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title=Config.PROJECT_NAME,
    version=Config.VERSION,
    description="EduMotion - Emotion-Aware Dynamic UI/UX System",
)


# ============================================================
# FRONTEND PATHS
# ============================================================

frontend_dir = Path(__file__).parent.parent / "frontend"

templates_dir = frontend_dir / "templates"


# ============================================================
# STATIC FILES
# ============================================================

if frontend_dir.exists():

    app.mount(
        "/static",
        StaticFiles(directory=str(frontend_dir)),
        name="static",
    )


# ============================================================
# TEMPLATES
# ============================================================

templates = Jinja2Templates(
    directory=str(templates_dir)
)


# ============================================================
# ROOT PAGE
# ============================================================

@app.get("/", include_in_schema=False)
async def root(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request
        }
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/api/health",
    response_model=SystemStatus
)
async def health_check():

    local_loaded = emotion_service.local_service.model is not None

    return HealthCheck(
        version=Config.VERSION,
        emotion_mode=Config.EMOTION_MODE,
        cloud_api_configured=bool(Config.CLOUD_API_URL),
        local_model_loaded=local_loaded,
    )


# ============================================================
# EMOTION STATUS
# ============================================================

@app.get(
    "/api/emotion/status",
    response_model=SystemStatus
)
async def emotion_status():

    local_loaded = emotion_service.local_service.model is not None

    return SystemStatus(
        status="operational",
        version=Config.VERSION,
        emotion_mode=Config.EMOTION_MODE,
        cloud_api_configured=bool(Config.CLOUD_API_URL),
        local_model_loaded=local_loaded,
    )


# ============================================================
# EMOTION PREDICTION
# ============================================================

@app.post(
    "/api/emotion/predict",
    response_model=EmotionPredictionResponse
)
async def predict_emotion(
    request: EmotionPredictionRequest
):

    try:

        start_time = time.time()

        # ----------------------------------------------------
        # Decode image
        # ----------------------------------------------------

        cv_image = emotion_service.image_service.decode_base64_image(
            request.image
        )

        if cv_image is None:

            raise HTTPException(
                status_code=400,
                detail="Invalid image data"
            )

        # ----------------------------------------------------
        # Get mode
        # ----------------------------------------------------

        mode = (
            request.mode
            or Config.EMOTION_MODE
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        result = await emotion_service.predict(
            request.image,
            mode
        )

        # ----------------------------------------------------
        # Processing time
        # ----------------------------------------------------

        processing_time_ms = int(
            (time.time() - start_time) * 1000
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        raw_emotion = result.get(
            "emotion",
            "neutral"
        )

        source = result.get(
            "source",
            "none"
        )

        confidence = result.get(
            "confidence",
            0.0
        )

        # ----------------------------------------------------
        # Learning state
        # ----------------------------------------------------

        learning_state = (
            AdaptationService.map_to_learning_state(
                raw_emotion,
                source
            )
        )

        # ----------------------------------------------------
        # UI adaptation
        # ----------------------------------------------------

        adaptation = (
            AdaptationService.get_adaptation_ui_changes(
                learning_state,
                confidence
            )
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        response = EmotionPredictionResponse(
            emotion=raw_emotion,
            learning_state=learning_state,
            confidence=confidence,
            source=source,
            processing_time_ms=processing_time_ms,
            adaptation=adaptation,
        )

        logger.info(
            "Emotion prediction: %s -> %s "
            "(confidence=%.2f, source=%s, "
            "processing_time=%sms)",
            raw_emotion,
            learning_state,
            confidence,
            source,
            processing_time_ms,
        )

        return response

    except HTTPException:

        raise

    except Exception as e:

        logger.exception(
            "Error in emotion prediction"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# BASE64 EMOTION PREDICTION
# ============================================================

@app.post(
    "/api/emotion/predict-base64",
    response_model=EmotionPredictionResponse
)
async def predict_emotion_base64(
    request: EmotionPredictionRequest
):

    try:

        start_time = time.time()

        # ----------------------------------------------------
        # Decode image
        # ----------------------------------------------------

        cv_image = emotion_service.image_service.decode_base64_image(
            request.image
        )

        if cv_image is None:

            raise HTTPException(
                status_code=400,
                detail="Invalid image data"
            )

        # ----------------------------------------------------
        # Prediction mode
        # ----------------------------------------------------

        mode = (
            request.mode
            or Config.EMOTION_MODE
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        result = await emotion_service.predict(
            request.image,
            mode
        )

        processing_time_ms = int(
            (time.time() - start_time) * 1000
        )

        # ----------------------------------------------------
        # Extract result
        # ----------------------------------------------------

        raw_emotion = result.get(
            "emotion",
            "neutral"
        )

        source = result.get(
            "source",
            "none"
        )

        confidence = result.get(
            "confidence",
            0.0
        )

        # ----------------------------------------------------
        # Learning state
        # ----------------------------------------------------

        learning_state = (
            AdaptationService.map_to_learning_state(
                raw_emotion,
                source
            )
        )

        # ----------------------------------------------------
        # Adaptation
        # ----------------------------------------------------

        adaptation = (
            AdaptationService.get_adaptation_ui_changes(
                learning_state,
                confidence
            )
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        response = EmotionPredictionResponse(
            emotion=raw_emotion,
            learning_state=learning_state,
            confidence=confidence,
            source=source,
            processing_time_ms=processing_time_ms,
            adaptation=adaptation,
        )

        # -- Terminal output (ASCII-safe for Windows CP1252 console) --
        conf_pct = int(confidence * 100)
        conf_bar = "#" * (conf_pct // 10) + "." * (10 - conf_pct // 10)
        print(
            f"\n[EduMotion] Posture Detection Result"
            f"\n  Raw Posture   : {raw_emotion}"
            f"\n  Learning State: {learning_state}"
            f"\n  Confidence    : {conf_pct}%  [{conf_bar}]"
            f"\n  Source        : {source}"
            f"\n  Processing    : {processing_time_ms}ms"
            f"\n  -> Sent to frontend OK",
            flush=True
        )

        logger.info(
            "Emotion prediction (base64): %s -> %s (conf=%.2f, src=%s, %dms)",
            raw_emotion, learning_state, confidence, source, processing_time_ms
        )

        return response

    except HTTPException:

        raise

    except Exception as e:

        print(f"[EduMotion] ERROR in base64 prediction: {e}", flush=True)
        logger.exception(
            "Error in base64 emotion prediction"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# CHANGE EMOTION MODE
# ============================================================

from pydantic import BaseModel
from typing import Optional as Opt

class ModeRequest(BaseModel):
    mode: str
    state: Opt[str] = None

@app.post("/api/emotion/mode")
async def set_emotion_mode(
    body: ModeRequest
):

    valid_modes = [
        "cloud",
        "local",
        "auto",
        "manual",
    ]

    if body.mode not in valid_modes:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid mode. "
                f"Must be one of: {valid_modes}"
            )
        )

    if body.mode != "manual":
        Config.EMOTION_MODE = body.mode

    # Load local model if required
    if body.mode == "local":
        emotion_service.local_service.load_model()

    logger.info(
        "Emotion mode changed to: %s (state=%s)",
        body.mode,
        body.state,
    )

    return {
        "mode": body.mode,
        "state": body.state,
        "message": (
            f"Emotion inference mode set to {body.mode}"
        ),
    }


# ============================================================
# LESSON
# ============================================================

@app.get(
    "/api/lesson",
    response_model=LessonContent
)
async def get_lesson():

    return LessonContent(

        id="lesson_1",

        course_name=Config.COURSE_NAME,

        lesson_title=Config.LESSON_TITLE,

        description=Config.LESSON_DESCRIPTION,

        video_url=Config.YOUTUBE_VIDEO_URL,

        content_body=Config.LESSON_DESCRIPTION,

        progress=0,
    )


# ============================================================
# ADAPTATION EVALUATION
# ============================================================

@app.post("/api/adaptation/evaluate")
async def evaluate_adaptation(
    state_data: dict
):

    try:

        emotion = state_data.get(
            "emotion",
            "neutral"
        )

        confidence = state_data.get(
            "confidence",
            0.0
        )

        source = state_data.get(
            "source",
            "cloud"
        )

        # ----------------------------------------------------
        # Learning state
        # ----------------------------------------------------

        learning_state = (
            AdaptationService.map_to_learning_state(
                emotion,
                source
            )
        )

        # ----------------------------------------------------
        # UI adaptation
        # ----------------------------------------------------

        adaptation = (
            AdaptationService.get_adaptation_ui_changes(
                learning_state,
                confidence
            )
        )

        # ----------------------------------------------------
        # Combination handling
        # ----------------------------------------------------

        current_states = [
            learning_state
        ]

        if confidence >= Config.MIN_CONFIDENCE:

            combo_result = (
                AdaptationService.handle_combinations(
                    current_states
                )
            )

        else:

            combo_result = learning_state

        return {

            "raw_emotion": emotion,

            "learning_state": learning_state,

            "confidence": confidence,

            "source": source,

            "adaptation": adaptation,

            "combination_mode": combo_result,
        }

    except Exception as e:

        logger.exception(
            "Error in adaptation evaluation"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# LEARNING PAGE
# ============================================================

@app.get(
    "/learning",
    include_in_schema=False
)
async def learning_page(
    request: Request
):

    return templates.TemplateResponse(
        "learning.html",
        {
            "request": request
        }
    )


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
