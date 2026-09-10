# from pydantic import BaseModel, Field
# from typing import List, Optional, Dict, Any

# class EmotionPredictionRequest(BaseModel):
#     image: str = Field(..., description="Base64 encoded JPEG/PNG image")
#     mode: Optional[str] = Field(None, description="Inference mode: cloud, local, auto")

# class EmotionPredictionResponse(BaseModel):
#     emotion: str
#     learning_state: str
#     confidence: float
#     source: str
#     processing_time_ms: int
#     adaptation: Dict[str, Any]
#     error: Optional[str] = None

# class SystemStatus(BaseModel):
#     status: str
#     version: str
#     emotion_mode: str
#     cloud_api_configured: bool
#     local_model_loaded: bool

# class LessonContent(BaseModel):
#     id: str
#     course_name: str
#     lesson_title: str
#     description: str
#     video_url: str
#     content_body: str
#     progress: int


from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class EmotionPredictionRequest(BaseModel):
    image: str = Field(
        ...,
        description="Base64 encoded JPEG/PNG image"
    )

    mode: Optional[str] = Field(
        None,
        description="Inference mode: cloud, local, auto"
    )


class EmotionPredictionResponse(BaseModel):
    emotion: str
    learning_state: str
    confidence: float
    source: str
    processing_time_ms: int
    adaptation: Dict[str, Any]
    error: Optional[str] = None


class SystemStatus(BaseModel):
    status: str
    version: str
    emotion_mode: str
    cloud_api_configured: bool
    local_model_loaded: bool


class LessonContent(BaseModel):
    id: str
    course_name: str
    lesson_title: str
    description: str
    video_url: str
    content_body: str
    progress: int


# --------------------------------------------------
# Health Check
# --------------------------------------------------

def HealthCheck(
    version: str,
    emotion_mode: str,
    cloud_api_configured: bool,
    local_model_loaded: bool
) -> SystemStatus:
    """
    Returns the current health/status of the EduMotion system.
    """

    # Determine overall system status
    if local_model_loaded or cloud_api_configured:
        status = "healthy"
    else:
        status = "degraded"

    return SystemStatus(
        status=status,
        version=version,
        emotion_mode=emotion_mode,
        cloud_api_configured=cloud_api_configured,
        local_model_loaded=local_model_loaded
    )