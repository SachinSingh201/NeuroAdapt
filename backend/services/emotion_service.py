import time
from .cloud_emotion_service import CloudEmotionService
from .local_emotion_service import LocalEmotionService
from .image_service import ImageService
from ..config import Config

class EmotionService:
    def __init__(self):
        self.cloud_service = CloudEmotionService()
        self.local_service = LocalEmotionService()
        self.image_service = ImageService()

    async def predict(self, base64_image: str, mode: str = None):
        """
        Orchestrates emotion prediction based on the selected mode (cloud, local, auto).
        """
        mode = mode or Config.EMOTION_MODE
        
        # 1. Decode image for local processing if needed
        # (Decoding is fast, so we do it if local might be needed)
        cv_image = self.image_service.decode_base64_image(base64_image)
        
        result = {
            "emotion": "neutral",
            "confidence": 0.0,
            "source": "none",
            "processing_time_ms": 0,
            "error": None
        }

        start_time = time.time()

        # CLOUD MODE
        if mode == "cloud":
            await self._run_cloud(base64_image, result)
        
        # LOCAL MODE
        elif mode == "local":
            self._run_local(cv_image, result)
        
        # AUTO MODE (Fallback logic)
        elif mode == "auto":
            # Try cloud first
            await self._run_cloud(base64_image, result)
            
            # If cloud failed or returned error, fallback to local
            if result["error"] or result["source"] == "none":
                print(f"Cloud API failed: {result['error']}. Falling back to local model.")
                result["error"] = None # Reset error for local attempt
                self._run_local(cv_image, result)
        
        # Add total processing time
        if result["processing_time_ms"] == 0:
            result["processing_time_ms"] = int((time.time() - start_time) * 1000)

        # --- Safety: if all attempts failed, return a neutral fallback ---
        # This prevents the caller from raising HTTP 500 on cloud/network errors.
        if result.get("error"):
            print(f"[EduMotion] All prediction attempts failed: {result['error']} -> returning neutral fallback.", flush=True)
            result["emotion"]    = "upright"   # neutral posture
            result["confidence"] = 0.0
            result["source"]     = "none"
            result["error"]      = None

        return result

    async def _run_cloud(self, base64_image, result):
        cloud_data, cloud_time = await self.cloud_service.predict(base64_image)
        
        if cloud_data and "error" not in cloud_data:
            emotion, confidence = self.cloud_service.normalize_response(cloud_data)
            if emotion:
                result["emotion"] = emotion
                result["confidence"] = confidence
                result["source"] = "cloud"
                result["processing_time_ms"] = cloud_time
                return
        
        result["error"] = cloud_data.get("error") if cloud_data else "Cloud API returned no data"

    def _run_local(self, cv_image, result):
        if cv_image is None:
            result["error"] = "Invalid image for local processing"
            return

        start_local = time.time()
        emotion, confidence = self.local_service.predict(cv_image)
        local_time = int((time.time() - start_local) * 1000)

        if emotion:
            result["emotion"] = emotion
            result["confidence"] = confidence
            result["source"] = "local"
            result["processing_time_ms"] = local_time
        else:
            result["error"] = "Local model failed to predict"
