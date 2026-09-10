import time
import asyncio
from .cloud_emotion_service import CloudEmotionService
from .local_emotion_service import LocalEmotionService
from .image_service import ImageService
from .adaptation_service import AdaptationService
from ..config import Config

class EmotionService:
    def __init__(self):
        self.cloud_service = CloudEmotionService()
        self.local_service = LocalEmotionService()
        self.image_service = ImageService()

    async def predict(self, base64_image: str, mode: str = None):
        """
        Runs backend emotion (local CNN) + posture (cloud API) detection.
        Returns combined and normalized results.
        """
        cv_image = self.image_service.decode_base64_image(base64_image)

        result = {
            "emotion": "engaged",
            "confidence": 0.0,
            "posture": "upright",
            "posture_confidence": 0.0,
            "source": "combined",
            "processing_time_ms": 0,
            "error": None
        }

        start_time = time.time()

        # Run Local Emotion CNN synchronously via asyncio.to_thread
        def run_emotion():
            return self.local_service.predict(cv_image)

        emotion_task = asyncio.to_thread(run_emotion)
        posture_task = self.cloud_service.predict(base64_image)

        # Await both concurrently
        local_res, cloud_res = await asyncio.gather(emotion_task, posture_task, return_exceptions=True)

        # Parse local emotion
        if isinstance(local_res, Exception):
            print(f"Local emotion failed: {local_res}")
            result["error"] = str(local_res)
        else:
            emotion, conf = local_res
            if emotion:
                result["emotion"] = AdaptationService.normalize_emotion(emotion)
                result["confidence"] = float(conf)

        # Parse cloud posture
        if isinstance(cloud_res, Exception):
            print(f"Cloud posture failed: {cloud_res}")
        else:
            cloud_data, _ = cloud_res
            if cloud_data and "error" not in cloud_data:
                posture, p_conf = self.cloud_service.normalize_response(cloud_data)
                if posture:
                    result["posture"] = AdaptationService.normalize_posture(posture)
                    result["posture_confidence"] = float(p_conf)
            else:
                err = cloud_data.get("error") if cloud_data else "Unknown posture error"
                print(f"Cloud posture note: {err}")

        result["processing_time_ms"] = int((time.time() - start_time) * 1000)
        return result
