import time
import asyncio
from .cloud_emotion_service import CloudEmotionService
from .local_emotion_service import LocalEmotionService
from .deepface_service import DeepFaceEmotionService
from .image_service import ImageService
from .adaptation_service import AdaptationService
from ..config import Config

class EmotionService:
    def __init__(self):
        self.cloud_service = CloudEmotionService()
        self.local_service = LocalEmotionService()
        self.deepface_service = DeepFaceEmotionService()
        self.image_service = ImageService()

    async def predict(self, base64_image: str, mode: str = None):
        """
        Runs Python DeepFace as primary emotion detector + Cloud posture detector concurrently.
        Computes 70% DeepFace + 30% Learning Statistics combined scores.
        """
        cv_image = self.image_service.decode_base64_image(base64_image)

        start_time = time.time()

        # 1. Posture task (Cloud API)
        posture_task = self.cloud_service.predict(base64_image)

        # 2. Concurrently get cloud posture while decoding
        cloud_res = await posture_task

        # Parse cloud posture
        posture = "upright"
        posture_confidence = 0.80

        if cloud_res:
            cloud_data, _ = cloud_res
            if cloud_data and "error" not in cloud_data:
                p_label, p_conf = self.cloud_service.normalize_response(cloud_data)
                if p_label:
                    posture = AdaptationService.normalize_posture(p_label)
                    posture_confidence = float(p_conf)

        # 3. Run DeepFace Analysis + Combined Scoring (in worker thread to not block event loop)
        def run_deepface_evaluation():
            return self.deepface_service.evaluate_combined(
                cv_image=cv_image,
                posture=posture,
                posture_confidence=posture_confidence,
            )

        eval_result = await asyncio.to_thread(run_deepface_evaluation)

        eval_result["processing_time_ms"] = int((time.time() - start_time) * 1000)
        eval_result["source"] = "deepface+stats"
        eval_result["error"] = None

        return eval_result
