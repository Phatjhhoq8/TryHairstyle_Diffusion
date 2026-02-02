from celery import Celery
import os
import shutil
from pathlib import Path
from .pipeline import HairSwapPipeline

# Initialize Celery
# Redis URL should be configurable via env vars, defaulting to localhost for dev
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("hair_transfer", broker=REDIS_URL, backend=REDIS_URL)

# Global pipeline instance (loaded once per worker process)
pipeline = None

@celery_app.task(bind=True, name="generate_hairstyle")
def generate_hairstyle_task(self, target_path: str, reference_path: str, output_path: str, description: str = ""):
    """
    Celery task to run the hairstyle transfer pipeline.
    """
    global pipeline
    
    # Lazy loading of the pipeline
    if pipeline is None:
        print("Worker: Loading Pipeline...")
        pipeline = HairSwapPipeline()
        pipeline.load_models()

    try:
        self.update_state(state='PROCESSING', meta={'status': 'Running Inference...'})
        print(f"Worker: Processing {target_path} -> {output_path}")
        
        result_path = pipeline.run_inference(target_path, reference_path, output_path)
        
        return {"status": "completed", "result_url": f"/static/results/{Path(result_path).name}"}
    except Exception as e:
        print(f"Worker Error: {e}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise e
