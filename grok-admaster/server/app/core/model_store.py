"""
Model Store
Abstracts where ML models are saved and loaded.
"""
import abc
import os
import pickle
import logging
from typing import Any
from datetime import datetime
logger = logging.getLogger(__name__)

class ModelStore(abc.ABC):
    @abc.abstractmethod
    def save(self, model: Any, name: str) -> bool:
        pass
        
    @abc.abstractmethod
    def load(self, name: str) -> Any:
        pass

class FileSystemModelStore(ModelStore):
    def __init__(self, base_path: str = "models"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        # Prefer joblib for sklearn efficiency, fallback to pickle
        try:
            import joblib
            self.backend = joblib
            self.ext = "joblib"
            logger.info("FileSystemModelStore using joblib backend")
        except ImportError:
            self.backend = pickle
            self.ext = "pkl"
            logger.info("FileSystemModelStore using pickle backend (install joblib for better performance)")

    def save(self, model: Any, name: str) -> bool:
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.{self.ext}"
            path = os.path.join(self.base_path, filename)
            
            # joblib.dump and pickle.dump have similar signatures for simple use cases
            with open(path, 'wb') as f:
                self.backend.dump(model, f)
            logger.info(f"Model saved to {path}")
            
            # Update 'latest' pointer (symlink or copy)
            # On Windows/Simple systems, we can just copy to a fixed 'latest' name for easy loading
            latest_path = os.path.join(self.base_path, f"{name}_latest.{self.ext}")
            try:
                with open(latest_path, 'wb') as f:
                    self.backend.dump(model, f)
            except Exception as e:
                logger.warning(f"Could not update 'latest' pointer: {e}")
                
            return True
        except Exception as e:
            logger.error(f"Failed to save model {name}: {e}")
            return False
            
    def load(self, name: str) -> Any:
        try:
            # 1. Try Specific Name (Exact Match)
            path = os.path.join(self.base_path, f"{name}.{self.ext}")
            
            # 2. Try 'Latest' Pointer for this name
            if not os.path.exists(path):
                path = os.path.join(self.base_path, f"{name}_latest.{self.ext}")
                
            # 3. Fallback: Search for newest timestamped file
            if not os.path.exists(path):
                # pattern: name_YYYYMMDD_HHMMSS.ext
                files = os.listdir(self.base_path)
                candidates = [f for f in files if f.startswith(name) and f.endswith(self.ext)]
                if candidates:
                    # simplistic sort by name works for YYYYMMDD format
                    candidates.sort(reverse=True)
                    path = os.path.join(self.base_path, candidates[0])
            
            if not os.path.exists(path):
                # Fallback check for alternate extension
                alt_ext = "pkl" if self.ext == "joblib" else "joblib"
                # ... (Simplified logical fallback for alt ext not strictly needed if we enforce one, but keeping structure)
                return None
            
            logger.info(f"Loading model from {path}")
            with open(path, 'rb') as f:
                return self.backend.load(f)
        except Exception as e:
            logger.error(f"Failed to load model {name}: {e}")
            return None

class S3ModelStore(ModelStore):
    """
    Cloud model store backed by AWS S3.

    Requires:
        - boto3 installed (`pip install boto3`)
        - AWS credentials configured (env vars, ~/.aws/credentials, or IAM role)
    """
    def __init__(self, bucket_name: str, prefix: str = "models/"):
        self.bucket = bucket_name
        self.prefix = prefix
        # Use joblib if available, fallback to pickle
        try:
            import joblib
            self.backend = joblib
            self.ext = "joblib"
        except ImportError:
            self.backend = pickle
            self.ext = "pkl"

    def _get_s3_client(self):
        import boto3
        return boto3.client("s3")

    def save(self, model: Any, name: str) -> bool:
        """Serialize model locally, then upload to S3."""
        import tempfile
        try:
            s3 = self._get_s3_client()
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            s3_key = f"{self.prefix}{name}_{timestamp}.{self.ext}"
            latest_key = f"{self.prefix}{name}_latest.{self.ext}"

            with tempfile.NamedTemporaryFile(suffix=f".{self.ext}", delete=False) as tmp:
                tmp_path = tmp.name
                self.backend.dump(model, tmp)

            # Upload timestamped version
            s3.upload_file(tmp_path, self.bucket, s3_key)
            # Upload/overwrite 'latest' pointer
            s3.upload_file(tmp_path, self.bucket, latest_key)

            os.remove(tmp_path)
            logger.info(f"Model '{name}' uploaded to s3://{self.bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"S3 upload failed for '{name}': {e}")
            return False

    def load(self, name: str) -> Any:
        """Download model from S3 and deserialize."""
        import tempfile
        try:
            s3 = self._get_s3_client()
            s3_key = f"{self.prefix}{name}_latest.{self.ext}"

            with tempfile.NamedTemporaryFile(suffix=f".{self.ext}", delete=False) as tmp:
                tmp_path = tmp.name

            s3.download_file(self.bucket, s3_key, tmp_path)
            logger.info(f"Model '{name}' downloaded from s3://{self.bucket}/{s3_key}")

            with open(tmp_path, "rb") as f:
                model = self.backend.load(f)

            os.remove(tmp_path)
            return model
        except Exception as e:
            logger.error(f"S3 download failed for '{name}': {e}")
            return None

# Factory/Singleton access
_current_store = FileSystemModelStore()

def get_model_store() -> ModelStore:
    return _current_store
