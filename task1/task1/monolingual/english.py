from task1.pipelines.IngestionPipeline import IngestionPipeline
from task1.config import ProjectPaths
from sentence_transformers import SentenceTransformer
import torch

if __name__ == "__main__":
    paths = ProjectPaths()
    
    # Check if MPS is available and set device accordingly
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize the model with explicit device handling
    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device=device,
        cache_folder="./cache"  # Add cache folder to avoid permission issues
    )
    
    ingestion_pipeline = IngestionPipeline(
        data_path=paths.english_data_dir,
        language="en",
        embedding_model=model
    )
    ingestion_pipeline.run()
    