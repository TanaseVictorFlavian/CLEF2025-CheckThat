from task1 import pipelines as ppl
from task1.config import ProjectPaths
from sentence_transformers import SentenceTransformer
import torch
from task1.models.MLP import MLP

if __name__ == "__main__":
    paths = ProjectPaths()
    
    # Check if MPS is available and set device accordingly
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # # Initialize the model with explicit device handling
    # embedding_model = SentenceTransformer(
    #     "all-MiniLM-L6-v2",
    #     device=device,
    #     cache_folder="./cache"  # Add cache folder to avoid permission issues
    # )
    
    # ingestion_pipeline = ppl.IngestionPipeline(
    #     data_path=paths.english_data_dir,
    #     language="en",
    #     embedding_model=embedding_model
    # )
    # ingestion_pipeline.run()
    train_pipeline = ppl.TrainPipelineNN(
        language="en",
        model=MLP(in_features=384, out_features=1),
        data_path=paths.english_data_dir,
        batch_size=128
    )
    train_pipeline.run()
    
    evaluation_pipeline = ppl.EvaluationPipelineNN(
        model=train_pipeline.model,
        data_path=paths.english_data_dir,
        data=train_pipeline.data
    )
    evaluation_pipeline.run()