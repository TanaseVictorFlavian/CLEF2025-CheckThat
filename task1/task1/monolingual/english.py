from task1 import pipelines as ppl
from task1.config import ProjectPaths
from sentence_transformers import SentenceTransformer
import torch
from task1.models.MLP import MLP
import os


torch.use_deterministic_algorithms(True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    paths = ProjectPaths()

    # Check if MPS is available and set device accordingly
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize the model with explicit device handling
    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device=device,
        cache_folder="./cache",  # Add cache folder to avoid permission issues
    )

    classifier = MLP(
        in_features=embedding_model.get_sentence_embedding_dimension(),
        out_features=1,
    )

    master_pipeline = ppl.MasterPipeline(
        paths=paths,
        data_path=paths.english_data_dir,
        embedding_model=embedding_model,
        classifier=classifier,
        language="en",
    )

    master_pipeline.run()
