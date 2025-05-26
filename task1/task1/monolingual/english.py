from task1 import pipelines as ppl
from task1.config import ProjectPaths
from sklearn.ensemble import RandomForestClassifier 
import torch
from task1.models.MLP import MLP
import os
from task1.models.encoders import SentenceTransformerEncoder, Word2VecEncoder

torch.use_deterministic_algorithms(True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    paths = ProjectPaths()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    encoder = SentenceTransformerEncoder(model_name="all-MiniLM-L6-v2")

    classifier = MLP(
        in_features=encoder.get_emb_dim(),
        out_features=1
    )

    master_pipeline = ppl.MasterPipeline(
        encoder=encoder,
        classifier=classifier,
        language="english",
    )


    # classifier = RandomForestClassifier(
    #     n_estimators=100,
    #     max_depth=10,
    #     random_state=42,
    # )
    # master_pipeline = ppl.MasterPipelineSklearn(
    #     paths=paths,
    #     data_path=paths.english_data_dir,
    #     embedding_model=embedding_model,
    #     classifier=classifier,
    #     language="en",
    # )

    master_pipeline.run()
