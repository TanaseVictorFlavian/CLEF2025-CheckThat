from task1 import pipelines as ppl
from task1.config import ProjectPaths
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
import torch
from task1.models.MLP import MLP
import os
from task1.models.encoders import SentenceTransformerEncoder, Word2VecEncoder, TfidfEncoder
from scipy.stats import randint, loguniform


torch.use_deterministic_algorithms(True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    paths = ProjectPaths()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    encoder = SentenceTransformerEncoder(model_name="all-MiniLM-L6-v2")
    
    classifier = LogisticRegression(
        random_state=42,
        n_jobs=-1,
    )
    
    master_pipeline = ppl.MasterPipelineSklearn(
        encoder=encoder,
        classifier=classifier,
        language=["english", "arabic", "bulgarian", "german", "italian"],
        test_language="german",
        model_hyperparams={"C": 0.39079671568228835,
                            "max_iter": 1000,
                            "penalty": "l2",
                            "solver": "lbfgs"
                        },
    )
    
    master_pipeline.run()
