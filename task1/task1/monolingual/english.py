from task1 import pipelines as ppl
from task1.config import ProjectPaths
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
import torch

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from task1.models.MLP import MLP, GRU
import os
from task1.models.encoders import SentenceTransformerEncoder, Word2VecEncoder, TfidfEncoder
from scipy.stats import randint, loguniform

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    paths = ProjectPaths()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # encoder = SentenceTransformerEncoder(model_name="all-MiniLM-L6-v2")
    encoder = Word2VecEncoder()
    #encoder = TfidfEncoder(max_features=10_000, ngram_range=(1,4))
    classifier = GRU(
        in_features=encoder.get_emb_dim(),
        out_features=1,
        bidirectional=False
    )

    # classifier = MLP(
    #     in_features=encoder.get_emb_dim(),
    #     out_features=1,
    # )

    master_pipeline = ppl.MasterPipeline(
        encoder=encoder,
        classifier=classifier,
        language="english",
        device=device,
    )
    
    # """ Logistic Regression Hyperparameters """
    # lr_space = {
    #     "C": loguniform(1e-4, 1e2),
    #     "penalty": ["l2"],
    #     "solver": ["lbfgs", "saga"],
    #     "max_iter": [1000],
    # }
    
    # classifier = LogisticRegression(
    #     random_state=42,
    #     n_jobs=-1,
    # )
    
    # master_pipeline = ppl.MasterPipelineSklearn(
    #     encoder=encoder,
    #     classifier=classifier,
    #     language="english",
    #     model_hyperparams=lr_space,
    # )
    
    """ Random Forest Classifier Hyperparameters """	
    # rf_space = {
    # "n_estimators": randint(50, 300),
    # "max_depth":    randint(5, 50),
    # "min_samples_split": randint(2, 10),
    # }
    # classifier = RandomForestClassifier(
    #      random_state=42,  
    # ) 
    # master_pipeline = ppl.MasterPipelineSklearn(
    #     encoder=encoder,
    #     classifier=classifier,
    #     language="english",
    #     model_hyperparams=rf_space,
    # )

    
    # """ SVM Classifier Hyperparameters """
    # svm_space = {
    #     "C": loguniform(1e-4, 1e2),
    #     "kernel": ["linear", "rbf", "poly"],
    #     "degree": randint(2, 5),  
    #     "gamma": ["scale", "auto"],
    # }
    # classifier = SVC(
    #     random_state=42,
    #     probability=True,  
    # )
    # master_pipeline = ppl.MasterPipelineSklearn(
    #     encoder=encoder,
    #     classifier=classifier,
    #     language="english",
    #     model_hyperparams=svm_space,
    # )

    master_pipeline.run()
