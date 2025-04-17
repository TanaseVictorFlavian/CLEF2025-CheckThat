from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path

class ProjectPaths(BaseModel):
    """
    Configuration for project paths.

    Attrib
    """

    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data")
    checkpoints_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[0] / "model_checkpoints")
    # Language-specific data directories
    english_data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "english")
    german_data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "german")
    italian_data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "italian")
    bulgarian_data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "bulgarian")
    arabic_data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "arabic")

    weights_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[0] / "model_checkpoints" / "weights")
    run_info_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[0] / "model_checkpoints" / "runs")
    
    model_config = ConfigDict(frozen=True)
    
