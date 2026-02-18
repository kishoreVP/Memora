from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openrouter_api_key: str = ""
    openrouter_model: str = "openrouter/auto"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    data_dir: Path = Path.home() / ".memora"

    @field_validator("data_dir", mode="before")
    @classmethod
    def expand_home(cls, v):
        return Path(str(v)).expanduser()

    @property
    def index_path(self) -> Path:
        return self.data_dir / "faiss.index"

    @property
    def meta_path(self) -> Path:
        return self.data_dir / "meta.json"

    def ensure_dirs(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
