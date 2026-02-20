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

    @property
    def bm25_path(self) -> Path:
        return self.data_dir / "bm25.pkl"

    def ensure_dirs(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)


SUPPORTED = {
    ".txt", ".md", ".pdf", ".docx", ".jpg", ".jpeg", ".png", ".py", ".js", 
    ".html", ".css", ".json", ".csv", ".xml", ".yaml", ".yml", ".go", ".rs", 
    ".cpp", ".c", ".h", ".hpp", ".sh", ".bat", ".ps1", ".sql", ".mdx", 
    ".ipynb", ".r", ".swift", ".kt", ".dart", ".lua", ".scala", ".hs", 
    ".erl", ".ex", ".exs", ".clj", ".cljs", ".groovy", ".vb", ".fsharp", 
    ".elm", ".nim", ".zig", ".asm", ".s", ".v", ".sv", ".vhd", ".vhdl", 
    ".verilog", ".systemverilog", ".m", ".mat", ".rmd", ".rmarkdown", 
    ".tex", ".bib", ".sty", ".cls", ".dtx", ".ins", ".ltx", ".log", ".aux", 
    ".out", ".toc", ".lof", ".lot", ".fls", ".fdb_latexmk", ".synctex.gz", 
    ".synctex", "Makefile", "Dockerfile", "Vagrantfile", "CMakeLists.txt", 
    "build.gradle", "pom.xml", "package.json", "yarn.lock", "Gemfile", 
    "requirements.txt", "Pipfile", "Pipfile.lock", "environment.yml", 
    "conda.yaml", "conda.yml", ".env", ".env.local", ".env.development", 
    ".env.production", ".env.test"
}

TEXT_EXTENSIONS = SUPPORTED - {".jpg", ".jpeg", ".png", ".pdf", ".docx"}

settings = Settings()
