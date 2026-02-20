import logging
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from memora.config import settings,SUPPORTED,TEXT_EXTENSIONS
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage 



def caption_image(path: Path) -> str:
    llm = ChatOpenAI(
        model="openrouter/auto",
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
    )

    image_bytes = path.read_bytes()
    image_base64 = base64.b64encode(image_bytes).decode()

    message = HumanMessage(
        content=[
            {"type": "text", "text": "You are a image captioner just Caption this image."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                },
            },
        ]
    )

    response = llm.invoke([message])
    return response.content


def read_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in TEXT_EXTENSIONS:
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        logging.getLogger("pypdf").setLevel(logging.ERROR)
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    if ext == ".docx":
        from docx import Document
        return "\n".join(p.text for p in Document(str(path)).paragraphs)
    if ext in {".jpg", ".jpeg", ".png"}:
        caption = caption_image(path)
        print(f"Caption for {path.name}: {caption}")
        return caption
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    return splitter.split_text(text)


def collect_files(path: Path, recursive: bool = False) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED else []
    pattern = "**/*" if recursive else "*"
    return [f for f in path.glob(pattern) if f.is_file() and f.suffix.lower() in SUPPORTED]

