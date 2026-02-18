import warnings
import logging
import os

# Suppress model loading warnings globally before any lazy imports trigger them
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="memora", help="Terminal Memory Assistant")
console = Console()

@app.command()
def add(
    path: Path = typer.Argument(..., help="File or directory to add"),
    recursive: bool = typer.Option(False, "-r", "--recursive", help="Recurse into directories"),
):
    """Add files to memory."""
    from memora.ingest import collect_files, read_file, chunk_text
    from memora.store import Store

    store = Store()
    files = collect_files(path, recursive)
    if not files:
        console.print("[yellow]No supported files found.[/]")
        raise typer.Exit()

    total = 0
    for f in files:
        try:
            with console.status(f"Processing {f.name}..."):
                text = read_file(f)
                chunks = chunk_text(text)
                n = store.add(chunks, str(f.resolve()))
                total += n
                console.print(f"  [green]✓[/] {f.name} → {n} chunks")
        except Exception as e:
            console.print(f"  [red]✗[/] {f.name} → {e}")

    console.print(f"\n[bold green]Added {total} chunks from {len(files)} file(s)[/]")


@app.command()
def ask(question: str = typer.Argument(..., help="Question to ask")):
    """Ask a question (one-shot)."""
    from memora.rag import RAG

    rag = RAG()
    with console.status("Thinking..."):
        answer = rag.ask(question)
    console.print(f"\n[bold cyan]Answer:[/] {answer}")


@app.command()
def chat():
    """Interactive chat (TUI)."""
    from memora.tui import MemoraApp

    MemoraApp().run()


@app.command(name="list")
def list_docs():
    """List indexed documents."""
    from memora.store import Store

    sources = Store().list_sources()
    if not sources:
        console.print("[yellow]No documents indexed.[/]")
        raise typer.Exit()

    table = Table(title="Indexed Documents")
    table.add_column("Source", style="cyan", no_wrap=False)
    table.add_column("Chunks", justify="right", style="green")
    for s in sources:
        table.add_row(s["source"], str(s["chunks"]))
    console.print(table)


@app.command()
def stats():
    """Show index statistics."""
    from memora.store import Store

    s = Store().stats()
    console.print(f"[bold]Memora Stats[/]")
    console.print(f"  Sources: [cyan]{s['total_sources']}[/]")
    console.print(f"  Chunks:  [cyan]{s['total_chunks']}[/]")
    console.print(f"  Vectors: [cyan]{s['index_size']}[/]")


@app.command()
def remove(source: str = typer.Argument(..., help="Source path to remove")):
    """Remove a document from memory."""
    from memora.store import Store

    n = Store().remove(source)
    if n:
        console.print(f"[green]Removed {n} chunks from {source}[/]")
    else:
        console.print(f"[yellow]No document found: {source}[/]")


if __name__ == "__main__":
    app()
