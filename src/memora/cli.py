import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="memora")
console = Console()

@app.command()
def add(path: Path, recursive: bool = False):
    """Add files to memory. Loads ML model only if necessary."""
    from memora.ingest import collect_files, read_file, chunk_text
    from memora.store import Store
    
    files = collect_files(path, recursive)
    if not files:
        console.print("[yellow]No supported files found.[/]")
        return
    
    store = Store()
    total = 0
    for f in files:
        with console.status(f"Indexing {f.name}..."):
            try:
                text = read_file(f)
                chunks = chunk_text(text)
                n = store.add(chunks, str(f.resolve()))
                total += n
                if n > 0:
                    console.print(f"  [green]✓[/] {f.name} ({n} chunks)")
                else:
                    console.print(f"  [dim]○[/] {f.name} (already indexed)")
            except Exception as e:
                console.print(f"  [red]✗[/] {f.name} ({e})")
                
    console.print(f"\n[bold green]Success![/] Added {total} new chunks.")

@app.command()
def remove(source: str):
    """Instant removal. Does NOT load ML model."""
    from memora.store import MetadataStore
    n = MetadataStore().remove(source)
    if n > 0:
        console.print(f"[green]✓[/] Removed {n} chunks. Index will update on next search.")
    else:
        console.print(f"[yellow]Source not found:[/] {source}")

@app.command(name="list")
def list_docs():
    """Instant listing of documents. Does NOT load ML model."""
    from memora.store import MetadataStore
    sources = MetadataStore().list_sources()
    if not sources:
        console.print("[dim]Memory is empty.[/]")
        return
        
    table = Table(title="Indexed Documents")
    table.add_column("Source", style="cyan")
    table.add_column("Chunks", justify="right", style="green")
    for s in sources:
        table.add_row(s["source"], str(s["chunks"]))
    console.print(table)

@app.command()
def ask(question: str):
    """Search & Answer. Loads ML model and LLM chain."""
    from memora.rag import RAG
    with console.status("Thinking..."):
        try:
            rag = RAG()
            answer = rag.ask(question)
            console.print(f"\n[bold cyan]Memora:[/] {answer}")
        except Exception as e:
            console.print(f"\n[red]Error:[/] {e}")

@app.command()
def stats():
    """Show storage statistics. Does NOT load ML model."""
    from memora.store import MetadataStore
    s = MetadataStore()
    console.print("[bold]Memora Stats[/]")
    console.print(f"  Sources: [cyan]{len(s.list_sources())}[/]")
    console.print(f"  Chunks:  [cyan]{len(s.meta)}[/]")
    if s.needs_rebuild:
        console.print("  Status:  [yellow]Rebuild pending[/]")

if __name__ == "__main__":
    app()
