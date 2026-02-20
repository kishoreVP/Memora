import asyncio
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, RichLog
from textual.containers import VerticalScroll
from rich.markup import escape

class MemoraApp(App):
    CSS = """
    RichLog { height: 1fr; border: solid $accent; padding: 1; }
    Input { dock: bottom; margin: 1 0 0 0; }
    """
    TITLE = "Memora Chat"
    BINDINGS = [("ctrl+q", "quit", "Quit")]

    def __init__(self):
        super().__init__()
        self._rag = None
        self._model_loaded = False

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield RichLog(id="log", markup=True, wrap=True)
        yield Input(placeholder="Loading model... please wait", id="input", disabled=True)
        yield Footer()

    def on_mount(self):
        log = self.query_one("#log", RichLog)
        log.write("[bold cyan]Memora[/] starting up...")
        log.write("[dim]Loading embedding model in background...[/]\n")
        asyncio.create_task(self._load_rag_async())

    async def _load_rag_async(self):
        """Load RAG model in background without blocking UI."""
        # Move ALL heavy imports here, after TUI is visible
        loop = asyncio.get_event_loop()
        
        def load_model():
            import warnings
            import logging
            warnings.filterwarnings("ignore")
            logging.getLogger("httpx").setLevel(logging.ERROR)
            logging.getLogger("openai").setLevel(logging.ERROR)
            logging.getLogger("langchain").setLevel(logging.ERROR)
            
            from memora.rag import RAG
            return RAG()
        
        # Run in thread pool so it doesn't block
        self._rag = await loop.run_in_executor(None, load_model)
        
        self._model_loaded = True
        
        log = self.query_one("#log", RichLog)
        inp = self.query_one("#input", Input)
        
        log.write("[green]✓ Model loaded. Ask away![/]\n")
        inp.placeholder = "Ask something... (Ctrl+Q to quit)"
        inp.disabled = False
        inp.focus()

    async def on_input_submitted(self, event: Input.Submitted):
        q = event.value.strip()
        if not q:
            return
        
        if not self._model_loaded:
            log = self.query_one("#log", RichLog)
            log.write("[yellow]Please wait for model to finish loading...[/]\n")
            return
        
        log = self.query_one("#log", RichLog)
        inp = self.query_one("#input", Input)
        inp.value = ""
        
        log.write(f"\n[bold yellow]You:[/] {escape(q)}")
        log.write("[dim]Thinking...[/]")
        
        answer = await asyncio.get_event_loop().run_in_executor(None, self._rag.ask, q)
        log.write(f"[bold cyan]Memora:[/] {escape(answer)}\n")
