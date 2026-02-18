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

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield RichLog(id="log", markup=True, wrap=True)
        yield Input(placeholder="Ask something... (Ctrl+Q to quit)", id="input")
        yield Footer()

    def on_mount(self):
        log = self.query_one("#log", RichLog)
        log.write("[bold cyan]Memora[/] ready. Loading model...")
        self.call_later(self._load_rag)

    def _load_rag(self):
        import warnings
        import logging
        warnings.filterwarnings("ignore")
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("openai").setLevel(logging.ERROR)
        logging.getLogger("langchain").setLevel(logging.ERROR)

        from memora.rag import RAG

        self._rag = RAG()
        self.query_one("#log", RichLog).write("[green]Model loaded. Ask away![/]\n")

    async def on_input_submitted(self, event: Input.Submitted):
        q = event.value.strip()
        if not q:
            return
        log = self.query_one("#log", RichLog)
        inp = self.query_one("#input", Input)
        inp.value = ""
        log.write(f"\n[bold yellow]You:[/] {escape(q)}")
        log.write("[dim]Thinking...[/]")
        answer = await asyncio.get_event_loop().run_in_executor(None, self._rag.ask, q)
        log.write(f"[bold cyan]Memora:[/] {escape(answer)}\n")