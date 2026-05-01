"""CLI entry point — typer app with chat, vision, and embeddings commands.

Usage examples:
  uv run python -m cli.main chat "What is the capital of France?"
  uv run python -m cli.main stream "Tell me a haiku about Python."
  uv run python -m cli.main extract "Alice is 32 years old and loves hiking."
  uv run python -m cli.main vision-url "https://example.com/photo.jpg"
  uv run python -m cli.main vision-file ./photo.jpg "What text do you see?"
  uv run python -m cli.main embed "Hello, world"
  uv run python -m cli.main similar "fast car" "sports automobile" "slow train" "rocket"
"""

import typer
from rich.console import Console
from rich.table import Table

from core.logging import configure_logging
from usecases.chat import basic_chat, stream_chat, structured_chat
from usecases.embeddings import embed_text, find_most_similar
from usecases.vision import describe_image_file, describe_image_url

app = typer.Typer(
    name="openai-learn",
    help="OpenAI API learning CLI — chat, vision, embeddings.",
    add_completion=False,
)
console = Console()


def _setup(verbose: bool) -> None:
    configure_logging("DEBUG" if verbose else "INFO")


# ── Use case 1: Chat Completions ──────────────────────────────────────────────


@app.command()
def chat(
    prompt: str = typer.Argument(..., help="User message"),
    system: str = typer.Option(
        "You are a helpful assistant.", "--system", "-s", help="System prompt"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Single-round chat completion."""
    _setup(verbose)
    response = basic_chat(prompt, system=system)
    console.print(response)


@app.command()
def stream(
    prompt: str = typer.Argument(..., help="User message"),
    system: str = typer.Option(
        "You are a helpful assistant.", "--system", "-s", help="System prompt"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Stream chat tokens as they arrive."""
    _setup(verbose)
    for token in stream_chat(prompt, system=system):
        console.print(token, end="", highlight=False)
    console.print()  # trailing newline


@app.command()
def extract(
    text: str = typer.Argument(..., help="Free text to extract name/age from"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Structured extraction: name, age, and summary from free text."""
    _setup(verbose)
    info = structured_chat(text)

    table = Table(title="Extracted Info", show_header=True)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")
    table.add_row("Name", info.name)
    table.add_row("Age", str(info.age))
    table.add_row("Summary", info.summary)
    console.print(table)


# ── Use case 5: Vision ────────────────────────────────────────────────────────


@app.command(name="vision-url")
def vision_url(
    url: str = typer.Argument(..., help="Public image URL"),
    question: str = typer.Option(
        "What is in this image? Describe it in detail.",
        "--question",
        "-q",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Describe an image hosted at a public URL."""
    _setup(verbose)
    response = describe_image_url(url, question=question)
    console.print(response)


@app.command(name="vision-file")
def vision_file(
    path: str = typer.Argument(..., help="Local image file path"),
    question: str = typer.Option(
        "What is in this image? Describe it in detail.",
        "--question",
        "-q",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Describe a local image file (base64-encoded inline)."""
    _setup(verbose)
    try:
        response = describe_image_file(path, question=question)
        console.print(response)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


# ── Use case 6: Embeddings ────────────────────────────────────────────────────


@app.command()
def embed(
    text: str = typer.Argument(..., help="Text to embed"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Embed text and print the first 8 dimensions of the vector."""
    _setup(verbose)
    vector = embed_text(text)
    preview = ", ".join(f"{v:.6f}" for v in vector[:8])
    console.print(f"Dims: [bold]{len(vector)}[/bold]")
    console.print(f"First 8: [{preview}, ...]")


@app.command()
def similar(
    query: str = typer.Argument(..., help="Query string"),
    candidates: list[str] = typer.Argument(..., help="Candidate strings to rank"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Rank candidate strings by semantic similarity to the query."""
    _setup(verbose)
    results = find_most_similar(query, list(candidates))

    table = Table(title=f'Most similar to: "{query}"', show_header=True)
    table.add_column("Rank", style="bold", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Text")

    for rank, (text, score) in enumerate(results, start=1):
        colour = "green" if score > 0.8 else "yellow" if score > 0.5 else "red"
        table.add_row(str(rank), f"[{colour}]{score:.4f}[/{colour}]", text)

    console.print(table)


if __name__ == "__main__":
    app()
