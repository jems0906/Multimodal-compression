import typer
from pathlib import Path
from .pipeline import run_pipeline
from .config import load_config

app = typer.Typer()

@app.command()
def analyze(config: Path = typer.Option(..., help="Path to config file")):
    """Analyze the model before compression."""
    cfg = load_config(config)
    run_pipeline(cfg, stage="analyze")

@app.command()
def compress(config: Path = typer.Option(..., help="Path to config file")):
    """Compress the model."""
    cfg = load_config(config)
    run_pipeline(cfg, stage="compress")

@app.command()
def finetune(config: Path = typer.Option(..., help="Path to config file")):
    """Finetune the compressed model."""
    cfg = load_config(config)
    run_pipeline(cfg, stage="finetune")

@app.command()
def benchmark(config: Path = typer.Option(..., help="Path to config file")):
    """Benchmark the model performance."""
    cfg = load_config(config)
    run_pipeline(cfg, stage="benchmark")

def main():
    """Console script entrypoint for the Typer application."""
    app()

if __name__ == "__main__":
    main()