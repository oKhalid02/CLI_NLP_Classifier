import click

from commands.eda import eda as eda_group 
from commands.preprocess import preprocess as preprocess_group
from commands.embed import embed as embed_group


@click.group()
def cli():
    """Arabic NLP Classification CLI Tool"""
    pass

# Register groups from commands/*
cli.add_command(eda_group, name="eda")

cli.add_command(preprocess_group, name="preprocess")

cli.add_command(embed_group, name="embed")


@cli.command()
@click.option("--csv_path", required=True, type=str)
def train(csv_path):
    """Train models (placeholder)"""
    click.echo(f"Training using: {csv_path}")


if __name__ == "__main__":
    cli()
