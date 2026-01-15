import click

from commands.eda import eda as eda_group  # <-- import the real EDA group


@click.group()
def cli():
    """Arabic NLP Classification CLI Tool"""
    pass


# Register groups from commands/*
cli.add_command(eda_group, name="eda")


@cli.group()
def preprocess():
    """Text preprocessing commands"""
    pass


@cli.group()
def embed():
    """Text embedding commands"""
    pass


@cli.command()
@click.option("--csv_path", required=True, type=str)
def train(csv_path):
    """Train models (placeholder)"""
    click.echo(f"Training using: {csv_path}")


if __name__ == "__main__":
    cli()
