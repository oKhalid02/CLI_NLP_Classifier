import click

from commands.eda import eda as eda_group
from commands.preprocess import preprocess as preprocess_group
from commands.embed import embed as embed_group
from commands.train import train


@click.group()
def cli():
    """Arabic NLP Classification CLI Tool"""
    pass


cli.add_command(eda_group, name="eda")
cli.add_command(preprocess_group, name="preprocess")
cli.add_command(embed_group, name="embed")
cli.add_command(train)   # <-- SINGLE command


if __name__ == "__main__":
    cli()
