import click
from .processor import process

@click.command()
@click.option('--config-file', '-c', required=False, type=click.Path(),
              help='Full path to an INI-style configuration text file.')

def cli(config_file):
    process(config_file=config_file)
