import os

import click

from utils.transformation_utils import Transformation


def validate_source(ctx, param, value):
    if os.path.isdir(value):
        return value
    elif value and not value.lower().endswith('.jpg'):
        raise click.BadParameter('Source file must be a .jpg image.')
    return value


def validate_directory(ctx, param, value):
    if value:
        try:
            if not os.path.exists(value):
                os.makedirs(value)
            if not os.path.isdir(value):
                raise click.BadParameter(f'{value} is not a valid directory.')
        except Exception as e:
            raise click.BadParameter(f'Could not create directory {value}: {e}')
    return value


@click.command()
@click.option('--src', required=True, help='Source image or directory', callback=validate_source)
@click.option('--dst', help='Destination directory', callback=validate_directory)
def transformation(src, dst):
    if os.path.isfile(src):
        click.echo(f'Source file is a valid .jpg image: {src}')
        transform = Transformation(image_path=src)

    elif os.path.isdir(src):
        if not dst:
            raise click.UsageError('If --src is a directory, --dst must also be specified and must be a directory.')
        click.echo(f'Source directory: {src}')
        click.echo(f'Destination directory: {dst}')
        transform = Transformation(input_dir=src, output_dir=dst)
    else:
        raise click.UsageError('Invalid source. Must be either a .jpg image or a directory.')


if __name__ == '__main__':
    transformation()
