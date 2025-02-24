"""Load project configurations from .env files.
Provides easy access to paths and credentials used in the project.
Meant to be used as an imported module.

If `config.py` is run on its own, it will create the appropriate
directories.

For information about the rationale behind decouple and this module,
see https://pypi.org/project/python-decouple/

Note that decouple mentions that it will help to ensure that
the project has "only one configuration module to rule all your instances."
This is achieved by putting all the configuration into the `.env` file.
You can have different sets of variables for difference instances, 
such as `.env.development` or `.env.production`. You would only
need to copy over the settings from one into `.env` to switch
over to the other configuration, for example.

"""

from decouple import config as _config
from pathlib import Path
from pandas import to_datetime


# Absolute path to root directory of the project
BASE_DIR = Path(__file__).absolute().parent.parent

def if_relative_make_abs(path):
    path = Path(path)
    if path.is_absolute():
        abs_path = path.resolve()
    else:
        abs_path = (BASE_DIR / path).resolve()
    return abs_path


# fmt: off
## Other .env variables
WRDS_USERNAME = _config("WRDS_USERNAME", default="")
NASDAQ_API_KEY = _config("NASDAQ_API_KEY", default="")
START_DATE = _config("START_DATE", default="1913-01-01", cast=to_datetime)
END_DATE = _config("END_DATE", default="2024-12-31", cast=to_datetime)
USER = _config("USER", default="")
PIPELINE_DEV_MODE = _config("PIPELINE_DEV_MODE", default=True, cast=bool)
PIPELINE_THEME = _config("PIPELINE_THEME", default="pipeline")

## Paths
DATA_DIR = if_relative_make_abs(_config('DATA_DIR', default=Path('_data'), cast=Path))
RAW_DATA_DIR = Path(DATA_DIR / "raw")
PROCESSED_DATA_DIR = Path(DATA_DIR / "processed")
MANUAL_DATA_DIR = if_relative_make_abs(_config('MANUAL_DATA_DIR', default=Path('data_manual'), cast=Path))
LOG_DIR = if_relative_make_abs(_config('LOG_DIR', default=Path('logs'), cast=Path))
OUTPUT_DIR = if_relative_make_abs(_config('OUTPUT_DIR', default=Path('_output'), cast=Path))
PUBLISH_DIR = if_relative_make_abs(_config('PUBLISH_DIR', default=Path('docs'), cast=Path))

# Plot settings
PLOT_WIDTH = 12
PLOT_HEIGHT = 8


def create_dirs():
    ## If they don't exist, create the _data and _output directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MANUAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    #PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    create_dirs()