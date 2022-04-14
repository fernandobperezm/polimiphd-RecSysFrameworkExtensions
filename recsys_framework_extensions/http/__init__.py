import os
import urllib.request

from recsys_framework_extensions.logging import get_logger

logger = get_logger(
    logger_name=__file__,
)


def download_remote_file(
    url: str,
    destination_filename: str,
    force_download=False,
):
    """
    Download a URL to a temporary file.
    See: https://docs.microsoft.com/en-us/azure/open-datasets/dataset-microsoft-news?tabs=azureml-opendatasets#functions
    """

    if (
        not force_download
        and os.path.isfile(destination_filename)
    ):
        logger.info(
            f'Bypassing download of already-downloaded file {os.path.basename(url)}'
        )

        return destination_filename

    logger.info(
        f'Downloading file {os.path.basename(url)} to {destination_filename}'
    )

    urllib.request.urlretrieve(
        url=url,
        filename=destination_filename,
        reporthook=None,
    )
    assert os.path.isfile(destination_filename)

    logger.info(
        f'Downloaded file {os.path.basename(url)} to {destination_filename}, {os.path.getsize(destination_filename)} bytes.'
    )
    return destination_filename
