# python:3.9-bullseye is a debian-based linux destribution with python 3.9.
# References:
# * https://stackoverflow.com/a/72465422
# * https://techytok.com/from-zero-to-julia-using-docker/#create-a-julia-container
FROM python:3.9-bullseye as python-base
MAINTAINER FERNANDO B. PEREZ MAURERA

# https://python-poetry.org/docs#ci-recommendations
ENV POETRY_VERSION=1.2.1
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv

# Tell Poetry where to place its cache and virtual environment
ENV POETRY_CACHE_DIR=/opt/.cache

# Install linux dependencies
RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt-get install -y \
      gcc python3-dev build-essential \
    && apt-get clean

# Delete temporal files
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create stage for Poetry installation
FROM python-base as poetry-base

# Creating a virtual environment just for poetry and install it with pip
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

# Create a new stage from the base python image
FROM python-base as app

# Copy Poetry to app image
COPY --from=poetry-base ${POETRY_VENV} ${POETRY_VENV}

# Add Poetry to PATH
ENV PATH="${PATH}:${POETRY_VENV}/bin"

WORKDIR /app

COPY ../RecSysFramework_public /app/RecSysFramework_public

RUN pwd
RUN ls /app
RUN ls /app/RecSysFramework_public

COPY pyproject.toml poetry.lock ./

# [OPTIONAL] Validate the project is properly configured
RUN poetry check

# Install only dependencies (without installing the current project)
RUN poetry install --no-root --no-interaction --no-cache

COPY . /app

RUN poetry install -vvv --no-interaction

# Install Dependencies
CMD ["poetry", "run", "python"]