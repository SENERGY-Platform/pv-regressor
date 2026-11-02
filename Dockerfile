FROM python:3.10-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /uvx /bin/
LABEL org.opencontainers.image.source https://github.com/SENERGY-Platform/pv-regressor
WORKDIR /usr/src/app
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends git
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY . .
RUN git log -1 --pretty=format:"commit=%H%ndate=%cd%n" > git_commit
RUN apt-get purge -y git && apt-get auto-remove -y && apt-get clean && rm -rf .git
CMD ["uv", "run", "--locked", "python", "-u", "main.py"]
