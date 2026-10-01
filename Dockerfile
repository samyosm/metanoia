# Metanoia: the Python model server and the website in one image.
#   docker build -t metanoia .                     (CPU)
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 -t metanoia .
FROM node:22-slim AS web
WORKDIR /web
RUN corepack enable
COPY interface/package.json interface/pnpm-lock.yaml interface/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY interface/ ./
RUN pnpm build \
    && cp -r .next/static .next/standalone/.next/static \
    && cp -r public .next/standalone/public

FROM python:3.12-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN useradd --create-home app
WORKDIR /app
COPY --from=web /usr/local/bin/node /usr/local/bin/node
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir "torch==2.11.0" --index-url ${TORCH_INDEX} \
    && pip install --no-cache-dir "transformers==5.17.0" ".[model,hf]"
COPY --from=web /web/.next/standalone ./interface
COPY model_cards.toml ./
ENV METANOIA_ROOT=/app \
    METANOIA_BIN=metanoia \
    METANOIA_DEVICE=cpu \
    HF_HOME=/hf \
    HOSTNAME=0.0.0.0 \
    PORT=3000 \
    NODE_ENV=production
RUN mkdir /hf && chown app /hf
USER app
EXPOSE 3000
CMD ["node", "interface/server.js"]
