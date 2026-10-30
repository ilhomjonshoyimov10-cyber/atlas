# 1) Web
FROM node:20-alpine AS web
WORKDIR /web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# 2) API + model
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY atlas/ atlas/
COPY --from=web /web/dist web/dist
ENV ATLAS_DATA=/data
VOLUME /data
EXPOSE 8000
CMD ["python", "-m", "atlas.cli", "serve"]
