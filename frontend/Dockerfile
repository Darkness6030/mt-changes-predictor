# UI module: Vite build, then static files served by nginx as an unprivileged user.
FROM node:22-alpine AS build
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/tsconfig.json frontend/vite.config.ts frontend/index.html ./
COPY frontend/src ./src
# The checked-in demo configuration makes a clean clone build without a local .env.
COPY .env.example ./.env
# The build fails on a type error, so a broken contract cannot reach the image.
ARG VITE_YANDEX_MAPS_API_KEY
# An empty Compose build argument must not mask Vite's value from .env.
RUN if [ -z "$VITE_YANDEX_MAPS_API_KEY" ]; then unset VITE_YANDEX_MAPS_API_KEY; fi; \
    npm run build

FROM nginxinc/nginx-unprivileged:1.27-alpine
COPY --from=build /ui/dist /usr/share/nginx/html
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
    CMD ["/bin/sh", "-c", "wget -q -O /dev/null http://127.0.0.1:8080/healthz"]
