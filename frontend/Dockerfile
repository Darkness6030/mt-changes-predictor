# UI module: Vite build, then static files served by nginx as an unprivileged user.
FROM node:22-alpine AS build
WORKDIR /ui
COPY package.json package-lock.json ./
RUN npm ci
COPY tsconfig.json vite.config.ts index.html ./
COPY src ./src
# The build fails on a type error, so a broken contract cannot reach the image.
RUN npm run build

FROM nginxinc/nginx-unprivileged:1.27-alpine
COPY --from=build /ui/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
    CMD ["/bin/sh", "-c", "wget -q -O /dev/null http://127.0.0.1:8080/healthz"]
