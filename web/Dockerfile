# Étape 1 : build de l'interface
FROM node:22-alpine AS build
WORKDIR /app
ENV npm_config_fetch_retries=6 npm_config_fetch_retry_maxtimeout=120000
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci --no-audit --no-fund
COPY . .
RUN npm run build

# Étape 2 : fichiers statiques servis par nginx, /api relayé vers la passerelle
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
