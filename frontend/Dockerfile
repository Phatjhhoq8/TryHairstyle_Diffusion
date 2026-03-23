# Frontend Dockerfile - TryHairStyle
# Multi-stage: build React app → serve via Nginx

# === Stage 1: Build ===
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files for caching
COPY package.json package-lock.json ./
RUN npm ci

# Copy source & build
COPY . .
RUN npm run build

# === Stage 2: Serve ===
FROM nginx:1.27-alpine

# Copy built files
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx config (proxy /api → backend)
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
