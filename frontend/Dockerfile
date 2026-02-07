# Frontend Dockerfile - TryHairStyle
# Multi-stage build: Node.js build + Nginx serve

# ============ Stage 1: Build React App ============
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files trước để tận dụng Docker cache
COPY package*.json ./

# Cài dependencies
RUN npm ci --silent

# Copy source code
COPY . .

# Build production bundle
RUN npm run build

# ============ Stage 2: Serve với Nginx ============
FROM nginx:alpine

# Copy build output từ stage 1
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
