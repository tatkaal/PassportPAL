# Build stage - use smaller Node image
FROM node:22-alpine as build

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies with clean npm cache
RUN npm ci && npm cache clean --force

# Copy source files
COPY public/ public/
COPY src/ src/
COPY vite.config.js .
COPY postcss.config.js .
COPY tailwind.config.js .
COPY index.html .

# Build the React application
RUN npm run build && \
    rm -rf node_modules

# Production stage - use smaller Alpine nginx
FROM nginx:alpine

# Configure nginx - copy before other operations for layer caching
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built app
COPY --from=build /app/dist /usr/share/nginx/html

# Create fallback index.html for better user experience
RUN if [ ! -f /usr/share/nginx/html/index.html ]; then \
    echo '<!DOCTYPE html><html><head><meta charset="UTF-8"><title>PassportPAL</title></head><body><div id="root">Loading PassportPAL application...</div></body></html>' > /usr/share/nginx/html/index.html; \
    fi && \
    rm -rf /var/cache/apk/* && \
    rm -rf /tmp/*

# Run nginx as root for simplicity (for now)
# This is not ideal for production, but works around permission issues with Docker volumes

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
