# Build stage - use smaller Node image
FROM node:22-alpine as build

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install all dependencies including dev dependencies needed for build
RUN npm ci && \
    npm cache clean --force

# Copy only the necessary files needed for build (not tests, docs, etc)
COPY public/ public/
COPY src/ src/
COPY vite.config.js postcss.config.js tailwind.config.js index.html ./

# Build the React application with production optimizations
RUN NODE_ENV=production npm run build && \
    rm -rf node_modules

# Production stage - use smallest possible nginx
FROM nginx:alpine-slim

# Configure nginx - copy before other operations for layer caching
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy only the built assets from the build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Create fallback index.html and clean up in a single layer
RUN if [ ! -f /usr/share/nginx/html/index.html ]; then \
    echo '<!DOCTYPE html><html><head><meta charset="UTF-8"><title>PassportPAL</title></head><body><div id="root">Loading PassportPAL application...</div></body></html>' > /usr/share/nginx/html/index.html; \
    fi && \
    # Clean up to reduce image size
    rm -rf /var/cache/apk/* && \
    rm -rf /tmp/* && \
    rm -rf /etc/nginx/conf.d/default.conf.default && \
    # Strip documentation
    rm -rf /usr/share/nginx/html/docs && \
    # Permissions and removing unnecessary files
    find /usr/share/nginx/html -type f -exec chmod 644 {} \; && \
    find /usr/share/nginx/html -type d -exec chmod 755 {} \;

# Setup environment variables
ENV NGINX_ENTRYPOINT_QUIET_LOGS=1

# Expose port
EXPOSE 80

# Run nginx in foreground
CMD ["nginx", "-g", "daemon off;"]
