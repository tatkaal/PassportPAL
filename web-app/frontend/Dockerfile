# Build stage
FROM node:18-alpine as build

WORKDIR /app

# Copy package.json and install dependencies
COPY package.json package-lock.json ./
RUN npm ci

# Copy all files
COPY . .

# Build the React application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy the build from the previous stage
COPY --from=build /app/dist /usr/share/nginx/html

# Add a fallback index.html if the build directory is empty
RUN if [ ! -f /usr/share/nginx/html/index.html ]; then \
    echo '<!DOCTYPE html><html><head><meta charset="UTF-8"><title>PassportPAL</title></head><body><div id="root">Loading PassportPAL application...</div></body></html>' > /usr/share/nginx/html/index.html; \
    fi

# Configure Nginx to handle frontend routing
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
