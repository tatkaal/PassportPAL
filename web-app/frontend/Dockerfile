# Build stage
FROM node:18-alpine as build

WORKDIR /app

# Copy package.json and install dependencies
COPY package.json .
RUN npm install

# Copy all files
COPY . .

# Build the React application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy the build from the previous stage
COPY --from=build /app/dist /usr/share/nginx/html

# Configure Nginx to handle frontend routing
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
