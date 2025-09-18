# build stage
FROM node:20-alpine AS build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./frontend/
COPY frontend/ ./frontend/
WORKDIR /app/frontend
RUN npm ci
RUN npm run build

# serve stage
FROM nginx:stable-alpine
COPY --from=build /app/frontend/build /usr/share/nginx/html
# default nginx serves static files
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
