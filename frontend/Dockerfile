FROM node:20-slim AS base

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./

# The API URL the browser will call. Override at build/run time if the
# API isn't reachable at localhost:8000 (e.g. in docker-compose, this is
# passed in as http://localhost:8000 since the browser -- not the
# container -- makes the request, so it needs the host-mapped port).
ENV NEXT_PUBLIC_API_URL=http://localhost:8000

RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
