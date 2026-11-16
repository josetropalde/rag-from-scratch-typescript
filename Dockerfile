FROM node:22-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci

# Copy application source and documents
COPY tsconfig.json ./
COPY src ./src
COPY files ./files

# Run the RAG application
ENTRYPOINT ["./node_modules/.bin/tsx", "src/index.ts"]
