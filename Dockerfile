FROM node:22-bullseye

# Install system dependencies including ffmpeg
RUN apt-get update && apt-get install -y \
    tesseract-ocr tesseract-ocr-eng libtesseract-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Verify ffmpeg installation
RUN ffmpeg -version
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build
RUN mkdir -p server/public && cp -r dist/public/* server/public/
RUN mkdir -p src/dist && cp dist/index.js src/dist/index.js
RUN mkdir -p src/dist/public && cp -r dist/public/* src/dist/public/
ENV NODE_ENV=production
ENV PORT=5000
EXPOSE 5000
CMD ["node", "src/dist/index.js"]