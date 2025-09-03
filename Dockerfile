FROM node:22-bullseye
RUN apt-get update && apt-get install -y \
    tesseract-ocr tesseract-ocr-eng libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build
ENV NODE_ENV=production
ENV PORT=5000
EXPOSE 5000
CMD ["npx", "tsx", "server/index.ts"]