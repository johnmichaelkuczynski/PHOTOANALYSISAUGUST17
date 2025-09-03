#!/usr/bin/env node

// Simple startup script for production
const { spawn } = require('child_process');
const path = require('path');

process.env.NODE_ENV = 'production';
process.env.PORT = process.env.PORT || '5000';

// Run tsx server/index.ts
const serverPath = path.join(__dirname, 'server', 'index.ts');
const tsx = spawn('npx', ['tsx', serverPath], {
  stdio: 'inherit',
  env: process.env
});

tsx.on('close', (code) => {
  console.log(`Server process exited with code ${code}`);
  process.exit(code);
});

tsx.on('error', (err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});