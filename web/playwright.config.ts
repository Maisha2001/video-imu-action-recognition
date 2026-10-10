import {defineConfig} from '@playwright/test';

export default defineConfig({
  testDir: './e2e', workers: 1, timeout: 90000,
  use: {baseURL: process.env.ACTIVITY_DEMO_URL || 'http://127.0.0.1:4173',
    channel: process.env.ACTIVITY_BROWSER_CHANNEL, headless: true, viewport: {width: 1365,height: 1000}},
  webServer: process.env.ACTIVITY_DEMO_URL ? undefined : {command: 'npx vite preview --host 127.0.0.1 --port 4173',url:'http://127.0.0.1:4173'},
});
