import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // Vite *library mode* does not automatically replace process.env.*.
  // React / React Flow dependencies can otherwise ship this Node-only global
  // into the browser and throw ReferenceError: process is not defined.
  define: {
    'process.env.NODE_ENV': JSON.stringify('production'),
  },
  base: './',
  build: {
    outDir: 'build',
    emptyOutDir: true,
    cssCodeSplit: false,
    lib: {
      entry: './src/index.tsx',
      formats: ['es'],
      fileName: 'index-[hash]',
      cssFileName: 'index',
    },
    rollupOptions: { output: { inlineDynamicImports: true } },
  },
});
