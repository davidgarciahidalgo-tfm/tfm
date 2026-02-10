import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import mkcert from 'vite-plugin-mkcert'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    mkcert() // Genera certificados SSL locales automáticamente
  ],
  server: {
    https: true,
    port: 3000,
    host: true,
    strictPort: true,
    // Configuración de seguridad
    headers: {
      'X-Frame-Options': 'DENY',
      'X-Content-Type-Options': 'nosniff',
      'X-XSS-Protection': '1; mode=block',
      'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
      'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline' https://api.mapbox.com; img-src 'self' data: https: blob:; font-src 'self' data:; connect-src 'self' https://api.mapbox.com https://*.supabase.co wss://*.supabase.co"
    }
  },
  build: {
    sourcemap: false, // No exponer código fuente en producción
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'react-router-dom'],
          mui: ['@mui/material', '@mui/icons-material'],
          maps: ['mapbox-gl', 'react-map-gl'],
          charts: ['recharts']
        }
      }
    }
  }
})
