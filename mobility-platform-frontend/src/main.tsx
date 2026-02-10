import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { HelmetProvider } from 'react-helmet-async'
import App from './App'
import './index.css'
import 'leaflet/dist/leaflet.css'

// Verificar que estamos en HTTPS en producción
if (import.meta.env.PROD && window.location.protocol !== 'https:') {
  window.location.href = 'https:' + window.location.href.substring(window.location.protocol.length)
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <HelmetProvider>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </HelmetProvider>
  </React.StrictMode>,
)
