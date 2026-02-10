/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string
  readonly VITE_API_TIMEOUT: string
  readonly VITE_MAPBOX_TOKEN: string
  readonly VITE_ENABLE_HTTPS: string
  readonly VITE_MAX_LOGIN_ATTEMPTS: string
  readonly VITE_SESSION_TIMEOUT_MINUTES: string
  readonly VITE_ENABLE_MOCK_DATA: string
  readonly VITE_ENABLE_DEVTOOLS: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
