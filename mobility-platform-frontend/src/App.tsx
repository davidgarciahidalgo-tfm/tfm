import { ThemeProvider, CssBaseline } from '@mui/material'
import { Routes, Route, Navigate } from 'react-router-dom'
import { Helmet } from 'react-helmet-async'
import { theme } from './theme'
import { useAuthStore } from './stores/authStore'

// Layouts
import MainLayout from './layouts/MainLayout'
import AuthLayout from './layouts/AuthLayout'

// Pages
import LoginPage from './pages/LoginPage'
import DashboardPage from './pages/DashboardPage'
import CompaniesPage from './pages/CompaniesPage'
import UsersPage from './pages/UsersPage'
import MapPage from './pages/MapPage'
import PredictionsPage from './pages/PredictionsPage'
import StationsPage from './pages/StationsPage'
import DevicesPage from './pages/DevicesPage'
import SettingsPage from './pages/SettingsPage'

// Protected Route Component
const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated)
  
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />
  }
  
  return <>{children}</>
}

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Helmet>
        <title>Mobility Platform - Gestión de Movilidad Compartida</title>
        <meta name="description" content="Plataforma de gestión de movilidad compartida con predicción de demanda mediante modelo VAR" />
      </Helmet>
      
      <Routes>
        {/* Public Routes */}
        <Route element={<AuthLayout />}>
          <Route path="/login" element={<LoginPage />} />
        </Route>

        {/* Protected Routes */}
        <Route
          element={
            <ProtectedRoute>
              <MainLayout />
            </ProtectedRoute>
          }
        >
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/companies" element={<CompaniesPage />} />
          <Route path="/users" element={<UsersPage />} />
          <Route path="/map" element={<MapPage />} />
          <Route path="/predictions" element={<PredictionsPage />} />
          <Route path="/stations" element={<StationsPage />} />
          <Route path="/devices" element={<DevicesPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Route>

        {/* 404 */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </ThemeProvider>
  )
}

export default App
