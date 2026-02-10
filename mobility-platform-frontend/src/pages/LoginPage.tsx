import { useState } from 'react'
import {
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Alert,
  InputAdornment,
  IconButton,
  LinearProgress,
} from '@mui/material'
import { Visibility, VisibilityOff, DirectionsBike } from '@mui/icons-material'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '../stores/authStore'

const LoginPage = () => {
  const [usernameOrEmail, setUsernameOrEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()
  const login = useAuthStore((state) => state.login)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (!usernameOrEmail) {
      setError('Usuario o email requerido')
      return
    }

    if (!password) {
      setError('La contraseña es requerida')
      return
    }

    setLoading(true)

    try {
      await login(usernameOrEmail, password)
      navigate('/dashboard')
    } catch (err: any) {
      setError(err.message || 'Error al iniciar sesión')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card sx={{ maxWidth: 450, width: '100%', mx: 2 }}>
      <CardContent sx={{ p: 4 }}>
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mb: 3 }}>
          <Box
            sx={{
              width: 80,
              height: 80,
              borderRadius: 3,
              background: 'linear-gradient(135deg, #90caf9 0%, #b39ddb 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              mb: 2,
            }}
          >
            <DirectionsBike sx={{ fontSize: 48, color: 'white' }} />
          </Box>
          <Typography variant="h4" fontWeight={600} gutterBottom>
            Mobility Platform
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Gestión de Movilidad Compartida
          </Typography>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 2, borderRadius: 2 }}>
            {error}
          </Alert>
        )}

        {import.meta.env.VITE_ENABLE_MOCK_DATA === 'true' && (
          <Alert severity="info" sx={{ mb: 2, borderRadius: 2 }}>
            <strong>Demo:</strong> superadmin / Admin123!
          </Alert>
        )}

        <form onSubmit={handleSubmit}>
          <TextField
            fullWidth
            label="Usuario o Email"
            type="text"
            value={usernameOrEmail}
            onChange={(e) => setUsernameOrEmail(e.target.value)}
            margin="normal"
            required
            autoComplete="username"
            autoFocus
          />

          <TextField
            fullWidth
            label="Contraseña"
            type={showPassword ? 'text' : 'password'}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            margin="normal"
            required
            autoComplete="current-password"
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton
                    onClick={() => setShowPassword(!showPassword)}
                    edge="end"
                  >
                    {showPassword ? <VisibilityOff /> : <Visibility />}
                  </IconButton>
                </InputAdornment>
              ),
            }}
          />

          {loading && <LinearProgress sx={{ mt: 2, borderRadius: 1 }} />}

          <Button
            fullWidth
            variant="contained"
            size="large"
            type="submit"
            disabled={loading}
            sx={{ mt: 3, mb: 2, py: 1.5 }}
          >
            {loading ? 'Iniciando sesión...' : 'Iniciar Sesión'}
          </Button>
        </form>

        <Typography variant="caption" color="text.secondary" align="center" display="block">
          Plataforma segura con HTTPS y autenticación JWT
        </Typography>
      </CardContent>
    </Card>
  )
}

export default LoginPage
