import { useState, useEffect } from 'react'
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Avatar,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Paper,
  Button,
  IconButton,
  Tooltip,
  CircularProgress,
  Alert
} from '@mui/material'
import {
  Dashboard,
  Business,
  People,
  Map,
  ShowChart,
  LocationOn,
  DirectionsBike,
  Edit,
  Email,
  Badge,
  CheckCircle,
  Block,
  Info,
  Verified
} from '@mui/icons-material'
import { useAuthStore } from '../stores/authStore'
import { useNavigate } from 'react-router-dom'
import { userService, UserData } from '../services/userService'

// Definición de permisos por rol
const rolePermissions = {
  superadmin: {
    name: 'Super Administrador',
    color: '#9c27b0',
    icon: <Verified />,
    description: 'Control total del sistema. Gestión de empresas y usuarios.',
    permissions: [
      { name: 'Dashboard', icon: <Dashboard />, description: 'Acceso al panel principal', allowed: true },
      { name: 'Empresas', icon: <Business />, description: 'Gestión completa de empresas (CRUD)', allowed: true },
      { name: 'Usuarios', icon: <People />, description: 'Gestión completa de usuarios (CRUD)', allowed: true },
      { name: 'Mapa', icon: <Map />, description: 'Visualización del mapa de estaciones', allowed: false },
      { name: 'Predicciones', icon: <ShowChart />, description: 'Acceso a predicciones VAR', allowed: false },
      { name: 'Estaciones', icon: <LocationOn />, description: 'Gestión de estaciones', allowed: false },
      { name: 'Dispositivos', icon: <DirectionsBike />, description: 'Gestión de dispositivos', allowed: false },
    ]
  },
  admin: {
    name: 'Administrador',
    color: '#1976d2',
    icon: <Badge />,
    description: 'Gestión operativa completa de la plataforma de movilidad.',
    permissions: [
      { name: 'Dashboard', icon: <Dashboard />, description: 'Acceso al panel principal', allowed: true },
      { name: 'Empresas', icon: <Business />, description: 'Gestión de empresas', allowed: false },
      { name: 'Usuarios', icon: <People />, description: 'Gestión de usuarios', allowed: false },
      { name: 'Mapa', icon: <Map />, description: 'Visualización del mapa de estaciones', allowed: true },
      { name: 'Predicciones', icon: <ShowChart />, description: 'Acceso a predicciones VAR', allowed: true },
      { name: 'Estaciones', icon: <LocationOn />, description: 'Gestión completa de estaciones (CRUD)', allowed: true },
      { name: 'Dispositivos', icon: <DirectionsBike />, description: 'Gestión completa de dispositivos (CRUD)', allowed: true },
    ]
  },
  operator: {
    name: 'Operador',
    color: '#2e7d32',
    icon: <Badge />,
    description: 'Gestión operativa de estaciones y dispositivos de movilidad.',
    permissions: [
      { name: 'Dashboard', icon: <Dashboard />, description: 'Acceso al panel principal', allowed: true },
      { name: 'Empresas', icon: <Business />, description: 'Gestión de empresas', allowed: false },
      { name: 'Usuarios', icon: <People />, description: 'Gestión de usuarios', allowed: false },
      { name: 'Mapa', icon: <Map />, description: 'Visualización del mapa de estaciones', allowed: true },
      { name: 'Predicciones', icon: <ShowChart />, description: 'Acceso a predicciones VAR', allowed: true },
      { name: 'Estaciones', icon: <LocationOn />, description: 'Gestión completa de estaciones (CRUD)', allowed: true },
      { name: 'Dispositivos', icon: <DirectionsBike />, description: 'Gestión completa de dispositivos (CRUD)', allowed: true },
    ]
  }
}

const SettingsPage = () => {
  const { user: authUser } = useAuthStore()
  const navigate = useNavigate()
  const [userData, setUserData] = useState<UserData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Cargar datos del usuario desde la base de datos
  useEffect(() => {
    const fetchUserData = async () => {
      if (!authUser?.id) {
        setLoading(false)
        return
      }

      try {
        setLoading(true)
        const data = await userService.getUserById(authUser.id)
        setUserData(data)
        setError(null)
      } catch (err: any) {
        console.error('Error loading user data:', err)
        setError(err.message || 'Error al cargar los datos del usuario')
      } finally {
        setLoading(false)
      }
    }

    fetchUserData()
  }, [authUser?.id])

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 400 }}>
        <CircularProgress />
      </Box>
    )
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">{error}</Alert>
      </Box>
    )
  }

  if (!authUser || !userData) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="warning">No hay sesión activa</Alert>
      </Box>
    )
  }

  const userRole = rolePermissions[userData.role.name as keyof typeof rolePermissions] || rolePermissions.operator
  const allowedPermissions = userRole.permissions.filter(p => p.allowed)
  const deniedPermissions = userRole.permissions.filter(p => !p.allowed)

  // Iniciales del usuario
  const initials = `${userData.firstName.charAt(0)}${userData.lastName.charAt(0)}`.toUpperCase()

  return (
    <Box>
      {/* Encabezado */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" fontWeight={600} gutterBottom>
          Mi Perfil
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Información de tu cuenta y permisos del sistema
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {/* Información del Usuario */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', py: 2 }}>
                <Avatar
                  sx={{
                    width: 120,
                    height: 120,
                    bgcolor: userRole.color,
                    fontSize: '3rem',
                    mb: 2,
                    boxShadow: 3
                  }}
                >
                  {initials}
                </Avatar>
                
                <Typography variant="h5" fontWeight={600} gutterBottom align="center">
                  {userData.firstName} {userData.lastName}
                </Typography>
                
                <Chip
                  icon={userRole.icon}
                  label={userRole.name}
                  sx={{
                    bgcolor: userRole.color,
                    color: 'white',
                    fontWeight: 600,
                    mb: 2
                  }}
                />

                <Tooltip title="Editar perfil (próximamente)">
                  <Button
                    variant="outlined"
                    startIcon={<Edit />}
                    disabled
                    sx={{ mt: 2 }}
                  >
                    Editar Perfil
                  </Button>
                </Tooltip>
              </Box>

              <Divider sx={{ my: 2 }} />

              <Box sx={{ px: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Email sx={{ mr: 2, color: 'text.secondary' }} />
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      Email
                    </Typography>
                    <Typography variant="body2" fontWeight={500}>
                      {userData.email}
                    </Typography>
                  </Box>
                </Box>

                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Badge sx={{ mr: 2, color: 'text.secondary' }} />
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      Usuario
                    </Typography>
                    <Typography variant="body2" fontWeight={500}>
                      {userData.username}
                    </Typography>
                  </Box>
                </Box>

                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Info sx={{ mr: 2, color: 'text.secondary' }} />
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      ID de Usuario
                    </Typography>
                    <Typography variant="body2" fontWeight={500}>
                      {userData.id}
                    </Typography>
                  </Box>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Descripción del Rol y Permisos */}
        <Grid item xs={12} md={8}>
          {/* Descripción del Rol */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                {userRole.icon}
                <Typography variant="h6" fontWeight={600} sx={{ ml: 1 }}>
                  Sobre tu Rol: {userRole.name}
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                {userRole.description}
              </Typography>

              <Paper sx={{ p: 2, bgcolor: 'primary.light', color: 'primary.contrastText' }}>
                <Typography variant="body2">
                  <strong>Nota:</strong> Los permisos están definidos a nivel de sistema y solo pueden ser
                  modificados por un Super Administrador.
                </Typography>
              </Paper>
            </CardContent>
          </Card>

          {/* Permisos Permitidos */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <CheckCircle sx={{ color: 'success.main', mr: 1 }} />
                <Typography variant="h6" fontWeight={600}>
                  Funcionalidades Permitidas
                </Typography>
                <Chip
                  label={allowedPermissions.length}
                  size="small"
                  color="success"
                  sx={{ ml: 2 }}
                />
              </Box>
              
              <List>
                {allowedPermissions.map((permission, index) => (
                  <ListItem
                    key={index}
                    sx={{
                      border: '1px solid',
                      borderColor: 'success.light',
                      borderRadius: 2,
                      mb: 1,
                      bgcolor: 'success.lighter',
                      transition: 'all 0.2s',
                      '&:hover': {
                        bgcolor: 'success.light',
                        transform: 'translateX(4px)'
                      }
                    }}
                  >
                    <ListItemIcon sx={{ color: 'success.main' }}>
                      {permission.icon}
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <Typography fontWeight={600}>
                          {permission.name}
                        </Typography>
                      }
                      secondary={permission.description}
                    />
                    <CheckCircle sx={{ color: 'success.main' }} />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>

          {/* Permisos No Permitidos */}
          {deniedPermissions.length > 0 && (
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Block sx={{ color: 'error.main', mr: 1 }} />
                  <Typography variant="h6" fontWeight={600}>
                    Funcionalidades No Disponibles
                  </Typography>
                  <Chip
                    label={deniedPermissions.length}
                    size="small"
                    color="error"
                    sx={{ ml: 2 }}
                  />
                </Box>
                
                <List>
                  {deniedPermissions.map((permission, index) => (
                    <ListItem
                      key={index}
                      sx={{
                        border: '1px solid',
                        borderColor: 'error.light',
                        borderRadius: 2,
                        mb: 1,
                        bgcolor: 'grey.50',
                        opacity: 0.7
                      }}
                    >
                      <ListItemIcon sx={{ color: 'text.disabled' }}>
                        {permission.icon}
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Typography fontWeight={500} color="text.secondary">
                            {permission.name}
                          </Typography>
                        }
                        secondary={
                          <Typography variant="caption" color="text.disabled">
                            {permission.description}
                          </Typography>
                        }
                      />
                      <Block sx={{ color: 'error.main' }} />
                    </ListItem>
                  ))}
                </List>

                <Paper sx={{ p: 2, bgcolor: 'warning.lighter', mt: 2 }}>
                  <Typography variant="caption" color="text.secondary">
                    💡 Si necesitas acceso a estas funcionalidades, contacta con tu Super Administrador.
                  </Typography>
                </Paper>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>

      {/* Estadísticas Rápidas */}
      <Grid container spacing={3} sx={{ mt: 1 }}>
        <Grid item xs={12} md={4}>
          <Card sx={{ bgcolor: 'primary.main', color: 'white' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="h3" fontWeight={600}>
                    {allowedPermissions.length}
                  </Typography>
                  <Typography variant="body2">
                    Funcionalidades Activas
                  </Typography>
                </Box>
                <CheckCircle sx={{ fontSize: 48, opacity: 0.5 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ bgcolor: 'success.main', color: 'white' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="h3" fontWeight={600}>
                    100%
                  </Typography>
                  <Typography variant="body2">
                    Acceso Verificado
                  </Typography>
                </Box>
                <Verified sx={{ fontSize: 48, opacity: 0.5 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ bgcolor: userRole.color, color: 'white' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="h3" fontWeight={600}>
                    {userRole.name.split(' ')[0]}
                  </Typography>
                  <Typography variant="body2">
                    Nivel de Acceso
                  </Typography>
                </Box>
                {userRole.icon}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}

export default SettingsPage

