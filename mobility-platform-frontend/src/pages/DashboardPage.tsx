import React from 'react'
import { Card, CardContent, Typography, Grid, Box, Chip, Button } from '@mui/material'
import {
  TrendingUp,
  DirectionsBike,
  LocationOn,
  People,
  Business,
  Assessment,
} from '@mui/icons-material'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '../stores/authStore'

const DashboardPage = () => {
  const navigate = useNavigate()
  const user = useAuthStore((state) => state.user)
  const isSuperAdmin = user?.role === 'superadmin'

  const superAdminStats = [
    { title: 'Empresas Totales', value: '2', icon: <Business />, color: '#9c27b0', trend: '+1', link: '/companies' },
    { title: 'Usuarios Totales', value: '3', icon: <People />, color: '#f06292', trend: '+2', link: '/users' },
  ]

  const operationalStats = [
    { title: 'Dispositivos Totales', value: '1,234', icon: <DirectionsBike />, color: '#90caf9', trend: '+12%', link: '/devices' },
    { title: 'Estaciones Activas', value: '48', icon: <LocationOn />, color: '#a5d6a7', trend: '+3', link: '/stations' },
    { title: 'Tasa de Uso', value: '87%', icon: <TrendingUp />, color: '#ffcc80', trend: '+5%', link: '/predictions' },
  ]

  const stats = isSuperAdmin 
    ? superAdminStats
    : operationalStats

  return (
    <Box>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" fontWeight={600} gutterBottom>
          Dashboard {isSuperAdmin && '- Super Admin'}
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Bienvenido, {user?.firstName}. {isSuperAdmin ? 'Gestión completa del sistema.' : 'Resumen de operaciones.'}
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {stats.map((stat, index) => (
          <Grid item xs={12} sm={6} md={isSuperAdmin ? 6 : 4} key={index}>
            <Card 
              sx={{ 
                cursor: 'pointer',
                transition: 'transform 0.2s, box-shadow 0.2s',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: 4,
                }
              }}
              onClick={() => stat.link && navigate(stat.link)}
            >
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Box
                    sx={{
                      width: 48,
                      height: 48,
                      borderRadius: 2,
                      backgroundColor: stat.color + '30',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      mr: 2,
                    }}
                  >
                    {React.cloneElement(stat.icon, { sx: { color: stat.color } })}
                  </Box>
                  <Chip
                    label={stat.trend}
                    size="small"
                    color="success"
                    sx={{ fontWeight: 600 }}
                  />
                </Box>
                <Typography variant="h4" fontWeight={600}>
                  {stat.value}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {stat.title}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {isSuperAdmin && (
        <Grid container spacing={3} sx={{ mt: 2 }}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Business sx={{ color: 'primary.main', mr: 1 }} />
                  <Typography variant="h6" fontWeight={600}>
                    Gestión de Empresas
                  </Typography>
                </Box>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Administra empresas del sistema con CRUD completo. Crea, edita y elimina empresas.
                </Typography>
                <Button 
                  variant="contained" 
                  onClick={() => navigate('/companies')}
                  sx={{ borderRadius: 2 }}
                >
                  Ir a Empresas
                </Button>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <People sx={{ color: 'secondary.main', mr: 1 }} />
                  <Typography variant="h6" fontWeight={600}>
                    Gestión de Usuarios
                  </Typography>
                </Box>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Administra usuarios y asigna a empresas. Control completo de roles y permisos.
                </Typography>
                <Button 
                  variant="contained" 
                  color="secondary"
                  onClick={() => navigate('/users')}
                  sx={{ borderRadius: 2 }}
                >
                  Ir a Usuarios
                </Button>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <Assessment sx={{ color: 'info.main', mr: 1 }} />
            <Typography variant="h6" fontWeight={600}>
              {isSuperAdmin ? 'Panel de Administración' : 'Información del Sistema'}
            </Typography>
          </Box>
          <Typography variant="body2" color="text.secondary">
            {isSuperAdmin 
              ? 'Como Super Admin tienes acceso completo al sistema. Puedes gestionar empresas, usuarios, estaciones, dispositivos y visualizar todas las métricas del sistema.'
              : 'Esta es una versión inicial del dashboard. Los datos mostrados son de ejemplo. Próximamente se integrarán datos reales desde el backend.'
            }
          </Typography>
        </CardContent>
      </Card>
    </Box>
  )
}

export default DashboardPage
