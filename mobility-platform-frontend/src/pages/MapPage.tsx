import { Box, Typography, Card, CardContent, Grid } from '@mui/material'
import { LocationOn, DirectionsBike, CheckCircle } from '@mui/icons-material'
import StationMap from '../components/maps/StationMap'

const MapPage = () => {
  return (
    <Box>
      <Typography variant="h4" fontWeight={600} gutterBottom>
        Mapa Interactivo
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Visualización geográfica de estaciones y disponibilidad en tiempo real
      </Typography>

      {/* Estadísticas rápidas */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={4}>
          <Card>
            <CardContent sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Box
                sx={{
                  width: 48,
                  height: 48,
                  borderRadius: 2,
                  backgroundColor: 'primary.light',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <LocationOn sx={{ color: 'primary.main' }} />
              </Box>
              <Box>
                <Typography variant="h5" fontWeight={600}>
                  6
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Estaciones Totales
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={4}>
          <Card>
            <CardContent sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Box
                sx={{
                  width: 48,
                  height: 48,
                  borderRadius: 2,
                  backgroundColor: 'success.light',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <CheckCircle sx={{ color: 'success.main' }} />
              </Box>
              <Box>
                <Typography variant="h5" fontWeight={600}>
                  5
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Estaciones Activas
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={4}>
          <Card>
            <CardContent sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Box
                sx={{
                  width: 48,
                  height: 48,
                  borderRadius: 2,
                  backgroundColor: 'info.light',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <DirectionsBike sx={{ color: 'info.main' }} />
              </Box>
              <Box>
                <Typography variant="h5" fontWeight={600}>
                  86
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Bicicletas Disponibles
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Mapa */}
      <Card sx={{ height: 'calc(100vh - 400px)', minHeight: 500 }}>
        <CardContent sx={{ height: '100%', p: 0 }}>
          <StationMap />
        </CardContent>
      </Card>
    </Box>
  )
}

export default MapPage
