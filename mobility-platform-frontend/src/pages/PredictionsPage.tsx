import { useState } from 'react'
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  AlertTitle
} from '@mui/material'
import {
  TrendingUp,
  TrendingDown,
  Remove,
  CalendarToday,
  AccessTime,
  DirectionsBike
} from '@mui/icons-material'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
  Area,
  AreaChart
} from 'recharts'

// Tipo para las predicciones de cada estación
interface StationPrediction {
  stationId: number
  stationName: string
  currentDevices: number
  predictedDevices: number
  change: number
  changePercent: number
  confidence: 'high' | 'medium' | 'low'
  hourlyPredictions: Array<{
    hour: string
    devices: number
  }>
}

const PredictionsPage = () => {
  // Fecha de la predicción (mañana)
  const tomorrow = new Date()
  tomorrow.setDate(tomorrow.getDate() + 1)
  const predictionDate = tomorrow.toLocaleDateString('es-ES', { 
    weekday: 'long', 
    year: 'numeric', 
    month: 'long', 
    day: 'numeric' 
  })

  // Datos simulados del modelo VAR
  const [predictions] = useState<StationPrediction[]>([
    {
      stationId: 1,
      stationName: 'Estación Atocha',
      currentDevices: 15,
      predictedDevices: 22,
      change: 7,
      changePercent: 46.7,
      confidence: 'high',
      hourlyPredictions: [
        { hour: '08:00', devices: 10 },
        { hour: '10:00', devices: 18 },
        { hour: '12:00', devices: 25 },
        { hour: '14:00', devices: 22 },
        { hour: '16:00', devices: 20 },
        { hour: '18:00', devices: 28 },
        { hour: '20:00', devices: 15 },
        { hour: '22:00', devices: 8 }
      ]
    },
    {
      stationId: 2,
      stationName: 'Estación Retiro',
      currentDevices: 20,
      predictedDevices: 18,
      change: -2,
      changePercent: -10,
      confidence: 'high',
      hourlyPredictions: [
        { hour: '08:00', devices: 12 },
        { hour: '10:00', devices: 16 },
        { hour: '12:00', devices: 22 },
        { hour: '14:00', devices: 20 },
        { hour: '16:00', devices: 18 },
        { hour: '18:00', devices: 24 },
        { hour: '20:00', devices: 14 },
        { hour: '22:00', devices: 10 }
      ]
    },
    {
      stationId: 3,
      stationName: 'Estación Sol',
      currentDevices: 8,
      predictedDevices: 16,
      change: 8,
      changePercent: 100,
      confidence: 'medium',
      hourlyPredictions: [
        { hour: '08:00', devices: 8 },
        { hour: '10:00', devices: 14 },
        { hour: '12:00', devices: 20 },
        { hour: '14:00', devices: 18 },
        { hour: '16:00', devices: 16 },
        { hour: '18:00', devices: 22 },
        { hour: '20:00', devices: 12 },
        { hour: '22:00', devices: 6 }
      ]
    },
    {
      stationId: 4,
      stationName: 'Estación Gran Vía',
      currentDevices: 25,
      predictedDevices: 24,
      change: -1,
      changePercent: -4,
      confidence: 'high',
      hourlyPredictions: [
        { hour: '08:00', devices: 15 },
        { hour: '10:00', devices: 22 },
        { hour: '12:00', devices: 28 },
        { hour: '14:00', devices: 26 },
        { hour: '16:00', devices: 24 },
        { hour: '18:00', devices: 30 },
        { hour: '20:00', devices: 18 },
        { hour: '22:00', devices: 12 }
      ]
    },
    {
      stationId: 5,
      stationName: 'Estación Cibeles',
      currentDevices: 0,
      predictedDevices: 0,
      change: 0,
      changePercent: 0,
      confidence: 'low',
      hourlyPredictions: [
        { hour: '08:00', devices: 0 },
        { hour: '10:00', devices: 0 },
        { hour: '12:00', devices: 0 },
        { hour: '14:00', devices: 0 },
        { hour: '16:00', devices: 0 },
        { hour: '18:00', devices: 0 },
        { hour: '20:00', devices: 0 },
        { hour: '22:00', devices: 0 }
      ]
    },
    {
      stationId: 6,
      stationName: 'Estación Chamberí',
      currentDevices: 18,
      predictedDevices: 21,
      change: 3,
      changePercent: 16.7,
      confidence: 'medium',
      hourlyPredictions: [
        { hour: '08:00', devices: 11 },
        { hour: '10:00', devices: 17 },
        { hour: '12:00', devices: 24 },
        { hour: '14:00', devices: 22 },
        { hour: '16:00', devices: 21 },
        { hour: '18:00', devices: 26 },
        { hour: '20:00', devices: 16 },
        { hour: '22:00', devices: 9 }
      ]
    }
  ])

  const [selectedStation, setSelectedStation] = useState<number>(1)

  const getTrendIcon = (change: number) => {
    if (change > 0) return <TrendingUp color="success" />
    if (change < 0) return <TrendingDown color="error" />
    return <Remove color="disabled" />
  }

  const getTrendColor = (change: number) => {
    if (change > 0) return 'success'
    if (change < 0) return 'error'
    return 'default'
  }

  const getConfidenceColor = (confidence: string) => {
    switch (confidence) {
      case 'high':
        return 'success'
      case 'medium':
        return 'warning'
      case 'low':
        return 'error'
      default:
        return 'default'
    }
  }

  const getConfidenceText = (confidence: string) => {
    switch (confidence) {
      case 'high':
        return 'Alta'
      case 'medium':
        return 'Media'
      case 'low':
        return 'Baja'
      default:
        return confidence
    }
  }

  const selectedStationData = predictions.find(p => p.stationId === selectedStation)

  // Preparar datos para el gráfico de barras comparativo
  const comparisonData = predictions.map(p => ({
    name: p.stationName.replace('Estación ', ''),
    actual: p.currentDevices,
    predicho: p.predictedDevices
  }))

  // Calcular totales
  const totalCurrent = predictions.reduce((sum, p) => sum + p.currentDevices, 0)
  const totalPredicted = predictions.reduce((sum, p) => sum + p.predictedDevices, 0)
  const totalChange = totalPredicted - totalCurrent
  const totalChangePercent = totalCurrent > 0 ? ((totalChange / totalCurrent) * 100).toFixed(1) : 0

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" fontWeight={600} gutterBottom>
            Predicciones de Demanda
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Modelo VAR - Predicción para el día siguiente
          </Typography>
        </Box>
        <Chip
          icon={<CalendarToday />}
          label={predictionDate}
          color="primary"
          variant="outlined"
          sx={{ textTransform: 'capitalize', px: 1 }}
        />
      </Box>

      {/* Alerta informativa */}
      <Alert severity="info" sx={{ mb: 3 }}>
        <AlertTitle>Modelo de Predicción VAR (Vector Autoregression)</AlertTitle>
        Las predicciones están basadas en patrones históricos de uso y factores temporales.
        La confianza indica la precisión esperada según la validación del modelo.
      </Alert>

      {/* Resumen general */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Box
                  sx={{
                    width: 56,
                    height: 56,
                    borderRadius: 2,
                    backgroundColor: 'primary.light',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <DirectionsBike sx={{ color: 'primary.main', fontSize: 32 }} />
                </Box>
                <Box>
                  <Typography variant="h4" fontWeight={600}>
                    {totalCurrent}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Dispositivos Actuales
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Box
                  sx={{
                    width: 56,
                    height: 56,
                    borderRadius: 2,
                    backgroundColor: 'success.light',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <TrendingUp sx={{ color: 'success.main', fontSize: 32 }} />
                </Box>
                <Box>
                  <Typography variant="h4" fontWeight={600}>
                    {totalPredicted}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Predicción Mañana
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Box
                  sx={{
                    width: 56,
                    height: 56,
                    borderRadius: 2,
                    backgroundColor: totalChange >= 0 ? 'success.light' : 'error.light',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  {getTrendIcon(totalChange)}
                </Box>
                <Box>
                  <Typography variant="h4" fontWeight={600}>
                    {totalChange >= 0 ? '+' : ''}{totalChange}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Cambio ({totalChange >= 0 ? '+' : ''}{totalChangePercent}%)
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Gráfico comparativo por estación */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" fontWeight={600} gutterBottom>
            Comparación por Estación
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Dispositivos actuales vs predicción para mañana
          </Typography>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={comparisonData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="actual" fill="#90caf9" name="Actual" />
              <Bar dataKey="predicho" fill="#a5d6a7" name="Predicho" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Tabla detallada */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" fontWeight={600} gutterBottom>
            Predicciones Detalladas por Estación
          </Typography>
        </CardContent>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Estación</TableCell>
                <TableCell align="center">Actual</TableCell>
                <TableCell align="center">Predicción</TableCell>
                <TableCell align="center">Cambio</TableCell>
                <TableCell align="center">% Cambio</TableCell>
                <TableCell align="center">Confianza</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {predictions.map((pred) => (
                <TableRow key={pred.stationId} hover>
                  <TableCell>
                    <Typography fontWeight={500}>{pred.stationName}</Typography>
                  </TableCell>
                  <TableCell align="center">
                    <Chip
                      label={pred.currentDevices}
                      size="small"
                      color="primary"
                      variant="outlined"
                    />
                  </TableCell>
                  <TableCell align="center">
                    <Chip
                      label={pred.predictedDevices}
                      size="small"
                      color="success"
                    />
                  </TableCell>
                  <TableCell align="center">
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 0.5 }}>
                      {getTrendIcon(pred.change)}
                      <Typography
                        variant="body2"
                        color={pred.change >= 0 ? 'success.main' : 'error.main'}
                        fontWeight={600}
                      >
                        {pred.change >= 0 ? '+' : ''}{pred.change}
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell align="center">
                    <Chip
                      label={`${pred.changePercent >= 0 ? '+' : ''}${pred.changePercent.toFixed(1)}%`}
                      size="small"
                      color={getTrendColor(pred.change) as any}
                      variant="outlined"
                    />
                  </TableCell>
                  <TableCell align="center">
                    <Chip
                      label={getConfidenceText(pred.confidence)}
                      size="small"
                      color={getConfidenceColor(pred.confidence) as any}
                    />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Card>

      {/* Predicción horaria por estación */}
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Box>
              <Typography variant="h6" fontWeight={600} gutterBottom>
                Predicción Horaria Detallada
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Distribución estimada de dispositivos por hora
              </Typography>
            </Box>
            <FormControl sx={{ minWidth: 250 }}>
              <InputLabel>Seleccionar Estación</InputLabel>
              <Select
                value={selectedStation}
                label="Seleccionar Estación"
                onChange={(e) => setSelectedStation(Number(e.target.value))}
              >
                {predictions.map((pred) => (
                  <MenuItem key={pred.stationId} value={pred.stationId}>
                    {pred.stationName}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>

          {selectedStationData && (
            <>
              <Box sx={{ mb: 3, p: 2, backgroundColor: 'grey.50', borderRadius: 2 }}>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={3}>
                    <Typography variant="caption" color="text.secondary">
                      Predicción Promedio
                    </Typography>
                    <Typography variant="h6" fontWeight={600}>
                      {selectedStationData.predictedDevices} dispositivos
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={3}>
                    <Typography variant="caption" color="text.secondary">
                      Pico Estimado
                    </Typography>
                    <Typography variant="h6" fontWeight={600}>
                      {Math.max(...selectedStationData.hourlyPredictions.map(h => h.devices))} dispositivos
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={3}>
                    <Typography variant="caption" color="text.secondary">
                      Valle Estimado
                    </Typography>
                    <Typography variant="h6" fontWeight={600}>
                      {Math.min(...selectedStationData.hourlyPredictions.map(h => h.devices))} dispositivos
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={3}>
                    <Typography variant="caption" color="text.secondary">
                      Nivel de Confianza
                    </Typography>
                    <Chip
                      label={getConfidenceText(selectedStationData.confidence)}
                      size="small"
                      color={getConfidenceColor(selectedStationData.confidence) as any}
                      sx={{ mt: 0.5 }}
                    />
                  </Grid>
                </Grid>
              </Box>

              <ResponsiveContainer width="100%" height={350}>
                <AreaChart data={selectedStationData.hourlyPredictions}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="hour" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="devices"
                    stroke="#90caf9"
                    fill="#90caf9"
                    fillOpacity={0.6}
                    name="Dispositivos Estimados"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </>
          )}
        </CardContent>
      </Card>
    </Box>
  )
}

export default PredictionsPage

