import { useState, useEffect } from 'react'
import {
  Box,
  Button,
  Card,
  CardContent,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Typography,
  Chip,
  Alert,
  CircularProgress,
  Snackbar,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Grid,
} from '@mui/material'
import {
  Add,
  Edit,
  Delete,
  LocationOn,
} from '@mui/icons-material'
import { stationService, type StationData, type CreateStationInput, type UpdateStationInput } from '../services/stationService'
import { companyService, type CompanyData } from '../services/companyService'
import { useAuthStore } from '../stores/authStore'

const StationsPage = () => {
  const user = useAuthStore((state) => state.user)
  const [stations, setStations] = useState<StationData[]>([])
  const [companies, setCompanies] = useState<CompanyData[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [openDialog, setOpenDialog] = useState(false)
  const [selectedStation, setSelectedStation] = useState<StationData | null>(null)
  const [saving, setSaving] = useState(false)
  
  const [formData, setFormData] = useState({
    name: '',
    address: '',
    latitude: '',
    longitude: '',
    capacity: '',
    status: 'active' as 'active' | 'inactive' | 'maintenance',
    companyId: '',
  })

  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success' as 'success' | 'error',
  })

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    try {
      setLoading(true)
      setError(null)
      const [stationsData, companiesData] = await Promise.all([
        stationService.getStations(),
        companyService.getCompanies(),
      ])
      setStations(stationsData)
      setCompanies(companiesData)
    } catch (err: any) {
      setError(err.message || 'Error al cargar estaciones')
    } finally {
      setLoading(false)
    }
  }

  const handleOpenDialog = (station?: StationData) => {
    if (station) {
      setSelectedStation(station)
      setFormData({
        name: station.name,
        address: station.address,
        latitude: station.latitude.toString(),
        longitude: station.longitude.toString(),
        capacity: station.capacity.toString(),
        status: station.status,
        companyId: station.companyId,
      })
    } else {
      setSelectedStation(null)
      setFormData({
        name: '',
        address: '',
        latitude: '',
        longitude: '',
        capacity: '',
        status: 'active',
        companyId: user?.companyId || '',
      })
    }
    setOpenDialog(true)
  }

  const handleCloseDialog = () => {
    setOpenDialog(false)
    setSelectedStation(null)
  }

  const handleSave = async () => {
    if (!formData.name || !formData.address || !formData.latitude || 
        !formData.longitude || !formData.capacity || !formData.companyId) {
      setSnackbar({
        open: true,
        message: 'Por favor completa todos los campos obligatorios',
        severity: 'error',
      })
      return
    }

    const latitude = parseFloat(formData.latitude)
    const longitude = parseFloat(formData.longitude)
    const capacity = parseInt(formData.capacity)

    if (isNaN(latitude) || isNaN(longitude) || isNaN(capacity)) {
      setSnackbar({
        open: true,
        message: 'Latitud, longitud y capacidad deben ser números válidos',
        severity: 'error',
      })
      return
    }

    try {
      setSaving(true)

      if (selectedStation) {
        const updateInput: UpdateStationInput = {
          name: formData.name,
          address: formData.address,
          latitude,
          longitude,
          capacity,
          status: formData.status,
        }
        await stationService.updateStation(selectedStation.id, updateInput)
        setSnackbar({
          open: true,
          message: 'Estación actualizada correctamente',
          severity: 'success',
        })
      } else {
        const createInput: CreateStationInput = {
          name: formData.name,
          address: formData.address,
          latitude,
          longitude,
          capacity,
          status: formData.status,
          companyId: formData.companyId,
        }
        await stationService.createStation(createInput)
        setSnackbar({
          open: true,
          message: 'Estación creada correctamente',
          severity: 'success',
        })
      }

      handleCloseDialog()
      await loadData()
    } catch (err: any) {
      setSnackbar({
        open: true,
        message: err.message || 'Error al guardar estación',
        severity: 'error',
      })
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async (stationId: string) => {
    if (!window.confirm('¿Estás seguro de que quieres eliminar esta estación?')) {
      return
    }

    try {
      await stationService.deleteStation(stationId)
      setSnackbar({
        open: true,
        message: 'Estación eliminada correctamente',
        severity: 'success',
      })
      await loadData()
    } catch (err: any) {
      setSnackbar({
        open: true,
        message: err.message || 'Error al eliminar estación',
        severity: 'error',
      })
    }
  }

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '400px' }}>
        <CircularProgress />
      </Box>
    )
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
        <Button variant="contained" onClick={loadData}>
          Reintentar
        </Button>
      </Box>
    )
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1" sx={{ fontWeight: 600 }}>
          Estaciones
        </Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => handleOpenDialog()}
        >
          Nueva Estación
        </Button>
      </Box>

      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 2, mb: 3 }}>
        <Card>
          <CardContent>
            <Typography color="text.secondary" gutterBottom>
              Total Estaciones
            </Typography>
            <Typography variant="h4">{stations.length}</Typography>
          </CardContent>
        </Card>
        <Card>
          <CardContent>
            <Typography color="text.secondary" gutterBottom>
              Estaciones Activas
            </Typography>
            <Typography variant="h4">
              {stations.filter(s => s.status === 'active').length}
            </Typography>
          </CardContent>
        </Card>
        <Card>
          <CardContent>
            <Typography color="text.secondary" gutterBottom>
              Capacidad Total
            </Typography>
            <Typography variant="h4">
              {stations.reduce((sum, s) => sum + s.capacity, 0)}
            </Typography>
          </CardContent>
        </Card>
      </Box>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Nombre</TableCell>
              <TableCell>Dirección</TableCell>
              <TableCell>Capacidad</TableCell>
              <TableCell>Disponibles</TableCell>
              <TableCell>Empresa</TableCell>
              <TableCell>Estado</TableCell>
              <TableCell align="right">Acciones</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {stations.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} align="center">
                  <Box sx={{ py: 4 }}>
                    <LocationOn sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                    <Typography variant="body1" color="text.secondary">
                      No hay estaciones registradas
                    </Typography>
                    <Button
                      variant="outlined"
                      startIcon={<Add />}
                      onClick={() => handleOpenDialog()}
                      sx={{ mt: 2 }}
                    >
                      Crear Primera Estación
                    </Button>
                  </Box>
                </TableCell>
              </TableRow>
            ) : (
              stations.map((station) => (
                <TableRow key={station.id} hover>
                  <TableCell>{station.name}</TableCell>
                  <TableCell>{station.address}</TableCell>
                  <TableCell>{station.capacity}</TableCell>
                  <TableCell>
                    <Chip
                      label={`${station.availableDevices || 0} / ${station.capacity}`}
                      size="small"
                      color={(station.availableDevices || 0) > 0 ? 'success' : 'default'}
                    />
                  </TableCell>
                  <TableCell>{station.companyName}</TableCell>
                  <TableCell>
                    <Chip
                      label={
                        station.status === 'active' ? 'Activa' :
                        station.status === 'maintenance' ? 'Mantenimiento' : 'Inactiva'
                      }
                      size="small"
                      color={
                        station.status === 'active' ? 'success' :
                        station.status === 'maintenance' ? 'warning' : 'default'
                      }
                    />
                  </TableCell>
                  <TableCell align="right">
                    <IconButton
                      size="small"
                      onClick={() => handleOpenDialog(station)}
                      color="primary"
                    >
                      <Edit />
                    </IconButton>
                    <IconButton
                      size="small"
                      onClick={() => handleDelete(station.id)}
                      color="error"
                    >
                      <Delete />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </TableContainer>

      <Dialog open={openDialog} onClose={handleCloseDialog} maxWidth="md" fullWidth>
        <DialogTitle>
          {selectedStation ? 'Editar Estación' : 'Nueva Estación'}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 2 }}>
            <TextField
              label="Nombre"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              fullWidth
              required
            />
            <TextField
              label="Dirección"
              value={formData.address}
              onChange={(e) => setFormData({ ...formData, address: e.target.value })}
              fullWidth
              required
              multiline
              rows={2}
            />
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <TextField
                  label="Latitud"
                  value={formData.latitude}
                  onChange={(e) => setFormData({ ...formData, latitude: e.target.value })}
                  fullWidth
                  required
                  type="number"
                  inputProps={{ step: 'any' }}
                  helperText="Ej: 40.4168"
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="Longitud"
                  value={formData.longitude}
                  onChange={(e) => setFormData({ ...formData, longitude: e.target.value })}
                  fullWidth
                  required
                  type="number"
                  inputProps={{ step: 'any' }}
                  helperText="Ej: -3.7038"
                />
              </Grid>
            </Grid>
            <TextField
              label="Capacidad"
              value={formData.capacity}
              onChange={(e) => setFormData({ ...formData, capacity: e.target.value })}
              fullWidth
              required
              type="number"
              helperText="Número máximo de dispositivos"
            />
            <FormControl fullWidth required>
              <InputLabel>Estado</InputLabel>
              <Select
                value={formData.status}
                onChange={(e) => setFormData({ ...formData, status: e.target.value as any })}
                label="Estado"
              >
                <MenuItem value="active">Activa</MenuItem>
                <MenuItem value="inactive">Inactiva</MenuItem>
                <MenuItem value="maintenance">Mantenimiento</MenuItem>
              </Select>
            </FormControl>
            {!selectedStation && (
              <FormControl fullWidth required>
                <InputLabel>Empresa</InputLabel>
                <Select
                  value={formData.companyId}
                  onChange={(e) => setFormData({ ...formData, companyId: e.target.value })}
                  label="Empresa"
                >
                  {companies.map((company) => (
                    <MenuItem key={company.id} value={company.id}>
                      {company.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog} disabled={saving}>
            Cancelar
          </Button>
          <Button
            onClick={handleSave}
            variant="contained"
            disabled={saving}
            startIcon={saving ? <CircularProgress size={20} /> : null}
          >
            {saving ? 'Guardando...' : 'Guardar'}
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert
          onClose={() => setSnackbar({ ...snackbar, open: false })}
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  )
}

export default StationsPage
