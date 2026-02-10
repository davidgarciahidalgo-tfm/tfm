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
  DirectionsBike,
  ElectricBike,
  ElectricScooter,
} from '@mui/icons-material'
import { deviceService, type DeviceData, type CreateDeviceInput, type UpdateDeviceInput } from '../services/deviceService'
import { stationService, type StationData } from '../services/stationService'
import { companyService, type CompanyData } from '../services/companyService'
import { useAuthStore } from '../stores/authStore'

const DevicesPage = () => {
  const user = useAuthStore((state) => state.user)
  const [devices, setDevices] = useState<DeviceData[]>([])
  const [stations, setStations] = useState<StationData[]>([])
  const [companies, setCompanies] = useState<CompanyData[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [openDialog, setOpenDialog] = useState(false)
  const [selectedDevice, setSelectedDevice] = useState<DeviceData | null>(null)
  const [saving, setSaving] = useState(false)
  
  const [formData, setFormData] = useState({
    code: '',
    type: 'bike' as 'bike' | 'ebike' | 'scooter',
    status: 'available' as 'available' | 'in_use' | 'maintenance' | 'retired',
    batteryLevel: '',
    stationId: '',
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
      const [devicesData, stationsData, companiesData] = await Promise.all([
        deviceService.getDevices(),
        stationService.getStations(),
        companyService.getCompanies(),
      ])
      setDevices(devicesData)
      setStations(stationsData)
      setCompanies(companiesData)
    } catch (err: any) {
      setError(err.message || 'Error al cargar dispositivos')
    } finally {
      setLoading(false)
    }
  }

  const handleOpenDialog = (device?: DeviceData) => {
    if (device) {
      setSelectedDevice(device)
      setFormData({
        code: device.code,
        type: device.type,
        status: device.status,
        batteryLevel: device.batteryLevel?.toString() || '',
        stationId: device.stationId || '',
        companyId: device.companyId,
      })
    } else {
      setSelectedDevice(null)
      setFormData({
        code: '',
        type: 'bike',
        status: 'available',
        batteryLevel: '',
        stationId: '',
        companyId: user?.companyId || '',
      })
    }
    setOpenDialog(true)
  }

  const handleCloseDialog = () => {
    setOpenDialog(false)
    setSelectedDevice(null)
  }

  const handleSave = async () => {
    if (!formData.code || !formData.type || !formData.companyId) {
      setSnackbar({
        open: true,
        message: 'Por favor completa todos los campos obligatorios',
        severity: 'error',
      })
      return
    }

    // Validar battery level para e-bikes y scooters
    const batteryLevel = formData.batteryLevel ? parseInt(formData.batteryLevel) : undefined
    if ((formData.type === 'ebike' || formData.type === 'scooter') && formData.batteryLevel) {
      if (isNaN(batteryLevel!) || batteryLevel! < 0 || batteryLevel! > 100) {
        setSnackbar({
          open: true,
          message: 'El nivel de batería debe estar entre 0 y 100',
          severity: 'error',
        })
        return
      }
    }

    try {
      setSaving(true)

      if (selectedDevice) {
        const updateInput: UpdateDeviceInput = {
          code: formData.code,
          type: formData.type,
          status: formData.status,
          batteryLevel,
          stationId: formData.stationId || undefined,
        }
        await deviceService.updateDevice(selectedDevice.id, updateInput)
        setSnackbar({
          open: true,
          message: 'Dispositivo actualizado correctamente',
          severity: 'success',
        })
      } else {
        const createInput: CreateDeviceInput = {
          code: formData.code,
          type: formData.type,
          status: formData.status,
          batteryLevel,
          stationId: formData.stationId || undefined,
          companyId: formData.companyId,
        }
        await deviceService.createDevice(createInput)
        setSnackbar({
          open: true,
          message: 'Dispositivo creado correctamente',
          severity: 'success',
        })
      }

      handleCloseDialog()
      await loadData()
    } catch (err: any) {
      setSnackbar({
        open: true,
        message: err.message || 'Error al guardar dispositivo',
        severity: 'error',
      })
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async (deviceId: string) => {
    if (!window.confirm('¿Estás seguro de que quieres eliminar este dispositivo?')) {
      return
    }

    try {
      await deviceService.deleteDevice(deviceId)
      setSnackbar({
        open: true,
        message: 'Dispositivo eliminado correctamente',
        severity: 'success',
      })
      await loadData()
    } catch (err: any) {
      setSnackbar({
        open: true,
        message: err.message || 'Error al eliminar dispositivo',
        severity: 'error',
      })
    }
  }

  const getDeviceIcon = (type: string) => {
    switch (type) {
      case 'bike': return <DirectionsBike />
      case 'ebike': return <ElectricBike />
      case 'scooter': return <ElectricScooter />
      default: return <DirectionsBike />
    }
  }

  const getDeviceTypeLabel = (type: string) => {
    switch (type) {
      case 'bike': return 'Bicicleta'
      case 'ebike': return 'E-Bike'
      case 'scooter': return 'Patinete'
      default: return type
    }
  }

  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'available': return 'Disponible'
      case 'in_use': return 'En uso'
      case 'maintenance': return 'Mantenimiento'
      case 'retired': return 'Retirado'
      default: return status
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
          Dispositivos
        </Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => handleOpenDialog()}
        >
          Nuevo Dispositivo
        </Button>
      </Box>

      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 2, mb: 3 }}>
        <Card>
          <CardContent>
            <Typography color="text.secondary" gutterBottom>
              Total Dispositivos
            </Typography>
            <Typography variant="h4">{devices.length}</Typography>
          </CardContent>
        </Card>
        <Card>
          <CardContent>
            <Typography color="text.secondary" gutterBottom>
              Disponibles
            </Typography>
            <Typography variant="h4">
              {devices.filter(d => d.status === 'available').length}
            </Typography>
          </CardContent>
        </Card>
        <Card>
          <CardContent>
            <Typography color="text.secondary" gutterBottom>
              En Uso
            </Typography>
            <Typography variant="h4">
              {devices.filter(d => d.status === 'in_use').length}
            </Typography>
          </CardContent>
        </Card>
        <Card>
          <CardContent>
            <Typography color="text.secondary" gutterBottom>
              Mantenimiento
            </Typography>
            <Typography variant="h4">
              {devices.filter(d => d.status === 'maintenance').length}
            </Typography>
          </CardContent>
        </Card>
      </Box>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Código</TableCell>
              <TableCell>Tipo</TableCell>
              <TableCell>Estado</TableCell>
              <TableCell>Batería</TableCell>
              <TableCell>Estación</TableCell>
              <TableCell>Empresa</TableCell>
              <TableCell align="right">Acciones</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {devices.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} align="center">
                  <Box sx={{ py: 4 }}>
                    <DirectionsBike sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                    <Typography variant="body1" color="text.secondary">
                      No hay dispositivos registrados
                    </Typography>
                    <Button
                      variant="outlined"
                      startIcon={<Add />}
                      onClick={() => handleOpenDialog()}
                      sx={{ mt: 2 }}
                    >
                      Crear Primer Dispositivo
                    </Button>
                  </Box>
                </TableCell>
              </TableRow>
            ) : (
              devices.map((device) => (
                <TableRow key={device.id} hover>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {getDeviceIcon(device.type)}
                      {device.code}
                    </Box>
                  </TableCell>
                  <TableCell>{getDeviceTypeLabel(device.type)}</TableCell>
                  <TableCell>
                    <Chip
                      label={getStatusLabel(device.status)}
                      size="small"
                      color={
                        device.status === 'available' ? 'success' :
                        device.status === 'in_use' ? 'primary' :
                        device.status === 'maintenance' ? 'warning' : 'default'
                      }
                    />
                  </TableCell>
                  <TableCell>
                    {device.batteryLevel !== undefined ? (
                      <Chip
                        label={`${device.batteryLevel}%`}
                        size="small"
                        color={
                          device.batteryLevel > 70 ? 'success' :
                          device.batteryLevel > 30 ? 'warning' : 'error'
                        }
                      />
                    ) : (
                      '-'
                    )}
                  </TableCell>
                  <TableCell>{device.stationName || 'Sin asignar'}</TableCell>
                  <TableCell>{device.companyName}</TableCell>
                  <TableCell align="right">
                    <IconButton
                      size="small"
                      onClick={() => handleOpenDialog(device)}
                      color="primary"
                    >
                      <Edit />
                    </IconButton>
                    <IconButton
                      size="small"
                      onClick={() => handleDelete(device.id)}
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

      <Dialog open={openDialog} onClose={handleCloseDialog} maxWidth="sm" fullWidth>
        <DialogTitle>
          {selectedDevice ? 'Editar Dispositivo' : 'Nuevo Dispositivo'}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 2 }}>
            <TextField
              label="Código"
              value={formData.code}
              onChange={(e) => setFormData({ ...formData, code: e.target.value })}
              fullWidth
              required
              helperText="Identificador único del dispositivo"
            />
            <FormControl fullWidth required>
              <InputLabel>Tipo</InputLabel>
              <Select
                value={formData.type}
                onChange={(e) => setFormData({ ...formData, type: e.target.value as any })}
                label="Tipo"
              >
                <MenuItem value="bike">Bicicleta</MenuItem>
                <MenuItem value="ebike">E-Bike</MenuItem>
                <MenuItem value="scooter">Patinete</MenuItem>
              </Select>
            </FormControl>
            <FormControl fullWidth required>
              <InputLabel>Estado</InputLabel>
              <Select
                value={formData.status}
                onChange={(e) => setFormData({ ...formData, status: e.target.value as any })}
                label="Estado"
              >
                <MenuItem value="available">Disponible</MenuItem>
                <MenuItem value="in_use">En uso</MenuItem>
                <MenuItem value="maintenance">Mantenimiento</MenuItem>
                <MenuItem value="retired">Retirado</MenuItem>
              </Select>
            </FormControl>
            {(formData.type === 'ebike' || formData.type === 'scooter') && (
              <TextField
                label="Nivel de Batería"
                value={formData.batteryLevel}
                onChange={(e) => setFormData({ ...formData, batteryLevel: e.target.value })}
                fullWidth
                type="number"
                inputProps={{ min: 0, max: 100 }}
                helperText="Porcentaje de batería (0-100)"
              />
            )}
            <FormControl fullWidth>
              <InputLabel>Estación</InputLabel>
              <Select
                value={formData.stationId}
                onChange={(e) => setFormData({ ...formData, stationId: e.target.value })}
                label="Estación"
              >
                <MenuItem value="">
                  <em>Sin asignar</em>
                </MenuItem>
                {stations.map((station) => (
                  <MenuItem key={station.id} value={station.id}>
                    {station.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            {!selectedDevice && (
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

export default DevicesPage
