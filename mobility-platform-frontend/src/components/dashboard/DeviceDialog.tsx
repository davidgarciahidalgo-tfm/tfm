import { useState, useEffect } from 'react'
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormHelperText
} from '@mui/material'

export interface Device {
  id: number
  serialNumber: string
  type: 'bike' | 'scooter' | 'ebike'
  model: string
  stationId: number
  stationName: string
  status: 'available' | 'in_use' | 'maintenance' | 'retired'
  batteryLevel?: number
}

interface DeviceDialogProps {
  open: boolean
  device: Device | null
  onClose: () => void
  onSave: (device: Omit<Device, 'id' | 'stationName'> | Device) => void
  stations: Array<{ id: number; name: string }>
}

const DeviceDialog = ({ open, device, onClose, onSave, stations }: DeviceDialogProps) => {
  const [formData, setFormData] = useState<Omit<Device, 'id' | 'stationName'>>({
    serialNumber: '',
    type: 'bike',
    model: '',
    stationId: 0,
    status: 'available',
    batteryLevel: 100
  })

  const [errors, setErrors] = useState<Record<string, string>>({})

  useEffect(() => {
    if (device) {
      setFormData({
        serialNumber: device.serialNumber,
        type: device.type,
        model: device.model,
        stationId: device.stationId,
        status: device.status,
        batteryLevel: device.batteryLevel
      })
    } else {
      setFormData({
        serialNumber: '',
        type: 'bike',
        model: '',
        stationId: stations.length > 0 ? stations[0].id : 0,
        status: 'available',
        batteryLevel: 100
      })
    }
    setErrors({})
  }, [device, open, stations])

  const validate = () => {
    const newErrors: Record<string, string> = {}

    if (!formData.serialNumber.trim()) {
      newErrors.serialNumber = 'El número de serie es obligatorio'
    }
    if (!formData.model.trim()) {
      newErrors.model = 'El modelo es obligatorio'
    }
    if (formData.stationId === 0) {
      newErrors.stationId = 'Debe seleccionar una estación'
    }
    if (formData.batteryLevel !== undefined && (formData.batteryLevel < 0 || formData.batteryLevel > 100)) {
      newErrors.batteryLevel = 'El nivel de batería debe estar entre 0 y 100'
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = () => {
    if (validate()) {
      if (device) {
        onSave({ ...formData, id: device.id, stationName: '' })
      } else {
        onSave(formData)
      }
      onClose()
    }
  }

  const handleChange = (field: keyof Omit<Device, 'id' | 'stationName'>, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }))
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }))
    }
  }

  const requiresBattery = formData.type === 'ebike' || formData.type === 'scooter'

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        {device ? 'Editar Dispositivo' : 'Nuevo Dispositivo'}
      </DialogTitle>
      <DialogContent>
        <Grid container spacing={2} sx={{ mt: 1 }}>
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Número de Serie"
              value={formData.serialNumber}
              onChange={(e) => handleChange('serialNumber', e.target.value)}
              error={!!errors.serialNumber}
              helperText={errors.serialNumber}
              required
            />
          </Grid>

          <Grid item xs={12} md={6}>
            <FormControl fullWidth error={!!errors.type}>
              <InputLabel>Tipo</InputLabel>
              <Select
                value={formData.type}
                label="Tipo"
                onChange={(e) => handleChange('type', e.target.value)}
              >
                <MenuItem value="bike">Bicicleta</MenuItem>
                <MenuItem value="ebike">Bicicleta Eléctrica</MenuItem>
                <MenuItem value="scooter">Patinete</MenuItem>
              </Select>
              {errors.type && <FormHelperText>{errors.type}</FormHelperText>}
            </FormControl>
          </Grid>

          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Modelo"
              value={formData.model}
              onChange={(e) => handleChange('model', e.target.value)}
              error={!!errors.model}
              helperText={errors.model}
              required
            />
          </Grid>

          <Grid item xs={12} md={6}>
            <FormControl fullWidth error={!!errors.status}>
              <InputLabel>Estado</InputLabel>
              <Select
                value={formData.status}
                label="Estado"
                onChange={(e) => handleChange('status', e.target.value)}
              >
                <MenuItem value="available">Disponible</MenuItem>
                <MenuItem value="in_use">En Uso</MenuItem>
                <MenuItem value="maintenance">Mantenimiento</MenuItem>
                <MenuItem value="retired">Retirado</MenuItem>
              </Select>
              {errors.status && <FormHelperText>{errors.status}</FormHelperText>}
            </FormControl>
          </Grid>

          <Grid item xs={12} md={requiresBattery ? 6 : 12}>
            <FormControl fullWidth error={!!errors.stationId}>
              <InputLabel>Estación</InputLabel>
              <Select
                value={formData.stationId}
                label="Estación"
                onChange={(e) => handleChange('stationId', e.target.value)}
              >
                {stations.map((station) => (
                  <MenuItem key={station.id} value={station.id}>
                    {station.name}
                  </MenuItem>
                ))}
              </Select>
              {errors.stationId && <FormHelperText>{errors.stationId}</FormHelperText>}
            </FormControl>
          </Grid>

          {requiresBattery && (
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Nivel de Batería (%)"
                type="number"
                value={formData.batteryLevel}
                onChange={(e) => handleChange('batteryLevel', parseInt(e.target.value))}
                error={!!errors.batteryLevel}
                helperText={errors.batteryLevel}
                inputProps={{ min: 0, max: 100 }}
              />
            </Grid>
          )}
        </Grid>
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2 }}>
        <Button onClick={onClose} color="inherit">
          Cancelar
        </Button>
        <Button onClick={handleSubmit} variant="contained">
          {device ? 'Guardar Cambios' : 'Crear Dispositivo'}
        </Button>
      </DialogActions>
    </Dialog>
  )
}

export default DeviceDialog
