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

export interface Station {
  id: number
  name: string
  address: string
  latitude: number
  longitude: number
  capacity: number
  status: 'active' | 'inactive' | 'maintenance'
}

interface StationDialogProps {
  open: boolean
  station: Station | null
  onClose: () => void
  onSave: (station: Omit<Station, 'id'> | Station) => void
}

const StationDialog = ({ open, station, onClose, onSave }: StationDialogProps) => {
  const [formData, setFormData] = useState<Omit<Station, 'id'>>({
    name: '',
    address: '',
    latitude: 40.4168,
    longitude: -3.7038,
    capacity: 20,
    status: 'active'
  })

  const [errors, setErrors] = useState<Record<string, string>>({})

  useEffect(() => {
    if (station) {
      setFormData({
        name: station.name,
        address: station.address,
        latitude: station.latitude,
        longitude: station.longitude,
        capacity: station.capacity,
        status: station.status
      })
    } else {
      setFormData({
        name: '',
        address: '',
        latitude: 40.4168,
        longitude: -3.7038,
        capacity: 20,
        status: 'active'
      })
    }
    setErrors({})
  }, [station, open])

  const validate = () => {
    const newErrors: Record<string, string> = {}

    if (!formData.name.trim()) {
      newErrors.name = 'El nombre es obligatorio'
    }
    if (!formData.address.trim()) {
      newErrors.address = 'La dirección es obligatoria'
    }
    if (formData.latitude < -90 || formData.latitude > 90) {
      newErrors.latitude = 'Latitud debe estar entre -90 y 90'
    }
    if (formData.longitude < -180 || formData.longitude > 180) {
      newErrors.longitude = 'Longitud debe estar entre -180 y 180'
    }
    if (formData.capacity < 1) {
      newErrors.capacity = 'La capacidad debe ser mayor a 0'
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = () => {
    if (validate()) {
      if (station) {
        onSave({ ...formData, id: station.id })
      } else {
        onSave(formData)
      }
      onClose()
    }
  }

  const handleChange = (field: keyof Omit<Station, 'id'>, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }))
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }))
    }
  }

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        {station ? 'Editar Estación' : 'Nueva Estación'}
      </DialogTitle>
      <DialogContent>
        <Grid container spacing={2} sx={{ mt: 1 }}>
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Nombre de la Estación"
              value={formData.name}
              onChange={(e) => handleChange('name', e.target.value)}
              error={!!errors.name}
              helperText={errors.name}
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
                <MenuItem value="active">Activa</MenuItem>
                <MenuItem value="inactive">Inactiva</MenuItem>
                <MenuItem value="maintenance">Mantenimiento</MenuItem>
              </Select>
              {errors.status && <FormHelperText>{errors.status}</FormHelperText>}
            </FormControl>
          </Grid>

          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Dirección"
              value={formData.address}
              onChange={(e) => handleChange('address', e.target.value)}
              error={!!errors.address}
              helperText={errors.address}
              required
            />
          </Grid>

          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              label="Latitud"
              type="number"
              value={formData.latitude}
              onChange={(e) => handleChange('latitude', parseFloat(e.target.value))}
              error={!!errors.latitude}
              helperText={errors.latitude}
              inputProps={{ step: 0.0001 }}
              required
            />
          </Grid>

          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              label="Longitud"
              type="number"
              value={formData.longitude}
              onChange={(e) => handleChange('longitude', parseFloat(e.target.value))}
              error={!!errors.longitude}
              helperText={errors.longitude}
              inputProps={{ step: 0.0001 }}
              required
            />
          </Grid>

          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              label="Capacidad"
              type="number"
              value={formData.capacity}
              onChange={(e) => handleChange('capacity', parseInt(e.target.value))}
              error={!!errors.capacity}
              helperText={errors.capacity}
              inputProps={{ min: 1 }}
              required
            />
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2 }}>
        <Button onClick={onClose} color="inherit">
          Cancelar
        </Button>
        <Button onClick={handleSubmit} variant="contained">
          {station ? 'Guardar Cambios' : 'Crear Estación'}
        </Button>
      </DialogActions>
    </Dialog>
  )
}

export default StationDialog
