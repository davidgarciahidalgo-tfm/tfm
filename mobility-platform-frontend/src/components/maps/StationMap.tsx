import { useEffect } from 'react'
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet'
import { Box, Chip, Typography } from '@mui/material'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

// Fix for default marker icons in React-Leaflet
import icon from 'leaflet/dist/images/marker-icon.png'
import iconShadow from 'leaflet/dist/images/marker-shadow.png'

const DefaultIcon = L.icon({
  iconUrl: icon,
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
})

L.Marker.prototype.options.icon = DefaultIcon

interface Station {
  id: number
  name: string
  latitude: number
  longitude: number
  capacity: number
  availableBikes?: number
  status: 'active' | 'inactive' | 'maintenance'
}

interface StationMapProps {
  stations?: Station[]
  center?: [number, number]
  zoom?: number
}

// Componente para ajustar el zoom cuando cambian las estaciones
const MapBounds = ({ stations }: { stations: Station[] }) => {
  const map = useMap()

  useEffect(() => {
    if (stations.length > 0) {
      const bounds = L.latLngBounds(
        stations.map(station => [station.latitude, station.longitude])
      )
      map.fitBounds(bounds, { padding: [50, 50] })
    }
  }, [stations, map])

  return null
}

const StationMap = ({ 
  stations = [], 
  center = [40.4168, -3.7038], // Madrid por defecto
  zoom = 13 
}: StationMapProps) => {
  // Datos de ejemplo de estaciones en Madrid
  const defaultStations: Station[] = [
    {
      id: 1,
      name: 'Estación Atocha',
      latitude: 40.4068,
      longitude: -3.6923,
      capacity: 30,
      availableBikes: 15,
      status: 'active'
    },
    {
      id: 2,
      name: 'Estación Retiro',
      latitude: 40.4153,
      longitude: -3.6838,
      capacity: 25,
      availableBikes: 20,
      status: 'active'
    },
    {
      id: 3,
      name: 'Estación Sol',
      latitude: 40.4169,
      longitude: -3.7035,
      capacity: 40,
      availableBikes: 8,
      status: 'active'
    },
    {
      id: 4,
      name: 'Estación Gran Vía',
      latitude: 40.4201,
      longitude: -3.7034,
      capacity: 35,
      availableBikes: 25,
      status: 'active'
    },
    {
      id: 5,
      name: 'Estación Cibeles',
      latitude: 40.4189,
      longitude: -3.6936,
      capacity: 30,
      availableBikes: 0,
      status: 'maintenance'
    },
    {
      id: 6,
      name: 'Estación Chamberí',
      latitude: 40.4318,
      longitude: -3.7027,
      capacity: 20,
      availableBikes: 18,
      status: 'active'
    }
  ]

  const displayStations = stations.length > 0 ? stations : defaultStations

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'success'
      case 'maintenance':
        return 'warning'
      case 'inactive':
        return 'error'
      default:
        return 'default'
    }
  }

  const getStatusText = (status: string) => {
    switch (status) {
      case 'active':
        return 'Activa'
      case 'maintenance':
        return 'Mantenimiento'
      case 'inactive':
        return 'Inactiva'
      default:
        return 'Desconocido'
    }
  }

  // Crear icono personalizado basado en el estado
  const createCustomIcon = (status: string) => {
    const color = status === 'active' ? '#4caf50' : status === 'maintenance' ? '#ff9800' : '#f44336'
    
    const svgIcon = `
      <svg width="25" height="41" viewBox="0 0 25 41" xmlns="http://www.w3.org/2000/svg">
        <path d="M12.5 0C5.6 0 0 5.6 0 12.5c0 8.4 12.5 28.5 12.5 28.5S25 20.9 25 12.5C25 5.6 19.4 0 12.5 0z" 
              fill="${color}" stroke="#fff" stroke-width="2"/>
        <circle cx="12.5" cy="12.5" r="6" fill="#fff"/>
      </svg>
    `
    
    return L.divIcon({
      html: svgIcon,
      iconSize: [25, 41],
      iconAnchor: [12, 41],
      popupAnchor: [1, -34],
      className: 'custom-marker-icon'
    })
  }

  return (
    <Box sx={{ height: '100%', width: '100%', position: 'relative' }}>
      <MapContainer
        center={center}
        zoom={zoom}
        style={{ height: '100%', width: '100%', borderRadius: '12px' }}
        scrollWheelZoom={true}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        
        {displayStations.length > 0 && <MapBounds stations={displayStations} />}
        
        {displayStations.map((station) => (
          <Marker
            key={station.id}
            position={[station.latitude, station.longitude]}
            icon={createCustomIcon(station.status)}
          >
            <Popup>
              <Box sx={{ minWidth: 200, p: 1 }}>
                <Typography variant="h6" fontWeight={600} gutterBottom>
                  {station.name}
                </Typography>
                
                <Box sx={{ mb: 1 }}>
                  <Chip
                    label={getStatusText(station.status)}
                    color={getStatusColor(station.status) as any}
                    size="small"
                    sx={{ mb: 1 }}
                  />
                </Box>
                
                <Typography variant="body2" color="text.secondary">
                  <strong>Capacidad:</strong> {station.capacity} bicicletas
                </Typography>
                
                {station.availableBikes !== undefined && (
                  <Typography variant="body2" color="text.secondary">
                    <strong>Disponibles:</strong> {station.availableBikes} bicicletas
                  </Typography>
                )}
                
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  <strong>Ubicación:</strong><br />
                  Lat: {station.latitude.toFixed(4)}<br />
                  Lng: {station.longitude.toFixed(4)}
                </Typography>
              </Box>
            </Popup>
          </Marker>
        ))}
      </MapContainer>
      
      {/* Leyenda */}
      <Box
        sx={{
          position: 'absolute',
          top: 16,
          right: 16,
          backgroundColor: 'white',
          borderRadius: 2,
          p: 2,
          boxShadow: 2,
          zIndex: 1000,
          maxWidth: 200
        }}
      >
        <Typography variant="subtitle2" fontWeight={600} gutterBottom>
          Leyenda
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box sx={{ width: 16, height: 16, backgroundColor: '#4caf50', borderRadius: '50%' }} />
            <Typography variant="caption">Activa</Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box sx={{ width: 16, height: 16, backgroundColor: '#ff9800', borderRadius: '50%' }} />
            <Typography variant="caption">Mantenimiento</Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box sx={{ width: 16, height: 16, backgroundColor: '#f44336', borderRadius: '50%' }} />
            <Typography variant="caption">Inactiva</Typography>
          </Box>
        </Box>
      </Box>
    </Box>
  )
}

export default StationMap
