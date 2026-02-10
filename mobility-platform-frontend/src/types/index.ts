// Tipos principales de la aplicación

export interface Company {
  id: string
  name: string
  slug: string
  logoUrl?: string
  primaryColor: string
  secondaryColor: string
  locale: string
  timezone: string
  measurementUnit: 'metric' | 'imperial'
  isActive: boolean
  createdAt: string
  updatedAt: string
}

export interface User {
  id: string
  companyId?: string // Opcional para superadmin
  email: string
  username: string
  firstName: string
  lastName: string
  roleId: string
  role: Role
  isActive: boolean
  lastLoginAt?: string
  createdAt: string
  updatedAt: string
}

export interface Role {
  id: string
  companyId?: string
  name: string
  description?: string
  permissions: string[]
  isSystemRole: boolean
  createdAt: string
}

export interface Station {
  id: string
  companyId: string
  name: string
  code: string
  location: {
    type: 'Point'
    coordinates: [number, number] // [longitude, latitude]
  }
  address?: string
  capacity: number
  isActive: boolean
  createdAt: string
  updatedAt: string
}

export interface DeviceType {
  id: string
  companyId: string
  name: string
  code: string // 'bike', 'scooter', 'e-moto'
  icon: string
  color: string
  hasBattery: boolean
  createdAt: string
}

export interface Device {
  id: string
  companyId: string
  deviceTypeId: string
  deviceType: DeviceType
  code: string
  currentStationId?: string
  currentLocation: {
    type: 'Point'
    coordinates: [number, number]
  }
  batteryLevel?: number
  status: 'available' | 'in_use' | 'maintenance' | 'lost'
  lastMaintenanceAt?: string
  createdAt: string
  updatedAt: string
}

export interface DailySnapshot {
  id: string
  snapshotDate: string // YYYY-MM-DD
  companyId: string
  stationId: string
  station: Station
  deviceTypeId: string
  deviceType: DeviceType
  totalDevices: number
  availableDevices: number
  inUseDevices: number
  maintenanceDevices: number
  occupancyRate: number
  createdAt: string
}

export interface VARPrediction {
  id: string
  stationId: string
  station: Station
  deviceTypeId: string
  deviceType: DeviceType
  predictionDate: string // YYYY-MM-DD
  predictedDemand: number
  confidenceIntervalLower: number
  confidenceIntervalUpper: number
  modelVersion: string
  createdAt: string
}

export interface ExportJob {
  id: string
  companyId: string
  createdBy: string
  exportType: 'predictions' | 'historical' | 'devices'
  parameters: Record<string, any>
  fileUrl?: string
  format: 'pdf' | 'csv' | 'excel'
  status: 'pending' | 'processing' | 'completed' | 'failed'
  errorMessage?: string
  createdAt: string
  completedAt?: string
}

// API Response types
export interface ApiResponse<T> {
  success: boolean
  data: T
  message?: string
}

export interface PaginatedResponse<T> {
  data: T[]
  pagination: {
    page: number
    perPage: number
    total: number
    totalPages: number
  }
}

// Map types
export interface MapBounds {
  north: number
  south: number
  east: number
  west: number
}

export interface MapViewState {
  longitude: number
  latitude: number
  zoom: number
  pitch?: number
  bearing?: number
}
