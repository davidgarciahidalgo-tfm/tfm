import { supabase } from '../config/supabase'

export interface StationData {
  id: string
  name: string
  address: string
  latitude: number
  longitude: number
  capacity: number
  status: 'active' | 'inactive' | 'maintenance'
  companyId: string
  companyName?: string
  createdAt: string
  updatedAt: string
  availableDevices?: number
}

export interface CreateStationInput {
  name: string
  address: string
  latitude: number
  longitude: number
  capacity: number
  status?: 'active' | 'inactive' | 'maintenance'
  companyId: string
}

export interface UpdateStationInput {
  name?: string
  address?: string
  latitude?: number
  longitude?: number
  capacity?: number
  status?: 'active' | 'inactive' | 'maintenance'
}

export const stationService = {
  /**
   * Obtener todas las estaciones
   */
  async getStations(): Promise<StationData[]> {
    try {
      const { data: stations, error } = await supabase
        .from('stations')
        .select(`
          *,
          companies (
            id,
            name
          )
        `)
        .order('name', { ascending: true })

      if (error) throw error

      // Obtener el conteo de dispositivos disponibles por estación
      const { data: deviceCounts } = await supabase
        .from('devices')
        .select('station_id')
        .eq('status', 'available')

      const countsByStation: Record<string, number> = {}
      deviceCounts?.forEach(d => {
        if (d.station_id) {
          countsByStation[d.station_id] = (countsByStation[d.station_id] || 0) + 1
        }
      })

      return (stations || []).map((station: any) => ({
        id: station.id,
        name: station.name,
        address: station.address,
        latitude: parseFloat(station.latitude),
        longitude: parseFloat(station.longitude),
        capacity: station.capacity,
        status: station.status,
        companyId: station.company_id,
        companyName: station.companies?.name || undefined,
        createdAt: station.created_at,
        updatedAt: station.updated_at,
        availableDevices: countsByStation[station.id] || 0,
      }))
    } catch (error: any) {
      console.error('Get stations error:', error)
      throw new Error('Error al obtener estaciones')
    }
  },

  /**
   * Obtener estación por ID
   */
  async getStationById(id: string): Promise<StationData> {
    try {
      const { data, error } = await supabase
        .from('stations')
        .select(`
          *,
          companies (
            id,
            name
          )
        `)
        .eq('id', id)
        .single()

      if (error) throw error
      if (!data) throw new Error('Estación no encontrada')

      // Contar dispositivos disponibles
      const { count } = await supabase
        .from('devices')
        .select('*', { count: 'exact', head: true })
        .eq('station_id', id)
        .eq('status', 'available')

      return {
        id: data.id,
        name: data.name,
        address: data.address,
        latitude: parseFloat(data.latitude),
        longitude: parseFloat(data.longitude),
        capacity: data.capacity,
        status: data.status,
        companyId: data.company_id,
        companyName: (data.companies as any)?.name || undefined,
        createdAt: data.created_at,
        updatedAt: data.updated_at,
        availableDevices: count || 0,
      }
    } catch (error: any) {
      console.error('Get station by id error:', error)
      throw new Error('Error al obtener estación')
    }
  },

  /**
   * Crear nueva estación
   */
  async createStation(input: CreateStationInput): Promise<StationData> {
    try {
      const { data, error } = await supabase
        .from('stations')
        .insert({
          name: input.name,
          address: input.address,
          latitude: input.latitude,
          longitude: input.longitude,
          capacity: input.capacity,
          status: input.status || 'active',
          company_id: input.companyId,
        })
        .select(`
          *,
          companies (
            id,
            name
          )
        `)
        .single()

      if (error) throw error
      if (!data) throw new Error('Error al crear estación')

      return {
        id: data.id,
        name: data.name,
        address: data.address,
        latitude: parseFloat(data.latitude),
        longitude: parseFloat(data.longitude),
        capacity: data.capacity,
        status: data.status,
        companyId: data.company_id,
        companyName: (data.companies as any)?.name || undefined,
        createdAt: data.created_at,
        updatedAt: data.updated_at,
        availableDevices: 0,
      }
    } catch (error: any) {
      console.error('Create station error:', error)
      throw new Error('Error al crear estación')
    }
  },

  /**
   * Actualizar estación
   */
  async updateStation(id: string, input: UpdateStationInput): Promise<StationData> {
    try {
      const updateData: any = {}

      if (input.name) updateData.name = input.name
      if (input.address) updateData.address = input.address
      if (input.latitude !== undefined) updateData.latitude = input.latitude
      if (input.longitude !== undefined) updateData.longitude = input.longitude
      if (input.capacity !== undefined) updateData.capacity = input.capacity
      if (input.status) updateData.status = input.status

      const { data, error } = await supabase
        .from('stations')
        .update(updateData)
        .eq('id', id)
        .select(`
          *,
          companies (
            id,
            name
          )
        `)
        .single()

      if (error) throw error
      if (!data) throw new Error('Estación no encontrada')

      // Contar dispositivos disponibles
      const { count } = await supabase
        .from('devices')
        .select('*', { count: 'exact', head: true })
        .eq('station_id', id)
        .eq('status', 'available')

      return {
        id: data.id,
        name: data.name,
        address: data.address,
        latitude: parseFloat(data.latitude),
        longitude: parseFloat(data.longitude),
        capacity: data.capacity,
        status: data.status,
        companyId: data.company_id,
        companyName: (data.companies as any)?.name || undefined,
        createdAt: data.created_at,
        updatedAt: data.updated_at,
        availableDevices: count || 0,
      }
    } catch (error: any) {
      console.error('Update station error:', error)
      throw new Error('Error al actualizar estación')
    }
  },

  /**
   * Eliminar estación
   */
  async deleteStation(id: string): Promise<void> {
    try {
      const { error } = await supabase
        .from('stations')
        .delete()
        .eq('id', id)

      if (error) throw error
    } catch (error: any) {
      console.error('Delete station error:', error)
      throw new Error('Error al eliminar estación')
    }
  },
}
