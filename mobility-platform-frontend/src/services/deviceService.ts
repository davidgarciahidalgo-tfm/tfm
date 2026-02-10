import { supabase } from '../config/supabase'

export interface DeviceData {
  id: string
  code: string
  type: 'bike' | 'ebike' | 'scooter'
  status: 'available' | 'in_use' | 'maintenance' | 'retired'
  batteryLevel?: number
  stationId?: string
  stationName?: string
  companyId: string
  companyName?: string
  createdAt: string
  updatedAt: string
}

export interface CreateDeviceInput {
  code: string
  type: 'bike' | 'ebike' | 'scooter'
  status?: 'available' | 'in_use' | 'maintenance' | 'retired'
  batteryLevel?: number
  stationId?: string
  companyId: string
}

export interface UpdateDeviceInput {
  code?: string
  type?: 'bike' | 'ebike' | 'scooter'
  status?: 'available' | 'in_use' | 'maintenance' | 'retired'
  batteryLevel?: number
  stationId?: string
}

export const deviceService = {
  /**
   * Obtener todos los dispositivos
   */
  async getDevices(): Promise<DeviceData[]> {
    try {
      const { data, error } = await supabase
        .from('devices')
        .select(`
          *,
          stations (
            id,
            name
          ),
          companies (
            id,
            name
          )
        `)
        .order('code', { ascending: true })

      if (error) throw error

      return (data || []).map((device: any) => ({
        id: device.id,
        code: device.code,
        type: device.type,
        status: device.status,
        batteryLevel: device.battery_level || undefined,
        stationId: device.station_id || undefined,
        stationName: device.stations?.name || undefined,
        companyId: device.company_id,
        companyName: device.companies?.name || undefined,
        createdAt: device.created_at,
        updatedAt: device.updated_at,
      }))
    } catch (error: any) {
      console.error('Get devices error:', error)
      throw new Error('Error al obtener dispositivos')
    }
  },

  /**
   * Obtener dispositivo por ID
   */
  async getDeviceById(id: string): Promise<DeviceData> {
    try {
      const { data, error } = await supabase
        .from('devices')
        .select(`
          *,
          stations (
            id,
            name
          ),
          companies (
            id,
            name
          )
        `)
        .eq('id', id)
        .single()

      if (error) throw error
      if (!data) throw new Error('Dispositivo no encontrado')

      return {
        id: data.id,
        code: data.code,
        type: data.type,
        status: data.status,
        batteryLevel: data.battery_level || undefined,
        stationId: data.station_id || undefined,
        stationName: (data.stations as any)?.name || undefined,
        companyId: data.company_id,
        companyName: (data.companies as any)?.name || undefined,
        createdAt: data.created_at,
        updatedAt: data.updated_at,
      }
    } catch (error: any) {
      console.error('Get device by id error:', error)
      throw new Error('Error al obtener dispositivo')
    }
  },

  /**
   * Crear nuevo dispositivo
   */
  async createDevice(input: CreateDeviceInput): Promise<DeviceData> {
    try {
      const { data, error } = await supabase
        .from('devices')
        .insert({
          code: input.code,
          type: input.type,
          status: input.status || 'available',
          battery_level: input.batteryLevel || null,
          station_id: input.stationId || null,
          company_id: input.companyId,
        })
        .select(`
          *,
          stations (
            id,
            name
          ),
          companies (
            id,
            name
          )
        `)
        .single()

      if (error) throw error
      if (!data) throw new Error('Error al crear dispositivo')

      return {
        id: data.id,
        code: data.code,
        type: data.type,
        status: data.status,
        batteryLevel: data.battery_level || undefined,
        stationId: data.station_id || undefined,
        stationName: (data.stations as any)?.name || undefined,
        companyId: data.company_id,
        companyName: (data.companies as any)?.name || undefined,
        createdAt: data.created_at,
        updatedAt: data.updated_at,
      }
    } catch (error: any) {
      console.error('Create device error:', error)
      
      if (error.message?.includes('unique') || error.code === '23505') {
        throw new Error('El código ya está en uso')
      }
      
      throw new Error('Error al crear dispositivo')
    }
  },

  /**
   * Actualizar dispositivo
   */
  async updateDevice(id: string, input: UpdateDeviceInput): Promise<DeviceData> {
    try {
      const updateData: any = {}

      if (input.code) updateData.code = input.code
      if (input.type) updateData.type = input.type
      if (input.status) updateData.status = input.status
      if (input.batteryLevel !== undefined) updateData.battery_level = input.batteryLevel || null
      if (input.stationId !== undefined) updateData.station_id = input.stationId || null

      const { data, error } = await supabase
        .from('devices')
        .update(updateData)
        .eq('id', id)
        .select(`
          *,
          stations (
            id,
            name
          ),
          companies (
            id,
            name
          )
        `)
        .single()

      if (error) throw error
      if (!data) throw new Error('Dispositivo no encontrado')

      return {
        id: data.id,
        code: data.code,
        type: data.type,
        status: data.status,
        batteryLevel: data.battery_level || undefined,
        stationId: data.station_id || undefined,
        stationName: (data.stations as any)?.name || undefined,
        companyId: data.company_id,
        companyName: (data.companies as any)?.name || undefined,
        createdAt: data.created_at,
        updatedAt: data.updated_at,
      }
    } catch (error: any) {
      console.error('Update device error:', error)
      
      if (error.message?.includes('unique') || error.code === '23505') {
        throw new Error('El código ya está en uso')
      }
      
      throw new Error('Error al actualizar dispositivo')
    }
  },

  /**
   * Eliminar dispositivo
   */
  async deleteDevice(id: string): Promise<void> {
    try {
      const { error } = await supabase
        .from('devices')
        .delete()
        .eq('id', id)

      if (error) throw error
    } catch (error: any) {
      console.error('Delete device error:', error)
      throw new Error('Error al eliminar dispositivo')
    }
  },
}
