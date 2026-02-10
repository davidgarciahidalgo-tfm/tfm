import { supabase } from '../config/supabase'

export interface RoleData {
  id: string
  name: string
  permissions: string[]
}

export const roleService = {
  /**
   * Obtener todos los roles
   */
  async getRoles(): Promise<RoleData[]> {
    try {
      const { data, error } = await supabase
        .from('roles')
        .select('*')
        .order('name', { ascending: true })

      if (error) throw error

      return (data || []).map((role) => ({
        id: role.id,
        name: role.name,
        permissions: Array.isArray(role.permissions) ? role.permissions : [],
      }))
    } catch (error: any) {
      console.error('Get roles error:', error)
      throw new Error('Error al obtener roles')
    }
  },

  /**
   * Obtener rol por ID
   */
  async getRoleById(id: string): Promise<RoleData> {
    try {
      const { data, error } = await supabase
        .from('roles')
        .select('*')
        .eq('id', id)
        .single()

      if (error) throw error
      if (!data) throw new Error('Rol no encontrado')

      return {
        id: data.id,
        name: data.name,
        permissions: Array.isArray(data.permissions) ? data.permissions : [],
      }
    } catch (error: any) {
      console.error('Get role by id error:', error)
      throw new Error('Error al obtener rol')
    }
  },
}
