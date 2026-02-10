import { supabase } from '../config/supabase'

export interface CompanyData {
  id: string
  name: string
  nif: string
  address?: string
  phone?: string
  email?: string
  isActive: boolean
  createdAt: string
  updatedAt: string
}

export interface CreateCompanyInput {
  name: string
  nif: string
  address?: string
  phone?: string
  email?: string
}

export interface UpdateCompanyInput {
  name?: string
  nif?: string
  address?: string
  phone?: string
  email?: string
  isActive?: boolean
}

export const companyService = {
  /**
   * Obtener todas las empresas
   */
  async getCompanies(): Promise<CompanyData[]> {
    try {
      const { data, error } = await supabase
        .from('companies')
        .select('*')
        .order('name', { ascending: true })

      if (error) throw error

      return (data || []).map((company) => ({
        id: company.id,
        name: company.name,
        nif: company.nif,
        address: company.address || undefined,
        phone: company.phone || undefined,
        email: company.email || undefined,
        isActive: company.is_active,
        createdAt: company.created_at,
        updatedAt: company.updated_at,
      }))
    } catch (error: any) {
      console.error('Get companies error:', error)
      throw new Error('Error al obtener empresas')
    }
  },

  /**
   * Obtener empresa por ID
   */
  async getCompanyById(id: string): Promise<CompanyData> {
    try {
      const { data, error } = await supabase
        .from('companies')
        .select('*')
        .eq('id', id)
        .single()

      if (error) throw error
      if (!data) throw new Error('Empresa no encontrada')

      return {
        id: data.id,
        name: data.name,
        nif: data.nif,
        address: data.address || undefined,
        phone: data.phone || undefined,
        email: data.email || undefined,
        isActive: data.is_active,
        createdAt: data.created_at,
        updatedAt: data.updated_at,
      }
    } catch (error: any) {
      console.error('Get company by id error:', error)
      throw new Error('Error al obtener empresa')
    }
  },

  /**
   * Crear nueva empresa
   */
  async createCompany(input: CreateCompanyInput): Promise<CompanyData> {
    try {
      const { data, error } = await supabase
        .from('companies')
        .insert({
          name: input.name,
          nif: input.nif,
          address: input.address || null,
          phone: input.phone || null,
          email: input.email || null,
          is_active: true,
        })
        .select('*')
        .single()

      if (error) throw error
      if (!data) throw new Error('Error al crear empresa')

      return {
        id: data.id,
        name: data.name,
        nif: data.nif,
        address: data.address || undefined,
        phone: data.phone || undefined,
        email: data.email || undefined,
        isActive: data.is_active,
        createdAt: data.created_at,
        updatedAt: data.updated_at,
      }
    } catch (error: any) {
      console.error('Create company error:', error)
      
      if (error.message?.includes('unique') || error.code === '23505') {
        throw new Error('El NIF ya está registrado')
      }
      
      throw new Error('Error al crear empresa')
    }
  },

  /**
   * Actualizar empresa
   */
  async updateCompany(id: string, input: UpdateCompanyInput): Promise<CompanyData> {
    try {
      const updateData: any = {}

      if (input.name) updateData.name = input.name
      if (input.nif) updateData.nif = input.nif
      if (input.address !== undefined) updateData.address = input.address || null
      if (input.phone !== undefined) updateData.phone = input.phone || null
      if (input.email !== undefined) updateData.email = input.email || null
      if (input.isActive !== undefined) updateData.is_active = input.isActive

      const { data, error } = await supabase
        .from('companies')
        .update(updateData)
        .eq('id', id)
        .select('*')
        .single()

      if (error) throw error
      if (!data) throw new Error('Empresa no encontrada')

      return {
        id: data.id,
        name: data.name,
        nif: data.nif,
        address: data.address || undefined,
        phone: data.phone || undefined,
        email: data.email || undefined,
        isActive: data.is_active,
        createdAt: data.created_at,
        updatedAt: data.updated_at,
      }
    } catch (error: any) {
      console.error('Update company error:', error)
      
      if (error.message?.includes('unique') || error.code === '23505') {
        throw new Error('El NIF ya está registrado')
      }
      
      throw new Error('Error al actualizar empresa')
    }
  },

  /**
   * Eliminar empresa
   */
  async deleteCompany(id: string): Promise<void> {
    try {
      const { error } = await supabase
        .from('companies')
        .delete()
        .eq('id', id)

      if (error) throw error
    } catch (error: any) {
      console.error('Delete company error:', error)
      throw new Error('Error al eliminar empresa')
    }
  },
}
