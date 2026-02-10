import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables')
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    persistSession: true,
    autoRefreshToken: true,
  },
})

// Types helper para mejor tipado con la base de datos
export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export interface Database {
  public: {
    Tables: {
      companies: {
        Row: {
          id: string
          name: string
          slug: string
          logo_url: string | null
          primary_color: string
          secondary_color: string
          locale: string
          timezone: string
          measurement_unit: 'metric' | 'imperial'
          is_active: boolean
          created_at: string
          updated_at: string
        }
        Insert: Omit<Database['public']['Tables']['companies']['Row'], 'id' | 'created_at' | 'updated_at'>
        Update: Partial<Database['public']['Tables']['companies']['Insert']>
      }
      roles: {
        Row: {
          id: string
          company_id: string | null
          name: string
          description: string | null
          permissions: Json
          is_system_role: boolean
          created_at: string
        }
        Insert: Omit<Database['public']['Tables']['roles']['Row'], 'id' | 'created_at'>
        Update: Partial<Database['public']['Tables']['roles']['Insert']>
      }
      users: {
        Row: {
          id: string
          company_id: string | null
          email: string
          username: string
          password_hash: string
          first_name: string
          last_name: string
          role_id: string | null
          is_active: boolean
          last_login_at: string | null
          created_at: string
          updated_at: string
        }
        Insert: Omit<Database['public']['Tables']['users']['Row'], 'id' | 'created_at' | 'updated_at'>
        Update: Partial<Database['public']['Tables']['users']['Insert']>
      }
      stations: {
        Row: {
          id: string
          company_id: string
          name: string
          code: string
          latitude: number
          longitude: number
          address: string | null
          capacity: number
          is_active: boolean
          created_at: string
          updated_at: string
        }
        Insert: Omit<Database['public']['Tables']['stations']['Row'], 'id' | 'created_at' | 'updated_at'>
        Update: Partial<Database['public']['Tables']['stations']['Insert']>
      }
      device_types: {
        Row: {
          id: string
          company_id: string
          name: string
          code: string
          icon: string | null
          color: string
          has_battery: boolean
          created_at: string
        }
        Insert: Omit<Database['public']['Tables']['device_types']['Row'], 'id' | 'created_at'>
        Update: Partial<Database['public']['Tables']['device_types']['Insert']>
      }
      devices: {
        Row: {
          id: string
          company_id: string
          device_type_id: string | null
          code: string
          current_station_id: string | null
          current_latitude: number | null
          current_longitude: number | null
          battery_level: number | null
          status: 'available' | 'in_use' | 'maintenance' | 'lost'
          last_maintenance_at: string | null
          created_at: string
          updated_at: string
        }
        Insert: Omit<Database['public']['Tables']['devices']['Row'], 'id' | 'created_at' | 'updated_at'>
        Update: Partial<Database['public']['Tables']['devices']['Insert']>
      }
    }
  }
}
