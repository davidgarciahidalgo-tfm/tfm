import { supabase } from '../config/supabase'
import bcrypt from 'bcryptjs'

export interface LoginCredentials {
  usernameOrEmail: string
  password: string
}

export interface AuthUser {
  id: string
  email: string
  username: string
  firstName: string
  lastName: string
  role: string
  companyId?: string
}

export const authService = {
  /**
   * Login con username o email
   */
  async login(credentials: LoginCredentials): Promise<AuthUser> {
    const { usernameOrEmail, password } = credentials

    try {
      console.log('🔐 [AuthService] Iniciando login para:', usernameOrEmail)
      
      // Buscar usuario por username o email
      const { data: users, error: queryError} = await supabase
        .from('users')
        .select(`
          *,
          roles!inner (
            id,
            name,
            permissions
          )
        `)
        .or(`username.eq.${usernameOrEmail},email.eq.${usernameOrEmail}`)
        .eq('is_active', true)
        .single()

      if (queryError) {
        console.error('❌ [AuthService] Error en query:', queryError)
        throw new Error('Credenciales inválidas')
      }

      if (!users) {
        console.error('❌ [AuthService] Usuario no encontrado')
        throw new Error('Credenciales inválidas')
      }

      console.log('✅ [AuthService] Usuario encontrado:', {
        username: users.username,
        email: users.email,
        role: (users.roles as any)?.name,
        is_active: users.is_active
      })

      // Verificar contraseña
      console.log('🔑 [AuthService] Verificando contraseña...')
      const isPasswordValid = await bcrypt.compare(password, users.password_hash)
      
      if (!isPasswordValid) {
        console.error('❌ [AuthService] Contraseña inválida')
        throw new Error('Credenciales inválidas')
      }

      console.log('✅ [AuthService] Contraseña válida')
      console.log('✅ [AuthService] Login exitoso')

      // Retornar datos del usuario
      return {
        id: users.id,
        email: users.email,
        username: users.username,
        firstName: users.first_name,
        lastName: users.last_name,
        role: (users.roles as any)?.name || 'operator',
        companyId: users.company_id || undefined,
      }
    } catch (error: any) {
      console.error('❌ [AuthService] Login error:', error)
      throw new Error(error.message || 'Error al iniciar sesión')
    }
  },

  /**
   * Verificar si el usuario está autenticado
   */
  async getCurrentUser(): Promise<AuthUser | null> {
    try {
      // En una implementación real con Supabase Auth, usarías:
      // const { data: { user } } = await supabase.auth.getUser()
      
      // Por ahora, retornamos null ya que usamos autenticación personalizada
      return null
    } catch (error) {
      console.error('Get current user error:', error)
      return null
    }
  },

  /**
   * Logout
   */
  async logout(): Promise<void> {
    // En una implementación real con Supabase Auth:
    // await supabase.auth.signOut()
    
    // Por ahora, solo limpiamos el store local (manejado por zustand)
    return Promise.resolve()
  },
}
