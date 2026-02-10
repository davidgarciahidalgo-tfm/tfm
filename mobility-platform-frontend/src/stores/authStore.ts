import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { authService } from '../services/authService'

interface User {
  id: string
  email: string
  username: string
  firstName: string
  lastName: string
  role: string
  companyId?: string
}

interface AuthState {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  loginAttempts: number
  lastLoginAttempt: number | null
  
  login: (usernameOrEmail: string, password: string) => Promise<void>
  logout: () => void
  resetLoginAttempts: () => void
  updateUser: (user: Partial<User>) => void
}

const MAX_LOGIN_ATTEMPTS = Number(import.meta.env.VITE_MAX_LOGIN_ATTEMPTS) || 5
const LOCKOUT_DURATION = 15 * 60 * 1000 // 15 minutos
const ENABLE_MOCK = import.meta.env.VITE_ENABLE_MOCK_DATA === 'true'

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      loginAttempts: 0,
      lastLoginAttempt: null,

      login: async (usernameOrEmail: string, password: string) => {
        const state = get()
        
        console.log('🔐 [AuthStore] Iniciando login...', { 
          usernameOrEmail, 
          enableMock: ENABLE_MOCK,
          loginAttempts: state.loginAttempts 
        })
        
        // Check if account is locked
        if (state.lastLoginAttempt && 
            Date.now() - state.lastLoginAttempt < LOCKOUT_DURATION &&
            state.loginAttempts >= MAX_LOGIN_ATTEMPTS) {
          const remainingTime = Math.ceil((LOCKOUT_DURATION - (Date.now() - state.lastLoginAttempt)) / 1000 / 60)
          console.error('❌ [AuthStore] Cuenta bloqueada')
          throw new Error(`Cuenta bloqueada. Intenta en ${remainingTime} minutos`)
        }

        try {
          // Si está en modo mock, usar datos simulados
          if (ENABLE_MOCK) {
            console.log('ℹ️ [AuthStore] Usando modo MOCK')
            await new Promise(resolve => setTimeout(resolve, 800))
            
            if ((usernameOrEmail === 'admin@mobility.com' || usernameOrEmail === 'superadmin') && password === 'Admin123!') {
              const mockUser: User = {
                id: '1',
                email: 'admin@mobility.com',
                username: 'superadmin',
                firstName: 'Admin',
                lastName: 'Usuario',
                role: 'superadmin',
              }
              
              console.log('✅ [AuthStore] Login mock exitoso')
              set({
                user: mockUser,
                token: 'mock-jwt-token',
                isAuthenticated: true,
                loginAttempts: 0,
                lastLoginAttempt: null
              })
              return
            }
            
            console.error('❌ [AuthStore] Credenciales mock inválidas')
            throw new Error('Credenciales inválidas')
          }

          console.log('ℹ️ [AuthStore] Usando modo REAL (Supabase)')
          
          // IMPORTANTE: Limpiar cualquier sesión anterior antes de hacer login
          // Esto evita que datos antiguos del localStorage interfieran
          set({
            user: null,
            token: null,
            isAuthenticated: false,
          })

          console.log('📡 [AuthStore] Llamando a authService.login...')
          
          // Login real con Supabase - siempre obtiene datos frescos de la BD
          const user = await authService.login({
            usernameOrEmail,
            password,
          })

          console.log('✅ [AuthStore] Login exitoso desde BD:', { 
            username: user.username,
            role: user.role,
            email: user.email
          })

          set({
            user: {
              id: user.id,
              email: user.email,
              username: user.username,
              firstName: user.firstName,
              lastName: user.lastName,
              role: user.role,
              companyId: user.companyId,
            },
            token: `supabase-session-${user.id}`, // En producción, usar token real
            isAuthenticated: true,
            loginAttempts: 0,
            lastLoginAttempt: null
          })
          
          console.log('✅ [AuthStore] Estado actualizado correctamente')
        } catch (error: any) {
          console.error('❌ [AuthStore] Error en login:', error)
          set({
            loginAttempts: state.loginAttempts + 1,
            lastLoginAttempt: Date.now()
          })
          throw error
        }
      },

      logout: () => {
        authService.logout()
        set({
          user: null,
          token: null,
          isAuthenticated: false,
          loginAttempts: 0,
          lastLoginAttempt: null
        })
      },

      resetLoginAttempts: () => {
        set({
          loginAttempts: 0,
          lastLoginAttempt: null
        })
      },

      updateUser: (updatedUser: Partial<User>) => {
        const state = get()
        if (state.user) {
          set({
            user: {
              ...state.user,
              ...updatedUser,
            }
          })
        }
      }
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
)

// Auto-logout cuando el token expira
if (typeof window !== 'undefined') {
  const SESSION_TIMEOUT = (Number(import.meta.env.VITE_SESSION_TIMEOUT_MINUTES) || 30) * 60 * 1000
  
  let timeoutId: ReturnType<typeof setTimeout>
  
  const resetTimeout = () => {
    clearTimeout(timeoutId)
    timeoutId = setTimeout(() => {
      useAuthStore.getState().logout()
    }, SESSION_TIMEOUT)
  }
  
  // Reset timeout en cada actividad del usuario
  ['mousedown', 'keydown', 'scroll', 'touchstart'].forEach(event => {
    document.addEventListener(event, resetTimeout, true)
  })
  
  resetTimeout()
}

