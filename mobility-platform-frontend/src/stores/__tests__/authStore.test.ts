import { describe, it, expect, beforeEach, vi } from 'vitest'
import { useAuthStore } from '../authStore'
import { authService } from '../../services/authService'

// Mock authService
vi.mock('../../services/authService', () => ({
  authService: {
    login: vi.fn(),
    logout: vi.fn(),
    getCurrentUser: vi.fn(),
  },
}))

describe('AuthStore - Authentication Flow', () => {
  beforeEach(() => {
    // Reset store state
    useAuthStore.setState({
      user: null,
      token: null,
      isAuthenticated: false,
      loginAttempts: 0,
      lastLoginAttempt: null,
    })
    vi.clearAllMocks()
  })

  describe('Superadmin Authentication', () => {
    it('should authenticate superadmin and update store', async () => {
      const mockSuperadmin = {
        id: '1',
        email: 'admin@mobility.com',
        username: 'superadmin',
        firstName: 'Admin',
        lastName: 'Usuario',
        role: 'superadmin',
        companyId: undefined,
      }

      vi.mocked(authService.login).mockResolvedValue(mockSuperadmin)

      await useAuthStore.getState().login('superadmin', 'Admin123!')

      const state = useAuthStore.getState()
      expect(state.isAuthenticated).toBe(true)
      expect(state.user).toEqual(mockSuperadmin)
      expect(state.user?.role).toBe('superadmin')
      expect(state.user?.companyId).toBeUndefined()
      expect(state.loginAttempts).toBe(0)
    })

    it('should set token for superadmin', async () => {
      const mockSuperadmin = {
        id: '1',
        email: 'admin@mobility.com',
        username: 'superadmin',
        firstName: 'Admin',
        lastName: 'Usuario',
        role: 'superadmin',
      }

      vi.mocked(authService.login).mockResolvedValue(mockSuperadmin)

      await useAuthStore.getState().login('superadmin', 'Admin123!')

      const state = useAuthStore.getState()
      expect(state.token).toContain('supabase-session')
      expect(state.token).toContain(mockSuperadmin.id)
    })
  })

  describe('Admin Role Authentication', () => {
    it('should authenticate admin and update store', async () => {
      const mockAdmin = {
        id: '2',
        email: 'juan.garcia@mobility.com',
        username: 'juan.garcia',
        firstName: 'Juan',
        lastName: 'García',
        role: 'admin',
        companyId: 'company-1',
      }

      vi.mocked(authService.login).mockResolvedValue(mockAdmin)

      await useAuthStore.getState().login('juan.garcia', 'Admin123!')

      const state = useAuthStore.getState()
      expect(state.isAuthenticated).toBe(true)
      expect(state.user).toEqual(mockAdmin)
      expect(state.user?.role).toBe('admin')
      expect(state.user?.companyId).toBe('company-1')
    })
  })

  describe('Operator Role Authentication', () => {
    it('should authenticate operator and update store', async () => {
      const mockOperator = {
        id: '3',
        email: 'maria.lopez@transporttech.com',
        username: 'maria.lopez',
        firstName: 'María',
        lastName: 'López',
        role: 'operator',
        companyId: 'company-2',
      }

      vi.mocked(authService.login).mockResolvedValue(mockOperator)

      await useAuthStore.getState().login('maria.lopez', 'Admin123!')

      const state = useAuthStore.getState()
      expect(state.isAuthenticated).toBe(true)
      expect(state.user).toEqual(mockOperator)
      expect(state.user?.role).toBe('operator')
      expect(state.user?.companyId).toBe('company-2')
    })
  })

  describe('Viewer Role Authentication', () => {
    it('should authenticate viewer and update store', async () => {
      const mockViewer = {
        id: '4',
        email: 'viewer@mobility.com',
        username: 'viewer',
        firstName: 'View',
        lastName: 'User',
        role: 'viewer',
        companyId: 'company-1',
      }

      vi.mocked(authService.login).mockResolvedValue(mockViewer)

      await useAuthStore.getState().login('viewer', 'Admin123!')

      const state = useAuthStore.getState()
      expect(state.isAuthenticated).toBe(true)
      expect(state.user).toEqual(mockViewer)
      expect(state.user?.role).toBe('viewer')
      expect(state.user?.companyId).toBe('company-1')
    })
  })

  describe('Login Attempts and Security', () => {
    it('should increment login attempts on failed login', async () => {
      vi.mocked(authService.login).mockRejectedValue(
        new Error('Credenciales inválidas')
      )

      try {
        await useAuthStore.getState().login('wrong', 'wrong')
      } catch (error) {
        // Expected error
      }

      const state = useAuthStore.getState()
      expect(state.loginAttempts).toBe(1)
      expect(state.lastLoginAttempt).toBeTruthy()
    })

    it('should reset login attempts on successful login', async () => {
      const mockUser = {
        id: '1',
        email: 'admin@mobility.com',
        username: 'superadmin',
        firstName: 'Admin',
        lastName: 'Usuario',
        role: 'superadmin',
      }

      // First, fail some attempts
      vi.mocked(authService.login).mockRejectedValue(
        new Error('Credenciales inválidas')
      )

      try {
        await useAuthStore.getState().login('wrong', 'wrong')
      } catch {}
      
      try {
        await useAuthStore.getState().login('wrong', 'wrong')
      } catch {}

      expect(useAuthStore.getState().loginAttempts).toBe(2)

      // Then succeed
      vi.mocked(authService.login).mockResolvedValue(mockUser)
      await useAuthStore.getState().login('superadmin', 'Admin123!')

      const state = useAuthStore.getState()
      expect(state.loginAttempts).toBe(0)
      expect(state.isAuthenticated).toBe(true)
    })

    it('should block login after max attempts', async () => {
      vi.mocked(authService.login).mockRejectedValue(
        new Error('Credenciales inválidas')
      )

      // Make 5 failed attempts
      for (let i = 0; i < 5; i++) {
        try {
          await useAuthStore.getState().login('wrong', 'wrong')
        } catch {}
      }

      // 6th attempt should be blocked
      await expect(
        useAuthStore.getState().login('superadmin', 'Admin123!')
      ).rejects.toThrow(/bloqueada/)
    })
  })

  describe('Logout Functionality', () => {
    it('should clear all auth data on logout', async () => {
      // First login
      const mockUser = {
        id: '1',
        email: 'admin@mobility.com',
        username: 'superadmin',
        firstName: 'Admin',
        lastName: 'Usuario',
        role: 'superadmin',
      }

      vi.mocked(authService.login).mockResolvedValue(mockUser)
      await useAuthStore.getState().login('superadmin', 'Admin123!')

      expect(useAuthStore.getState().isAuthenticated).toBe(true)

      // Then logout
      useAuthStore.getState().logout()

      const state = useAuthStore.getState()
      expect(state.user).toBeNull()
      expect(state.token).toBeNull()
      expect(state.isAuthenticated).toBe(false)
      expect(state.loginAttempts).toBe(0)
      expect(state.lastLoginAttempt).toBeNull()
    })

    it('should call authService.logout', () => {
      useAuthStore.getState().logout()
      expect(authService.logout).toHaveBeenCalled()
    })
  })

  describe('Role-based State Verification', () => {
    it('should maintain correct state for each role', async () => {
      const roles = [
        {
          username: 'superadmin',
          user: {
            id: '1',
            email: 'admin@mobility.com',
            username: 'superadmin',
            firstName: 'Admin',
            lastName: 'Usuario',
            role: 'superadmin',
            companyId: undefined,
          },
        },
        {
          username: 'juan.garcia',
          user: {
            id: '2',
            email: 'juan.garcia@mobility.com',
            username: 'juan.garcia',
            firstName: 'Juan',
            lastName: 'García',
            role: 'admin',
            companyId: 'company-1',
          },
        },
        {
          username: 'maria.lopez',
          user: {
            id: '3',
            email: 'maria.lopez@transporttech.com',
            username: 'maria.lopez',
            firstName: 'María',
            lastName: 'López',
            role: 'operator',
            companyId: 'company-2',
          },
        },
        {
          username: 'viewer',
          user: {
            id: '4',
            email: 'viewer@mobility.com',
            username: 'viewer',
            firstName: 'View',
            lastName: 'User',
            role: 'viewer',
            companyId: 'company-1',
          },
        },
      ]

      for (const { username, user } of roles) {
        // Reset state
        useAuthStore.setState({
          user: null,
          token: null,
          isAuthenticated: false,
          loginAttempts: 0,
          lastLoginAttempt: null,
        })

        vi.mocked(authService.login).mockResolvedValue(user)
        await useAuthStore.getState().login(username, 'Admin123!')

        const state = useAuthStore.getState()
        expect(state.user).toEqual(user)
        expect(state.isAuthenticated).toBe(true)
        expect(state.user?.role).toBe(user.role)

        if (user.role === 'superadmin') {
          expect(state.user?.companyId).toBeUndefined()
        } else {
          expect(state.user?.companyId).toBeDefined()
        }
      }
    })
  })
})
