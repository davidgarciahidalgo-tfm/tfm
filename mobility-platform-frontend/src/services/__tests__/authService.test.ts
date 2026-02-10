import { describe, it, expect, beforeEach, vi } from 'vitest'

// Mock the supabase module FIRST (before any imports that use it)
vi.mock('../../config/supabase', () => {
  const bcrypt = require('bcryptjs')
  
  const mockUsers = [
    {
      id: '1',
      email: 'admin@mobility.com',
      username: 'superadmin',
      password_hash: bcrypt.hashSync('Admin123!', 10),
      first_name: 'Admin',
      last_name: 'Usuario',
      company_id: null,
      is_active: true,
      role_id: 'role-1',
      roles: {
        id: 'role-1',
        name: 'superadmin',
        permissions: ['all'],
      },
    },
    {
      id: '2',
      email: 'juan.garcia@mobility.com',
      username: 'juan.garcia',
      password_hash: bcrypt.hashSync('Admin123!', 10),
      first_name: 'Juan',
      last_name: 'García',
      company_id: 'company-1',
      is_active: true,
      role_id: 'role-2',
      roles: {
        id: 'role-2',
        name: 'admin',
        permissions: ['manage_company', 'manage_users'],
      },
    },
    {
      id: '3',
      email: 'maria.lopez@transporttech.com',
      username: 'maria.lopez',
      password_hash: bcrypt.hashSync('Admin123!', 10),
      first_name: 'María',
      last_name: 'López',
      company_id: 'company-2',
      is_active: true,
      role_id: 'role-3',
      roles: {
        id: 'role-3',
        name: 'operator',
        permissions: ['manage_devices', 'view_stations'],
      },
    },
    {
      id: '4',
      email: 'viewer@mobility.com',
      username: 'viewer',
      password_hash: bcrypt.hashSync('Admin123!', 10),
      first_name: 'View',
      last_name: 'User',
      company_id: 'company-1',
      is_active: true,
      role_id: 'role-4',
      roles: {
        id: 'role-4',
        name: 'viewer',
        permissions: ['view_dashboard'],
      },
    },
    {
      id: '5',
      email: 'inactive@mobility.com',
      username: 'inactive_user',
      password_hash: bcrypt.hashSync('Admin123!', 10),
      first_name: 'Inactive',
      last_name: 'User',
      company_id: 'company-1',
      is_active: false,
      role_id: 'role-4',
      roles: {
        id: 'role-4',
        name: 'viewer',
        permissions: ['view_dashboard'],
      },
    },
  ]

  return {
    supabase: {
      from: vi.fn((table: string) => ({
        select: vi.fn(() => ({
          or: vi.fn((query: string) => ({
            eq: vi.fn((field: string, value: any) => ({
              single: vi.fn(() => {
                const usernameOrEmail = query.split(',')[0].split('.eq.')[1] || 
                                       query.split(',')[1]?.split('.eq.')[1]
                
                const user = mockUsers.find(
                  u => u.username === usernameOrEmail || u.email === usernameOrEmail
                )
                
                if (user && value === true && user.is_active) {
                  return { data: user, error: null }
                }
                
                return { data: null, error: new Error('User not found') }
              }),
            })),
          })),
        })),
        update: vi.fn(() => ({
          eq: vi.fn(() => ({ data: null, error: null })),
        })),
      })),
    },
  }
})

import { authService } from '../authService'

describe('AuthService - Login Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Superadmin Login', () => {
    it('should login successfully with superadmin username', async () => {
      const result = await authService.login({
        usernameOrEmail: 'superadmin',
        password: 'Admin123!',
      })

      expect(result).toBeDefined()
      expect(result.username).toBe('superadmin')
      expect(result.email).toBe('admin@mobility.com')
      expect(result.role).toBe('superadmin')
      expect(result.firstName).toBe('Admin')
      expect(result.lastName).toBe('Usuario')
      expect(result.companyId).toBeUndefined()
    })

    it('should login successfully with superadmin email', async () => {
      const result = await authService.login({
        usernameOrEmail: 'admin@mobility.com',
        password: 'Admin123!',
      })

      expect(result).toBeDefined()
      expect(result.username).toBe('superadmin')
      expect(result.role).toBe('superadmin')
      expect(result.companyId).toBeUndefined()
    })

    it('should fail login with wrong password for superadmin', async () => {
      await expect(
        authService.login({
          usernameOrEmail: 'superadmin',
          password: 'WrongPassword123!',
        })
      ).rejects.toThrow('Credenciales inválidas')
    })

    it('should verify superadmin has no company association', async () => {
      const result = await authService.login({
        usernameOrEmail: 'superadmin',
        password: 'Admin123!',
      })

      expect(result.companyId).toBeUndefined()
      expect(result.role).toBe('superadmin')
    })
  })

  describe('Admin Role Login', () => {
    it('should login successfully with admin username', async () => {
      const result = await authService.login({
        usernameOrEmail: 'juan.garcia',
        password: 'Admin123!',
      })

      expect(result).toBeDefined()
      expect(result.username).toBe('juan.garcia')
      expect(result.email).toBe('juan.garcia@mobility.com')
      expect(result.role).toBe('admin')
      expect(result.firstName).toBe('Juan')
      expect(result.lastName).toBe('García')
      expect(result.companyId).toBe('company-1')
    })

    it('should login successfully with admin email', async () => {
      const result = await authService.login({
        usernameOrEmail: 'juan.garcia@mobility.com',
        password: 'Admin123!',
      })

      expect(result).toBeDefined()
      expect(result.role).toBe('admin')
      expect(result.companyId).toBeDefined()
    })

    it('should verify admin has company association', async () => {
      const result = await authService.login({
        usernameOrEmail: 'juan.garcia',
        password: 'Admin123!',
      })

      expect(result.companyId).toBe('company-1')
      expect(result.role).toBe('admin')
    })

    it('should fail login with wrong password for admin', async () => {
      await expect(
        authService.login({
          usernameOrEmail: 'juan.garcia',
          password: 'WrongPassword!',
        })
      ).rejects.toThrow('Credenciales inválidas')
    })
  })

  describe('Operator Role Login', () => {
    it('should login successfully with operator username', async () => {
      const result = await authService.login({
        usernameOrEmail: 'maria.lopez',
        password: 'Admin123!',
      })

      expect(result).toBeDefined()
      expect(result.username).toBe('maria.lopez')
      expect(result.email).toBe('maria.lopez@transporttech.com')
      expect(result.role).toBe('operator')
      expect(result.firstName).toBe('María')
      expect(result.lastName).toBe('López')
      expect(result.companyId).toBe('company-2')
    })

    it('should login successfully with operator email', async () => {
      const result = await authService.login({
        usernameOrEmail: 'maria.lopez@transporttech.com',
        password: 'Admin123!',
      })

      expect(result).toBeDefined()
      expect(result.role).toBe('operator')
      expect(result.companyId).toBeDefined()
    })

    it('should verify operator has company association', async () => {
      const result = await authService.login({
        usernameOrEmail: 'maria.lopez',
        password: 'Admin123!',
      })

      expect(result.companyId).toBe('company-2')
      expect(result.role).toBe('operator')
    })

    it('should fail login with wrong password for operator', async () => {
      await expect(
        authService.login({
          usernameOrEmail: 'maria.lopez',
          password: 'IncorrectPass123!',
        })
      ).rejects.toThrow('Credenciales inválidas')
    })
  })

  describe('Viewer Role Login', () => {
    it('should login successfully with viewer username', async () => {
      const result = await authService.login({
        usernameOrEmail: 'viewer',
        password: 'Admin123!',
      })

      expect(result).toBeDefined()
      expect(result.username).toBe('viewer')
      expect(result.email).toBe('viewer@mobility.com')
      expect(result.role).toBe('viewer')
      expect(result.firstName).toBe('View')
      expect(result.lastName).toBe('User')
      expect(result.companyId).toBe('company-1')
    })

    it('should login successfully with viewer email', async () => {
      const result = await authService.login({
        usernameOrEmail: 'viewer@mobility.com',
        password: 'Admin123!',
      })

      expect(result).toBeDefined()
      expect(result.role).toBe('viewer')
      expect(result.companyId).toBeDefined()
    })

    it('should verify viewer has company association', async () => {
      const result = await authService.login({
        usernameOrEmail: 'viewer',
        password: 'Admin123!',
      })

      expect(result.companyId).toBe('company-1')
      expect(result.role).toBe('viewer')
    })

    it('should fail login with wrong password for viewer', async () => {
      await expect(
        authService.login({
          usernameOrEmail: 'viewer',
          password: 'BadPassword!',
        })
      ).rejects.toThrow('Credenciales inválidas')
    })
  })

  describe('Edge Cases and Security', () => {
    it('should fail login with non-existent username', async () => {
      await expect(
        authService.login({
          usernameOrEmail: 'nonexistent',
          password: 'Admin123!',
        })
      ).rejects.toThrow('Credenciales inválidas')
    })

    it('should fail login with non-existent email', async () => {
      await expect(
        authService.login({
          usernameOrEmail: 'nonexistent@test.com',
          password: 'Admin123!',
        })
      ).rejects.toThrow('Credenciales inválidas')
    })

    it('should fail login for inactive user', async () => {
      await expect(
        authService.login({
          usernameOrEmail: 'inactive_user',
          password: 'Admin123!',
        })
      ).rejects.toThrow('Credenciales inválidas')
    })

    it('should fail login with empty username', async () => {
      await expect(
        authService.login({
          usernameOrEmail: '',
          password: 'Admin123!',
        })
      ).rejects.toThrow()
    })

    it('should fail login with empty password', async () => {
      await expect(
        authService.login({
          usernameOrEmail: 'superadmin',
          password: '',
        })
      ).rejects.toThrow()
    })
  })

  describe('Role-based Company Association', () => {
    it('should verify only superadmin has no company', async () => {
      const superadmin = await authService.login({
        usernameOrEmail: 'superadmin',
        password: 'Admin123!',
      })

      const admin = await authService.login({
        usernameOrEmail: 'juan.garcia',
        password: 'Admin123!',
      })

      const operator = await authService.login({
        usernameOrEmail: 'maria.lopez',
        password: 'Admin123!',
      })

      const viewer = await authService.login({
        usernameOrEmail: 'viewer',
        password: 'Admin123!',
      })

      expect(superadmin.companyId).toBeUndefined()
      expect(admin.companyId).toBeDefined()
      expect(operator.companyId).toBeDefined()
      expect(viewer.companyId).toBeDefined()
    })

    it('should verify all non-superadmin roles have company', async () => {
      const roles = ['juan.garcia', 'maria.lopez', 'viewer']
      
      for (const username of roles) {
        const result = await authService.login({
          usernameOrEmail: username,
          password: 'Admin123!',
        })
        
        expect(result.companyId).toBeDefined()
        expect(result.role).not.toBe('superadmin')
      }
    })
  })

  describe('Password Security', () => {
    it('should not accept passwords with small variations', async () => {
      const wrongPasswords = [
        'admin123!',  // lowercase
        'ADMIN123!',  // uppercase
        'Admin123',   // missing special char
        'Admin1234!', // different number
      ]

      for (const wrongPass of wrongPasswords) {
        await expect(
          authService.login({
            usernameOrEmail: 'superadmin',
            password: wrongPass,
          })
        ).rejects.toThrow('Credenciales inválidas')
      }
    })
  })

  describe('Login Response Structure', () => {
    it('should return complete user data for superadmin', async () => {
      const result = await authService.login({
        usernameOrEmail: 'superadmin',
        password: 'Admin123!',
      })

      expect(result).toHaveProperty('id')
      expect(result).toHaveProperty('email')
      expect(result).toHaveProperty('username')
      expect(result).toHaveProperty('firstName')
      expect(result).toHaveProperty('lastName')
      expect(result).toHaveProperty('role')
      expect(result.role).toBe('superadmin')
    })

    it('should return complete user data for all roles', async () => {
      const users = [
        { username: 'superadmin', role: 'superadmin' },
        { username: 'juan.garcia', role: 'admin' },
        { username: 'maria.lopez', role: 'operator' },
        { username: 'viewer', role: 'viewer' },
      ]

      for (const user of users) {
        const result = await authService.login({
          usernameOrEmail: user.username,
          password: 'Admin123!',
        })

        expect(result).toHaveProperty('id')
        expect(result).toHaveProperty('email')
        expect(result).toHaveProperty('username')
        expect(result).toHaveProperty('firstName')
        expect(result).toHaveProperty('lastName')
        expect(result).toHaveProperty('role')
        expect(result.role).toBe(user.role)
      }
    })
  })
})
