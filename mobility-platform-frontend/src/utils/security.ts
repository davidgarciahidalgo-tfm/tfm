import DOMPurify from 'dompurify'

/**
 * Sanitiza input del usuario para prevenir XSS
 */
export const sanitizeInput = (input: string): string => {
  return DOMPurify.sanitize(input, { 
    ALLOWED_TAGS: [],
    ALLOWED_ATTR: [] 
  }).trim()
}

/**
 * Sanitiza HTML permitiendo solo tags seguros
 */
export const sanitizeHTML = (html: string): string => {
  return DOMPurify.sanitize(html, {
    ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'p', 'br', 'span'],
    ALLOWED_ATTR: []
  })
}

/**
 * Valida formato de email
 */
export const isValidEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  return emailRegex.test(email)
}

/**
 * Valida contraseña segura
 * - Mínimo 8 caracteres
 * - Al menos una mayúscula
 * - Al menos una minúscula
 * - Al menos un número
 * - Al menos un carácter especial
 */
export const isStrongPassword = (password: string): boolean => {
  const minLength = 8
  const hasUpperCase = /[A-Z]/.test(password)
  const hasLowerCase = /[a-z]/.test(password)
  const hasNumbers = /\d/.test(password)
  const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(password)
  
  return (
    password.length >= minLength &&
    hasUpperCase &&
    hasLowerCase &&
    hasNumbers &&
    hasSpecialChar
  )
}

/**
 * Genera un ID único seguro
 */
export const generateSecureId = (): string => {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
}

/**
 * Ofusca información sensible (para logs)
 */
export const obfuscateSensitiveData = (data: string, visibleChars: number = 4): string => {
  if (data.length <= visibleChars) return '*'.repeat(data.length)
  return data.substring(0, visibleChars) + '*'.repeat(data.length - visibleChars)
}

/**
 * Valida que una URL sea segura
 */
export const isSafeURL = (url: string): boolean => {
  try {
    const parsed = new URL(url)
    return ['http:', 'https:'].includes(parsed.protocol)
  } catch {
    return false
  }
}

/**
 * Previene clickjacking verificando si estamos en un iframe
 */
export const preventClickjacking = (): void => {
  if (window.self !== window.top) {
    window.top!.location.href = window.self.location.href
  }
}

// Ejecutar al cargar
if (typeof window !== 'undefined') {
  preventClickjacking()
}
