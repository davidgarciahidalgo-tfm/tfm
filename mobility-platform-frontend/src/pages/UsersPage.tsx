import { useState, useEffect } from 'react'
import {
  Box,
  Button,
  Card,
  CardContent,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Typography,
  Chip,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  CircularProgress,
  Snackbar,
} from '@mui/material'
import {
  Add,
  Edit,
  Delete,
  PersonAdd,
} from '@mui/icons-material'
import { userService, type UserData, type CreateUserInput, type UpdateUserInput } from '../services/userService'
import { roleService, type RoleData } from '../services/roleService'
import { companyService, type CompanyData } from '../services/companyService'
import { useAuthStore } from '../stores/authStore'

const UsersPage = () => {
  // Auth store
  const { user: currentUser, updateUser: updateAuthUser } = useAuthStore()

  // State for users
  const [users, setUsers] = useState<UserData[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // State for roles and companies
  const [roles, setRoles] = useState<RoleData[]>([])
  const [companies, setCompanies] = useState<CompanyData[]>([])

  // Dialog state
  const [openDialog, setOpenDialog] = useState(false)
  const [selectedUser, setSelectedUser] = useState<UserData | null>(null)
  const [saving, setSaving] = useState(false)

  // Form data
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    username: '',
    email: '',
    password: '',
    roleId: '',
    companyId: '',
  })

  // Snackbar
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success' as 'success' | 'error',
  })

  // Load data on mount
  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    try {
      setLoading(true)
      setError(null)
      const [usersData, rolesData, companiesData] = await Promise.all([
        userService.getUsers(),
        roleService.getRoles(),
        companyService.getCompanies(),
      ])
      setUsers(usersData)
      setRoles(rolesData)
      setCompanies(companiesData)
    } catch (err: any) {
      console.error('Load data error:', err)
      setError(err.message || 'Error al cargar datos')
    } finally {
      setLoading(false)
    }
  }

  const handleOpenDialog = (user?: UserData) => {
    if (user) {
      setSelectedUser(user)
      setFormData({
        firstName: user.firstName,
        lastName: user.lastName,
        username: user.username,
        email: user.email,
        password: '', // No prellenar password en edición
        roleId: user.role.id,
        companyId: user.companyId || '',
      })
    } else {
      setSelectedUser(null)
      setFormData({
        firstName: '',
        lastName: '',
        username: '',
        email: '',
        password: '',
        roleId: '',
        companyId: '',
      })
    }
    setOpenDialog(true)
  }

  const handleCloseDialog = () => {
    setOpenDialog(false)
    setSelectedUser(null)
    setFormData({
      firstName: '',
      lastName: '',
      username: '',
      email: '',
      password: '',
      roleId: '',
      companyId: '',
    })
  }

  const handleSave = async () => {
    // Validación básica
    if (!formData.firstName || !formData.lastName || !formData.username || 
        !formData.email || !formData.roleId) {
      setSnackbar({
        open: true,
        message: 'Por favor completa todos los campos obligatorios',
        severity: 'error',
      })
      return
    }

    // Si es nuevo usuario, password es obligatorio
    if (!selectedUser && !formData.password) {
      setSnackbar({
        open: true,
        message: 'La contraseña es obligatoria para usuarios nuevos',
        severity: 'error',
      })
      return
    }

    try {
      setSaving(true)

      if (selectedUser) {
        // Actualizar usuario
        const updateInput: UpdateUserInput = {
          firstName: formData.firstName,
          lastName: formData.lastName,
          username: formData.username,
          email: formData.email,
          roleId: formData.roleId,
          companyId: formData.companyId || undefined,
        }
        
        // Solo incluir password si se ingresó uno nuevo
        if (formData.password) {
          updateInput.password = formData.password
        }

        const updatedUser = await userService.updateUser(selectedUser.id, updateInput)
        
        // Si el usuario actualizado es el usuario actual logueado, actualizar el authStore
        if (currentUser && currentUser.id === selectedUser.id) {
          updateAuthUser({
            firstName: updatedUser.firstName,
            lastName: updatedUser.lastName,
            username: updatedUser.username,
            email: updatedUser.email,
            role: updatedUser.role.name,
            companyId: updatedUser.companyId,
          })
        }
        
        setSnackbar({
          open: true,
          message: 'Usuario actualizado correctamente',
          severity: 'success',
        })
      } else {
        // Crear nuevo usuario
        const createInput: CreateUserInput = {
          firstName: formData.firstName,
          lastName: formData.lastName,
          username: formData.username,
          email: formData.email,
          password: formData.password,
          roleId: formData.roleId,
          companyId: formData.companyId || undefined,
        }

        await userService.createUser(createInput)
        setSnackbar({
          open: true,
          message: 'Usuario creado correctamente',
          severity: 'success',
        })
      }

      handleCloseDialog()
      await loadData() // Recargar lista
    } catch (err: any) {
      console.error('Save user error:', err)
      setSnackbar({
        open: true,
        message: err.message || 'Error al guardar usuario',
        severity: 'error',
      })
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async (userId: string) => {
    if (!window.confirm('¿Estás seguro de que quieres eliminar este usuario?')) {
      return
    }

    try {
      await userService.deleteUser(userId)
      setSnackbar({
        open: true,
        message: 'Usuario eliminado correctamente',
        severity: 'success',
      })
      await loadData() // Recargar lista
    } catch (err: any) {
      console.error('Delete user error:', err)
      setSnackbar({
        open: true,
        message: err.message || 'Error al eliminar usuario',
        severity: 'error',
      })
    }
  }

  const handleToggleStatus = async (user: UserData) => {
    try {
      await userService.updateUser(user.id, {
        isActive: !user.isActive,
      })
      setSnackbar({
        open: true,
        message: `Usuario ${!user.isActive ? 'activado' : 'desactivado'} correctamente`,
        severity: 'success',
      })
      await loadData()
    } catch (err: any) {
      console.error('Toggle status error:', err)
      setSnackbar({
        open: true,
        message: err.message || 'Error al cambiar estado',
        severity: 'error',
      })
    }
  }

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '400px' }}>
        <CircularProgress />
      </Box>
    )
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
        <Button variant="contained" onClick={loadData}>
          Reintentar
        </Button>
      </Box>
    )
  }

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1" sx={{ fontWeight: 600 }}>
          Usuarios
        </Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => handleOpenDialog()}
        >
          Nuevo Usuario
        </Button>
      </Box>

      {/* Stats Cards */}
      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 2, mb: 3 }}>
        <Card>
          <CardContent>
            <Typography color="text.secondary" gutterBottom>
              Total Usuarios
            </Typography>
            <Typography variant="h4">{users.length}</Typography>
          </CardContent>
        </Card>
        <Card>
          <CardContent>
            <Typography color="text.secondary" gutterBottom>
              Usuarios Activos
            </Typography>
            <Typography variant="h4">
              {users.filter(u => u.isActive).length}
            </Typography>
          </CardContent>
        </Card>
        <Card>
          <CardContent>
            <Typography color="text.secondary" gutterBottom>
              Usuarios Inactivos
            </Typography>
            <Typography variant="h4">
              {users.filter(u => !u.isActive).length}
            </Typography>
          </CardContent>
        </Card>
      </Box>

      {/* Users Table */}
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Nombre</TableCell>
              <TableCell>Usuario</TableCell>
              <TableCell>Email</TableCell>
              <TableCell>Rol</TableCell>
              <TableCell>Empresa</TableCell>
              <TableCell>Estado</TableCell>
              <TableCell align="right">Acciones</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {users.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} align="center">
                  <Box sx={{ py: 4 }}>
                    <PersonAdd sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                    <Typography variant="body1" color="text.secondary">
                      No hay usuarios registrados
                    </Typography>
                    <Button
                      variant="outlined"
                      startIcon={<Add />}
                      onClick={() => handleOpenDialog()}
                      sx={{ mt: 2 }}
                    >
                      Crear Primer Usuario
                    </Button>
                  </Box>
                </TableCell>
              </TableRow>
            ) : (
              users.map((user) => (
                <TableRow key={user.id} hover>
                  <TableCell>
                    <Typography variant="body2">
                      {user.firstName} {user.lastName}
                    </Typography>
                  </TableCell>
                  <TableCell>{user.username}</TableCell>
                  <TableCell>{user.email}</TableCell>
                  <TableCell>
                    <Chip
                      label={user.role.name}
                      size="small"
                      color={
                        user.role.name === 'superadmin'
                          ? 'error'
                          : user.role.name === 'admin'
                          ? 'primary'
                          : 'default'
                      }
                    />
                  </TableCell>
                  <TableCell>{user.companyName || '-'}</TableCell>
                  <TableCell>
                    <Chip
                      label={user.isActive ? 'Activo' : 'Inactivo'}
                      size="small"
                      color={user.isActive ? 'success' : 'default'}
                      onClick={() => handleToggleStatus(user)}
                      sx={{ cursor: 'pointer' }}
                    />
                  </TableCell>
                  <TableCell align="right">
                    <IconButton
                      size="small"
                      onClick={() => handleOpenDialog(user)}
                      color="primary"
                    >
                      <Edit />
                    </IconButton>
                    <IconButton
                      size="small"
                      onClick={() => handleDelete(user.id)}
                      color="error"
                    >
                      <Delete />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Create/Edit Dialog */}
      <Dialog open={openDialog} onClose={handleCloseDialog} maxWidth="sm" fullWidth>
        <DialogTitle>
          {selectedUser ? 'Editar Usuario' : 'Nuevo Usuario'}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 2 }}>
            <TextField
              label="Nombre"
              value={formData.firstName}
              onChange={(e) => setFormData({ ...formData, firstName: e.target.value })}
              fullWidth
              required
            />
            <TextField
              label="Apellidos"
              value={formData.lastName}
              onChange={(e) => setFormData({ ...formData, lastName: e.target.value })}
              fullWidth
              required
            />
            <TextField
              label="Usuario"
              value={formData.username}
              onChange={(e) => setFormData({ ...formData, username: e.target.value })}
              fullWidth
              required
            />
            <TextField
              label="Email"
              type="email"
              value={formData.email}
              onChange={(e) => setFormData({ ...formData, email: e.target.value })}
              fullWidth
              required
            />
            <TextField
              label={selectedUser ? 'Nueva Contraseña (opcional)' : 'Contraseña'}
              type="password"
              value={formData.password}
              onChange={(e) => setFormData({ ...formData, password: e.target.value })}
              fullWidth
              required={!selectedUser}
              helperText={selectedUser ? 'Dejar vacío para mantener la contraseña actual' : ''}
            />
            <FormControl fullWidth required>
              <InputLabel>Rol</InputLabel>
              <Select
                value={formData.roleId}
                onChange={(e) => setFormData({ ...formData, roleId: e.target.value })}
                label="Rol"
              >
                {roles.map((role) => (
                  <MenuItem key={role.id} value={role.id}>
                    {role.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl fullWidth>
              <InputLabel>Empresa</InputLabel>
              <Select
                value={formData.companyId}
                onChange={(e) => setFormData({ ...formData, companyId: e.target.value })}
                label="Empresa"
              >
                <MenuItem value="">
                  <em>Sin empresa</em>
                </MenuItem>
                {companies.map((company) => (
                  <MenuItem key={company.id} value={company.id}>
                    {company.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog} disabled={saving}>
            Cancelar
          </Button>
          <Button
            onClick={handleSave}
            variant="contained"
            disabled={saving}
            startIcon={saving ? <CircularProgress size={20} /> : null}
          >
            {saving ? 'Guardando...' : 'Guardar'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert
          onClose={() => setSnackbar({ ...snackbar, open: false })}
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  )
}

export default UsersPage
