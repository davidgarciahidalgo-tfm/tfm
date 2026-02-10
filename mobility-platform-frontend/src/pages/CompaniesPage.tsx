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
  Alert,
  CircularProgress,
  Snackbar,
} from '@mui/material'
import {
  Add,
  Edit,
  Delete,
  Business,
} from '@mui/icons-material'
import { companyService, type CompanyData, type CreateCompanyInput, type UpdateCompanyInput } from '../services/companyService'

const CompaniesPage = () => {
  const [companies, setCompanies] = useState<CompanyData[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [openDialog, setOpenDialog] = useState(false)
  const [selectedCompany, setSelectedCompany] = useState<CompanyData | null>(null)
  const [saving, setSaving] = useState(false)
  
  const [formData, setFormData] = useState({
    name: '',
    nif: '',
    address: '',
    phone: '',
    email: '',
  })

  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success' as 'success' | 'error',
  })

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await companyService.getCompanies()
      setCompanies(data)
    } catch (err: any) {
      setError(err.message || 'Error al cargar empresas')
    } finally {
      setLoading(false)
    }
  }

  const handleOpenDialog = (company?: CompanyData) => {
    if (company) {
      setSelectedCompany(company)
      setFormData({
        name: company.name,
        nif: company.nif,
        address: company.address || '',
        phone: company.phone || '',
        email: company.email || '',
      })
    } else {
      setSelectedCompany(null)
      setFormData({
        name: '',
        nif: '',
        address: '',
        phone: '',
        email: '',
      })
    }
    setOpenDialog(true)
  }

  const handleCloseDialog = () => {
    setOpenDialog(false)
    setSelectedCompany(null)
    setFormData({
      name: '',
      nif: '',
      address: '',
      phone: '',
      email: '',
    })
  }

  const handleSave = async () => {
    if (!formData.name || !formData.nif) {
      setSnackbar({
        open: true,
        message: 'Por favor completa los campos obligatorios (Nombre y NIF)',
        severity: 'error',
      })
      return
    }

    try {
      setSaving(true)

      if (selectedCompany) {
        const updateInput: UpdateCompanyInput = {
          name: formData.name,
          nif: formData.nif,
          address: formData.address || undefined,
          phone: formData.phone || undefined,
          email: formData.email || undefined,
        }
        await companyService.updateCompany(selectedCompany.id, updateInput)
        setSnackbar({
          open: true,
          message: 'Empresa actualizada correctamente',
          severity: 'success',
        })
      } else {
        const createInput: CreateCompanyInput = {
          name: formData.name,
          nif: formData.nif,
          address: formData.address || undefined,
          phone: formData.phone || undefined,
          email: formData.email || undefined,
        }
        await companyService.createCompany(createInput)
        setSnackbar({
          open: true,
          message: 'Empresa creada correctamente',
          severity: 'success',
        })
      }

      handleCloseDialog()
      await loadData()
    } catch (err: any) {
      setSnackbar({
        open: true,
        message: err.message || 'Error al guardar empresa',
        severity: 'error',
      })
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async (companyId: string) => {
    if (!window.confirm('¿Estás seguro de que quieres eliminar esta empresa?')) {
      return
    }

    try {
      await companyService.deleteCompany(companyId)
      setSnackbar({
        open: true,
        message: 'Empresa eliminada correctamente',
        severity: 'success',
      })
      await loadData()
    } catch (err: any) {
      setSnackbar({
        open: true,
        message: err.message || 'Error al eliminar empresa',
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
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1" sx={{ fontWeight: 600 }}>
          Empresas
        </Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => handleOpenDialog()}
        >
          Nueva Empresa
        </Button>
      </Box>

      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 2, mb: 3 }}>
        <Card>
          <CardContent>
            <Typography color="text.secondary" gutterBottom>
              Total Empresas
            </Typography>
            <Typography variant="h4">{companies.length}</Typography>
          </CardContent>
        </Card>
        <Card>
          <CardContent>
            <Typography color="text.secondary" gutterBottom>
              Empresas Activas
            </Typography>
            <Typography variant="h4">
              {companies.filter(c => c.isActive).length}
            </Typography>
          </CardContent>
        </Card>
      </Box>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Nombre</TableCell>
              <TableCell>NIF</TableCell>
              <TableCell>Dirección</TableCell>
              <TableCell>Teléfono</TableCell>
              <TableCell>Email</TableCell>
              <TableCell>Estado</TableCell>
              <TableCell align="right">Acciones</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {companies.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} align="center">
                  <Box sx={{ py: 4 }}>
                    <Business sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                    <Typography variant="body1" color="text.secondary">
                      No hay empresas registradas
                    </Typography>
                    <Button
                      variant="outlined"
                      startIcon={<Add />}
                      onClick={() => handleOpenDialog()}
                      sx={{ mt: 2 }}
                    >
                      Crear Primera Empresa
                    </Button>
                  </Box>
                </TableCell>
              </TableRow>
            ) : (
              companies.map((company) => (
                <TableRow key={company.id} hover>
                  <TableCell>{company.name}</TableCell>
                  <TableCell>{company.nif}</TableCell>
                  <TableCell>{company.address || '-'}</TableCell>
                  <TableCell>{company.phone || '-'}</TableCell>
                  <TableCell>{company.email || '-'}</TableCell>
                  <TableCell>
                    <Chip
                      label={company.isActive ? 'Activa' : 'Inactiva'}
                      size="small"
                      color={company.isActive ? 'success' : 'default'}
                    />
                  </TableCell>
                  <TableCell align="right">
                    <IconButton
                      size="small"
                      onClick={() => handleOpenDialog(company)}
                      color="primary"
                    >
                      <Edit />
                    </IconButton>
                    <IconButton
                      size="small"
                      onClick={() => handleDelete(company.id)}
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

      <Dialog open={openDialog} onClose={handleCloseDialog} maxWidth="sm" fullWidth>
        <DialogTitle>
          {selectedCompany ? 'Editar Empresa' : 'Nueva Empresa'}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 2 }}>
            <TextField
              label="Nombre"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              fullWidth
              required
            />
            <TextField
              label="NIF"
              value={formData.nif}
              onChange={(e) => setFormData({ ...formData, nif: e.target.value })}
              fullWidth
              required
            />
            <TextField
              label="Dirección"
              value={formData.address}
              onChange={(e) => setFormData({ ...formData, address: e.target.value })}
              fullWidth
            />
            <TextField
              label="Teléfono"
              value={formData.phone}
              onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
              fullWidth
            />
            <TextField
              label="Email"
              type="email"
              value={formData.email}
              onChange={(e) => setFormData({ ...formData, email: e.target.value })}
              fullWidth
            />
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

export default CompaniesPage
