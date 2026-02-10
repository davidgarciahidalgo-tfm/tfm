import { useState } from 'react'
import {
  Box,
  Drawer,
  AppBar,
  Toolbar,
  List,
  Typography,
  Divider,
  IconButton,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Avatar,
  Menu,
  MenuItem,
} from '@mui/material'
import {
  Menu as MenuIcon,
  Dashboard,
  Map,
  ShowChart,
  LocationOn,
  DirectionsBike,
  Settings,
  Logout,
  AccountCircle,
  Business,
  People,
} from '@mui/icons-material'
import { Outlet, useNavigate, useLocation } from 'react-router-dom'
import { useAuthStore } from '../stores/authStore'

const drawerWidth = 260

const MainLayout = () => {
  const [mobileOpen, setMobileOpen] = useState(false)
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null)
  const navigate = useNavigate()
  const location = useLocation()
  const { user, logout } = useAuthStore()

  const handleDrawerToggle = () => setMobileOpen(!mobileOpen)
  const handleMenu = (event: React.MouseEvent<HTMLElement>) => setAnchorEl(event.currentTarget)
  const handleClose = () => setAnchorEl(null)
  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  const baseMenuItems = [
    { text: 'Dashboard', icon: <Dashboard />, path: '/dashboard' },
  ]

  const superAdminMenuItems = [
    { text: 'Empresas', icon: <Business />, path: '/companies', roles: ['superadmin'] },
    { text: 'Usuarios', icon: <People />, path: '/users', roles: ['superadmin'] },
  ]

  const operationalMenuItems = [
    { text: 'Mapa', icon: <Map />, path: '/map' },
    { text: 'Predicciones', icon: <ShowChart />, path: '/predictions' },
    { text: 'Estaciones', icon: <LocationOn />, path: '/stations' },
    { text: 'Dispositivos', icon: <DirectionsBike />, path: '/devices' },
  ]

  // Combinar menús según el rol del usuario
  const menuItems = user?.role === 'superadmin'
    ? [...baseMenuItems, ...superAdminMenuItems]
    : [...baseMenuItems, ...operationalMenuItems]

  const drawer = (
    <div>
      <Toolbar sx={{ background: 'linear-gradient(135deg, #90caf9 0%, #b39ddb 100%)', color: 'white' }}>
        <DirectionsBike sx={{ mr: 2 }} />
        <Typography variant="h6" noWrap>
          Mobility
        </Typography>
      </Toolbar>
      <Divider />
      <List sx={{ px: 1, py: 2 }}>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding sx={{ mb: 0.5 }}>
            <ListItemButton
              selected={location.pathname === item.path}
              onClick={() => navigate(item.path)}
              sx={{
                borderRadius: 2,
                '&.Mui-selected': {
                  backgroundColor: 'rgba(144, 202, 249, 0.2)',
                  '&:hover': {
                    backgroundColor: 'rgba(144, 202, 249, 0.3)',
                  },
                },
              }}
            >
              <ListItemIcon sx={{ color: location.pathname === item.path ? 'primary.main' : 'inherit' }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      <Divider />
      <List sx={{ px: 1, py: 1 }}>
        <ListItem disablePadding>
          <ListItemButton onClick={() => navigate('/settings')} sx={{ borderRadius: 2 }}>
            <ListItemIcon>
              <Settings />
            </ListItemIcon>
            <ListItemText primary="Configuración" />
          </ListItemButton>
        </ListItem>
      </List>
    </div>
  )

  return (
    <Box sx={{ display: 'flex', height: '100vh' }}>
      <AppBar
        position="fixed"
        sx={{
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          ml: { sm: `${drawerWidth}px` },
          backgroundColor: 'white',
          color: 'text.primary',
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            {menuItems.find(item => item.path === location.pathname)?.text || 'Mobility Platform'}
          </Typography>
          <IconButton onClick={handleMenu} color="inherit">
            <Avatar sx={{ width: 36, height: 36, bgcolor: 'primary.main' }}>
              {user?.firstName?.charAt(0)}
            </Avatar>
          </IconButton>
          <Menu anchorEl={anchorEl} open={Boolean(anchorEl)} onClose={handleClose}>
            <MenuItem onClick={() => { handleClose(); navigate('/settings'); }}>
              <AccountCircle sx={{ mr: 1 }} />
              Mi Perfil
            </MenuItem>
            <MenuItem onClick={handleLogout}>
              <Logout sx={{ mr: 1 }} />
              Cerrar Sesión
            </MenuItem>
          </Menu>
        </Toolbar>
      </AppBar>

      <Box component="nav" sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}>
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{ keepMounted: true }}
          sx={{
            display: { xs: 'block', sm: 'none' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
        >
          {drawer}
        </Drawer>
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', sm: 'block' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          mt: '64px',
          height: 'calc(100vh - 64px)',
          overflow: 'auto',
          background: 'linear-gradient(135deg, #f5f7fa 0%, #e3f2fd 100%)',
        }}
      >
        <Outlet />
      </Box>
    </Box>
  )
}

export default MainLayout
