import { Box } from '@mui/material'
import { Outlet } from 'react-router-dom'

const AuthLayout = () => {
  return (
    <Box
      sx={{
        display: 'flex',
        minHeight: '100vh',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%)',
      }}
    >
      <Outlet />
    </Box>
  )
}

export default AuthLayout
