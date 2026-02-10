import { createTheme } from '@mui/material/styles'

// Paleta de colores pastel moderna
export const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#90caf9', // Azul pastel
      light: '#b3e5fc',
      dark: '#64b5f6',
      contrastText: '#1a237e',
    },
    secondary: {
      main: '#f8bbd0', // Rosa pastel
      light: '#fce4ec',
      dark: '#f48fb1',
      contrastText: '#880e4f',
    },
    success: {
      main: '#a5d6a7', // Verde pastel
      light: '#c8e6c9',
      dark: '#81c784',
    },
    warning: {
      main: '#ffcc80', // Naranja pastel
      light: '#ffe0b2',
      dark: '#ffb74d',
    },
    error: {
      main: '#ef9a9a', // Rojo pastel
      light: '#ffcdd2',
      dark: '#e57373',
    },
    info: {
      main: '#b39ddb', // Púrpura pastel
      light: '#d1c4e9',
      dark: '#9575cd',
    },
    background: {
      default: '#f5f7fa',
      paper: '#ffffff',
    },
    text: {
      primary: '#37474f',
      secondary: '#78909c',
    },
  },
  typography: {
    fontFamily: [
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
    h1: {
      fontWeight: 600,
      fontSize: '2.5rem',
      letterSpacing: '-0.01562em',
    },
    h2: {
      fontWeight: 600,
      fontSize: '2rem',
      letterSpacing: '-0.00833em',
    },
    h3: {
      fontWeight: 600,
      fontSize: '1.75rem',
      letterSpacing: '0em',
    },
    h4: {
      fontWeight: 600,
      fontSize: '1.5rem',
      letterSpacing: '0.00735em',
    },
    h5: {
      fontWeight: 600,
      fontSize: '1.25rem',
      letterSpacing: '0em',
    },
    h6: {
      fontWeight: 600,
      fontSize: '1rem',
      letterSpacing: '0.0075em',
    },
    button: {
      textTransform: 'none',
      fontWeight: 500,
    },
  },
  shape: {
    borderRadius: 16, // Bordes redondeados por defecto
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          padding: '10px 24px',
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0 4px 8px rgba(0,0,0,0.1)',
          },
        },
        contained: {
          '&:hover': {
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 20,
          boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
          '&:hover': {
            boxShadow: '0 6px 16px rgba(0,0,0,0.12)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
        },
        elevation1: {
          boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
        },
        elevation2: {
          boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 12,
            '&:hover fieldset': {
              borderColor: '#90caf9',
            },
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 10,
          fontWeight: 500,
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          borderRadius: 0,
          borderRight: 'none',
        },
      },
    },
    MuiIconButton: {
      styleOverrides: {
        root: {
          borderRadius: 10,
          '&:hover': {
            backgroundColor: 'rgba(144, 202, 249, 0.1)',
          },
        },
      },
    },
  },
})
