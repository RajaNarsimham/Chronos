import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Box from '@mui/material/Box';
import Metrics from './components/Metrics';
import Models from './components/Models';
import Inference from './components/Inference';
import Results from './components/Results';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6">Chronos</Typography>
          </Toolbar>
        </AppBar>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs>
            <Tab label="Metrics" component={Link} to="/metrics" />
            <Tab label="Models" component={Link} to="/models" />
            <Tab label="Inference" component={Link} to="/inference" />
            <Tab label="Results" component={Link} to="/results" />
          </Tabs>
        </Box>
        <Container maxWidth="lg" sx={{ mt: 4 }}>
          <Routes>
            <Route path="/metrics" element={<Metrics />} />
            <Route path="/models" element={<Models />} />
            <Route path="/inference" element={<Inference />} />
            <Route path="/results" element={<Results />} />
            <Route path="/" element={<Metrics />} />
          </Routes>
        </Container>
      </Router>
    </ThemeProvider>
  );
}

export default App;
