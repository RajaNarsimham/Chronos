import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Button,
  TextField,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Typography,
  Card,
  CardContent,
  CardActions,
  Box,
  Alert,
  LinearProgress,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';

const API_BASE = 'http://127.0.0.1:8000';

function Models() {
  const [models, setModels] = useState([]);
  const [contexts, setContexts] = useState([]);
  const [contextId, setContextId] = useState(localStorage.getItem('contextId') || '');
  const [training, setTraining] = useState(false);
  const [message, setMessage] = useState('');

  useEffect(() => {
    fetchContexts();
  }, []);

  useEffect(() => {
    if (contextId) {
      fetchModels();
    }
  }, [contextId]);

  useEffect(() => {
    localStorage.setItem('contextId', contextId);
  }, [contextId]);

  const fetchContexts = async () => {
    try {
      const response = await axios.get(`${API_BASE}/contexts/`);
      setContexts(response.data);
      if (!contextId && response.data.length > 0) {
        setContextId(response.data[0].name);
      }
    } catch (error) {
      setMessage('Error fetching contexts');
    }
  };

  const fetchModels = async () => {
    try {
      const response = await axios.get(`${API_BASE}/models/?context_name=${contextId}`);
      setModels(response.data);
    } catch (error) {
      setMessage('Error fetching models');
    }
  };

  const handleTrain = async () => {
    setTraining(true);
    setMessage('Training started...');
    try {
      await axios.post(`${API_BASE}/models/train/`, { context_name: contextId });
      setMessage('Training completed');
      fetchModels();
    } catch (error) {
      setMessage('Training failed');
    }
    setTraining(false);
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Model Training
      </Typography>
      {message && <Alert severity="info" sx={{ mb: 2 }}>{message}</Alert>}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6">Train a New Model</Typography>
          <FormControl fullWidth margin="normal">
            <InputLabel>Context Name</InputLabel>
            <Select
              value={contextId}
              onChange={(e) => setContextId(e.target.value)}
              label="Context Name"
            >
              {contexts.map((ctx) => (
                <MenuItem key={ctx.id} value={ctx.name}>{ctx.name}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </CardContent>
        <CardActions>
          <Button
            onClick={handleTrain}
            disabled={training}
            variant="contained"
            startIcon={<PlayArrowIcon />}
          >
            {training ? 'Training...' : 'Start Training'}
          </Button>
        </CardActions>
        {training && <LinearProgress />}
      </Card>
      <Typography variant="h6" gutterBottom>
        Trained Models
      </Typography>
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>ID</TableCell>
              <TableCell>Name</TableCell>
              <TableCell>Accuracy</TableCell>
              <TableCell>Trained At</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {models.map((model) => (
              <TableRow key={model.id}>
                <TableCell>{model.id}</TableCell>
                <TableCell>{model.name}</TableCell>
                <TableCell>{model.accuracy}</TableCell>
                <TableCell>{new Date(model.trained_at).toLocaleString()}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}

export default Models;