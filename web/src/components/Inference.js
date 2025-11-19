import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Typography,
  Card,
  CardContent,
  CardActions,
  Box,
  Alert,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';

const API_BASE = 'http://127.0.0.1:8000';

function Inference() {
  const [contexts, setContexts] = useState([]);
  const [models, setModels] = useState([]);
  const [contextId, setContextId] = useState(localStorage.getItem('contextId') || '');
  const [selectedModel, setSelectedModel] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
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

  const handleRun = async () => {
    if (!selectedModel) return;
    setLoading(true);
    setMessage('Running inference...');
    try {
      const response = await axios.post(`${API_BASE}/inference/`, {
        context_name: contextId,
        model_id: selectedModel,
      });
      setResult(response.data);
      setMessage('Inference completed');
    } catch (error) {
      setMessage('Inference failed');
    }
    setLoading(false);
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Run Inference
      </Typography>
      {message && <Alert severity="info" sx={{ mb: 2 }}>{message}</Alert>}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6">Select Context and Model</Typography>
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
          <FormControl fullWidth margin="normal">
            <InputLabel>Model</InputLabel>
            <Select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
              {models.map((model) => (
                <MenuItem key={model.id} value={model.id}>
                  {model.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </CardContent>
        <CardActions>
          <Button
            onClick={handleRun}
            disabled={!selectedModel || loading}
            variant="contained"
            startIcon={<PlayArrowIcon />}
          >
            {loading ? 'Running...' : 'Run Inference'}
          </Button>
        </CardActions>
      </Card>
      {result && (
        <Card>
          <CardContent>
            <Typography variant="h6">Predicted Timeseries</Typography>
            {result.result && (() => {
              const predictions = JSON.parse(result.result);
              return Object.entries(predictions).map(([metricName, values]) => (
                <Box key={metricName} sx={{ mb: 3 }}>
                  <Typography variant="subtitle1" sx={{ fontWeight: 'bold', mb: 1 }}>
                    {metricName}
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    Next {values.length} predicted values: {values.slice(0, 10).join(', ')}
                    {values.length > 10 && `... (${values.length - 10} more)`}
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {values.slice(0, 20).map((val, idx) => (
                      <Typography 
                        key={idx} 
                        variant="caption" 
                        sx={{ 
                          bgcolor: 'primary.light', 
                          color: 'primary.contrastText',
                          px: 1, 
                          py: 0.5, 
                          borderRadius: 1,
                          fontFamily: 'monospace'
                        }}
                      >
                        {val.toFixed(3)}
                      </Typography>
                    ))}
                  </Box>
                </Box>
              ));
            })()}
            <Typography variant="body2" sx={{ mt: 2, color: 'text.secondary' }}>
              Result ID: {result.id} | Created: {new Date(result.created_at).toLocaleString()}
            </Typography>
          </CardContent>
        </Card>
      )}
    </Box>
  );
}

export default Inference;