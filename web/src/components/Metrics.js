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
  IconButton,
  Typography,
  Card,
  CardContent,
  CardActions,
  Box,
  Alert,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import UploadIcon from '@mui/icons-material/Upload';

const API_BASE = 'http://127.0.0.1:8000';

function Metrics() {
  const [metrics, setMetrics] = useState([]);
  const [contexts, setContexts] = useState([]);
  const [contextId, setContextId] = useState(localStorage.getItem('contextId') || '');
  const [minValue, setMinValue] = useState('');
  const [maxValue, setMaxValue] = useState('');
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [newContextName, setNewContextName] = useState('');

  useEffect(() => {
    fetchContexts();
  }, []);

  useEffect(() => {
    if (contextId) {
      fetchMetrics();
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

  const fetchMetrics = async () => {
    try {
      const response = await axios.get(`${API_BASE}/metrics/?context_name=${contextId}`);
      setMetrics(response.data);
    } catch (error) {
      setMessage('Error fetching metrics');
    }
  };

  const createContext = async () => {
    if (!newContextName.trim()) return;
    try {
      await axios.post(`${API_BASE}/contexts/`, { name: newContextName });
      setMessage('Context created');
      setNewContextName('');
      fetchContexts();
    } catch (error) {
      // Handle Pydantic validation errors
      if (error.response?.data?.detail) {
        const detail = error.response.data.detail;
        if (Array.isArray(detail)) {
          setMessage(detail.map(err => `${err.loc.join('.')}: ${err.msg}`).join(', '));
        } else if (typeof detail === 'string') {
          setMessage(detail);
        } else {
          setMessage('Error creating context');
        }
      } else {
        setMessage(error.message || 'Error creating context');
      }
    }
  };

  const handleUpload = async () => {
    if (!file || !minValue || !maxValue) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('context_name', contextId);
    formData.append('min_value', minValue);
    formData.append('max_value', maxValue);
    try {
      await axios.post(`${API_BASE}/metrics/upload`, formData);
      setMessage('Upload successful');
      fetchMetrics();
      fetchContexts();
      setFile(null);
      setMinValue('');
      setMaxValue('');
    } catch (error) {
      // Handle Pydantic validation errors
      if (error.response?.data?.detail) {
        const detail = error.response.data.detail;
        if (Array.isArray(detail)) {
          setMessage(detail.map(err => `${err.loc.join('.')}: ${err.msg}`).join(', '));
        } else if (typeof detail === 'string') {
          setMessage(detail);
        } else {
          setMessage('Upload failed');
        }
      } else {
        setMessage(error.message || 'Upload failed');
      }
    }
    setLoading(false);
  };

  const handleDelete = async (id) => {
    try {
      await axios.delete(`${API_BASE}/metrics/${id}`);
      setMessage('Deleted successfully');
      fetchMetrics();
    } catch (error) {
      setMessage('Delete failed');
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Metrics Management
      </Typography>
      {message && <Alert severity="info" sx={{ mb: 2 }}>{message}</Alert>}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6">Create New Context</Typography>
          <TextField
            label="New Context Name"
            value={newContextName}
            onChange={(e) => setNewContextName(e.target.value)}
            fullWidth
            margin="normal"
          />
        </CardContent>
        <CardActions>
          <Button onClick={createContext} variant="outlined">
            Create Context
          </Button>
        </CardActions>
      </Card>
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6">Upload Timeseries Data</Typography>
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
          <TextField
            label="Min Value"
            type="number"
            value={minValue}
            onChange={(e) => setMinValue(e.target.value)}
            fullWidth
            margin="normal"
          />
          <TextField
            label="Max Value"
            type="number"
            value={maxValue}
            onChange={(e) => setMaxValue(e.target.value)}
            fullWidth
            margin="normal"
          />
          <Button variant="contained" component="label" startIcon={<UploadIcon />}>
            Select File
            <input type="file" hidden onChange={(e) => setFile(e.target.files[0])} />
          </Button>
          {file && <Typography variant="body2">{file.name}</Typography>}
        </CardContent>
        <CardActions>
          <Button onClick={handleUpload} disabled={!file || !minValue || !maxValue || loading} variant="contained">
            {loading ? 'Uploading...' : 'Upload'}
          </Button>
        </CardActions>
      </Card>
      <Typography variant="h6" gutterBottom>
        Uploaded Metrics
      </Typography>
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>ID</TableCell>
              <TableCell>Filename</TableCell>
              <TableCell>Min Value</TableCell>
              <TableCell>Max Value</TableCell>
              <TableCell>Avg Value</TableCell>
              <TableCell>Data Points</TableCell>
              <TableCell>Uploaded At</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {metrics.map((metric) => (
              <TableRow key={metric.id}>
                <TableCell>{metric.id}</TableCell>
                <TableCell>{metric.filename}</TableCell>
                <TableCell>{metric.min_value !== null ? metric.min_value.toFixed(2) : 'N/A'}</TableCell>
                <TableCell>{metric.max_value !== null ? metric.max_value.toFixed(2) : 'N/A'}</TableCell>
                <TableCell>{metric.avg_value !== null ? metric.avg_value.toFixed(2) : 'N/A'}</TableCell>
                <TableCell>{metric.data_points}</TableCell>
                <TableCell>{new Date(metric.uploaded_at).toLocaleString()}</TableCell>
                <TableCell>
                  <IconButton onClick={() => handleDelete(metric.id)}>
                    <DeleteIcon />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}

export default Metrics;