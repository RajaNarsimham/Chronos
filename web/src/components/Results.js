import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  TextField,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Typography,
  Box,
  Alert,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Card,
  CardContent,
  CardActions,
  Button,
} from '@mui/material';

const API_BASE = 'http://127.0.0.1:8000';

function Results() {
  const [results, setResults] = useState([]);
  const [contexts, setContexts] = useState([]);
  const [contextId, setContextId] = useState(localStorage.getItem('contextId') || '');
  const [message, setMessage] = useState('');

  useEffect(() => {
    fetchContexts();
  }, []);

  useEffect(() => {
    if (contextId) {
      fetchResults();
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

  const fetchResults = async () => {
    try {
      const response = await axios.get(`${API_BASE}/results/?context_name=${contextId}`);
      setResults(response.data);
    } catch (error) {
      setMessage('Error fetching results');
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Inference Results
      </Typography>
      {message && <Alert severity="info" sx={{ mb: 2 }}>{message}</Alert>}
      <FormControl fullWidth margin="normal" sx={{ mb: 4 }}>
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
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>ID</TableCell>
              <TableCell>Metric</TableCell>
              <TableCell>Model</TableCell>
              <TableCell>Result</TableCell>
              <TableCell>Run At</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {results.map((res) => (
              <TableRow key={res.id}>
                <TableCell>{res.id}</TableCell>
                <TableCell>{res.metric.filename}</TableCell>
                <TableCell>{res.model.name}</TableCell>
                <TableCell>
                  <pre style={{ maxWidth: 300, overflow: 'auto' }}>
                    {JSON.stringify(JSON.parse(res.result), null, 2)}
                  </pre>
                </TableCell>
                <TableCell>{new Date(res.run_at).toLocaleString()}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}

export default Results;