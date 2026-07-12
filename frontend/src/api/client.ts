import axios from 'axios';

export const apiClient = axios.create({
  baseURL: '/api',  // Vite proxy handles this
  headers: { 'Content-Type': 'application/json' }
});
