/**
 * Frontend Configuration Module
 * Provides backend API endpoints with a safe local development default.
 */

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';
