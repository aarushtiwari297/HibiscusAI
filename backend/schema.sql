-- ==============================================================================
-- Hibiscus Leaf Disease Detection — MySQL Database Schema
-- Database Name: hibiscus_ai
-- DBS Academic Project
-- ==============================================================================

-- 1. Create database if it does not exist
CREATE DATABASE IF NOT EXISTS hibiscus_ai;
USE hibiscus_ai;

-- 2. Create scan_history table for persisting leaf disease diagnoses
CREATE TABLE IF NOT EXISTS scan_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    predicted_class VARCHAR(50) NOT NULL,
    raw_score DECIMAL(6, 4) NOT NULL,
    threshold DECIMAL(6, 4) NOT NULL,
    threshold_status VARCHAR(50) NOT NULL,
    scan_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_timestamp (scan_timestamp DESC),
    INDEX idx_class (predicted_class)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Sample query: Retrieve 10 most recent scans
-- SELECT id, filename, predicted_class, raw_score, threshold, threshold_status, scan_timestamp 
-- FROM scan_history 
-- ORDER BY scan_timestamp DESC 
-- LIMIT 10;
