import React, { useState, useEffect, useRef } from 'react';
import { API_BASE_URL } from './config';

const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp'];
const MAX_SIZE_MB = 16;

export default function App() {
  // Navigation & View state
  const [activeTab, setActiveTab] = useState('diagnose'); // 'diagnose' | 'history'

  // Diagnostic state
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [fileDetails, setFileDetails] = useState(null);
  const [isDragActive, setIsDragActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);
  const [predictionResult, setPredictionResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  // System & Database Health
  const [backendStatus, setBackendStatus] = useState('checking'); // 'online' | 'offline' | 'checking'
  const [modelReady, setModelReady] = useState(false);
  const [geminiConfigured, setGeminiConfigured] = useState(false);
  const [dbConnected, setDbConnected] = useState(false);
  const [dbStatusMsg, setDbStatusMsg] = useState('');

  // Stage 2: Gemini Visual Symptom Analysis State
  const [symptomResult, setSymptomResult] = useState(null);
  const [symptomLoading, setSymptomLoading] = useState(false);
  const [symptomError, setSymptomError] = useState(null);

  // History state
  const [historyRecords, setHistoryRecords] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState(null);

  const fileInputRef = useRef(null);

  // 1. Health check & History fetch
  const checkHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/health`);
      let data = null;
      try {
        data = await response.json();
      } catch {
        // Non-JSON response
      }

      if (response.ok && data) {
        setBackendStatus('online');
        setModelReady(data.model_loaded === true);
        setGeminiConfigured(Boolean(data.gemini_configured));
        setDbConnected(data.database_connected === true);
        setDbStatusMsg(data.database_status || '');
        if (data.database_connected) {
          fetchHistory();
        }
      } else if (response.status === 503 && data) {
        // Backend HTTP server is reachable, but model is not ready or service is degraded
        setBackendStatus(data.model_loaded ? 'online' : 'degraded');
        setModelReady(data.model_loaded === true);
        setGeminiConfigured(Boolean(data.gemini_configured));
        setDbConnected(data.database_connected === true);
        setDbStatusMsg(data.database_status || 'Service degraded.');
      } else if (data) {
        setBackendStatus(data.model_loaded ? 'online' : 'degraded');
        setModelReady(Boolean(data.model_loaded));
        setGeminiConfigured(Boolean(data.gemini_configured));
        setDbConnected(Boolean(data.database_connected));
        setDbStatusMsg(data.database_status || 'Service degraded.');
      } else {
        setBackendStatus('offline');
        setModelReady(false);
        setGeminiConfigured(false);
        setDbConnected(false);
      }
    } catch {
      // Network error, connection refused, or CORS failure - server is truly unreachable
      setBackendStatus('offline');
      setModelReady(false);
      setGeminiConfigured(false);
      setDbConnected(false);
    }
  };

  const fetchHistory = async () => {
    setHistoryLoading(true);
    setHistoryError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/api/history?limit=25`);
      const data = await res.json();
      if (res.ok && data.success) {
        setHistoryRecords(data.records || []);
      } else {
        setHistoryRecords([]);
        setHistoryError(data.details || data.error || 'MySQL database is currently unreachable.');
      }
    } catch (err) {
      setHistoryRecords([]);
      setHistoryError('Failed to communicate with history service: ' + err.message);
    } finally {
      setHistoryLoading(false);
    }
  };

  useEffect(() => {
    checkHealth();
    const timer = setInterval(checkHealth, 30000);
    return () => clearInterval(timer);
  }, []);

  // Cycle loading steps during inference
  useEffect(() => {
    let interval;
    if (isLoading) {
      setLoadingStep(0);
      const steps = [
        "Validating RGB dimensions & file integrity...",
        "Executing Keras 3 forward pass (MobileNetV2)...",
        "Evaluating provisional decision threshold (0.85)..."
      ];
      interval = setInterval(() => {
        setLoadingStep((prev) => (prev + 1) % steps.length);
      }, 700);
    }
    return () => clearInterval(interval);
  }, [isLoading]);

  // 2. Validate and process selected file
  const handleFileSelection = (file) => {
    setErrorMessage(null);
    setPredictionResult(null);

    if (!file) return;

    const ext = file.name.split('.').pop()?.toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext) && !ALLOWED_TYPES.includes(file.type)) {
      setErrorMessage(`Unsupported format (.${ext}). Please select a JPG, JPEG, PNG, or WEBP image.`);
      return;
    }

    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      setErrorMessage(`Image exceeds the ${MAX_SIZE_MB}MB size limit.`);
      return;
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    const objectUrl = URL.createObjectURL(file);
    setSelectedFile(file);
    setPreviewUrl(objectUrl);

    const img = new Image();
    img.src = objectUrl;
    img.onload = () => {
      setFileDetails({
        name: file.name,
        size: (file.size / (1024 * 1024)).toFixed(2) + ' MB',
        width: img.width,
        height: img.height,
        type: ext?.toUpperCase() || 'IMAGE'
      });
    };
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  };

  const handleClearSelection = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setFileDetails(null);
    setPredictionResult(null);
    setErrorMessage(null);
    setSymptomResult(null);
    setSymptomError(null);
    setSymptomLoading(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // 3. Submit image for prediction & Gemini symptom classification
  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setErrorMessage(null);
    setPredictionResult(null);

    setSymptomLoading(true);
    setSymptomError(null);
    setSymptomResult(null);

    // Track formData for both requests
    const predictFormData = new FormData();
    predictFormData.append('image', selectedFile);

    const symptomFormData = new FormData();
    symptomFormData.append('image', selectedFile);

    // Call 1: Primary Keras 3 Binary Prediction (preserved contract)
    const runPredict = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
          method: 'POST',
          body: predictFormData,
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
          throw new Error(data.error || 'Failed to complete leaf disease diagnosis.');
        }

        setPredictionResult(data);

        // Refresh scan history if database is active
        if (data.saved_to_db) {
          fetchHistory();
        }
      } catch (err) {
        setErrorMessage(
          err.message || 'Unable to communicate with the ML backend. Ensure the Flask server is running.'
        );
      } finally {
        setIsLoading(false);
      }
    };

    // Call 2: Stage 2 Gemini Visual Symptom Classification
    const runSymptomClassification = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/classify-symptom`, {
          method: 'POST',
          body: symptomFormData,
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
          setSymptomError(data.error || 'Gemini visual symptom analysis was unavailable.');
        } else {
          setSymptomResult(data);
        }
      } catch (err) {
        setSymptomError(
          'Could not reach Gemini symptom analysis service. Ensure backend is running.'
        );
      } finally {
        setSymptomLoading(false);
      }
    };

    // Run both concurrently: Keras model finishes and displays immediately,
    // while Gemini visual analysis streams its structured result into the symptom card.
    await Promise.allSettled([runPredict(), runSymptomClassification()]);
  };

  // Re-run Gemini symptom classification independently if needed
  const handleRetrySymptom = async () => {
    if (!selectedFile) return;
    setSymptomLoading(true);
    setSymptomError(null);
    try {
      const formData = new FormData();
      formData.append('image', selectedFile);
      const response = await fetch(`${API_BASE_URL}/api/classify-symptom`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (!response.ok || !data.success) {
        setSymptomError(data.error || 'Gemini visual symptom analysis was unavailable.');
      } else {
        setSymptomResult(data);
      }
    } catch {
      setSymptomError('Could not reach Gemini symptom analysis service.');
    } finally {
      setSymptomLoading(false);
    }
  };

  const getCategoryClass = (category) => {
    switch (category) {
      case 'Healthy':
        return 'category-healthy';
      case 'Chewed':
        return 'category-chewed';
      case 'Dried':
        return 'category-dried';
      case 'Leaf Spot':
        return 'category-leaf-spot';
      case 'Yellowing':
        return 'category-yellowing';
      case 'Other / Uncertain':
      default:
        return 'category-uncertain';
    }
  };


  const loadingMessages = [
    "Resizing & formatting 224×224 RGB tensor...",
    "Running Keras 3 forward pass via PyTorch engine...",
    "Applying provisional empirical threshold (0.85)..."
  ];

  return (
    <div className="app-container">
      {/* Navbar Header */}
      <header className="navbar">
        <div className="navbar-inner">
          <div className="brand-wrapper">
            <div className="brand-icon">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/>
                <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>
              </svg>
            </div>
            <div>
              <div className="brand-title">
                HibiscusAI
                <span className="brand-badge">DBS Project</span>
              </div>
              <div className="brand-subtitle">Leaf Disease Diagnostic System</div>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="nav-tabs">
            <button
              className={`nav-tab-btn ${activeTab === 'diagnose' ? 'active' : ''}`}
              onClick={() => setActiveTab('diagnose')}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                <circle cx="8.5" cy="8.5" r="1.5"/>
                <polyline points="21 15 16 10 5 21"/>
              </svg>
              Diagnose Leaf
            </button>
            <button
              className={`nav-tab-btn ${activeTab === 'history' ? 'active' : ''}`}
              onClick={() => { setActiveTab('history'); fetchHistory(); }}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10"/>
                <polyline points="12 6 12 12 16 14"/>
              </svg>
              SQL Scan History
              {historyRecords.length > 0 && (
                <span className="nav-counter-pill">{historyRecords.length}</span>
              )}
            </button>
          </div>

          {/* Status Badges Group */}
          <div className="status-group">
            {/* Backend ML Engine Status */}
            <div
              className={`status-pill ${
                backendStatus === 'online' && modelReady
                  ? 'online'
                  : backendStatus === 'degraded' || (backendStatus === 'online' && !modelReady)
                  ? 'degraded'
                  : backendStatus === 'checking'
                  ? 'checking'
                  : 'offline'
              }`}
              title={`API URL: ${API_BASE_URL}`}
            >
              <span className="status-dot"></span>
              <span>
                {backendStatus === 'checking'
                  ? 'Connecting...'
                  : backendStatus === 'online' && modelReady
                  ? 'ML Engine Ready'
                  : backendStatus === 'online' || backendStatus === 'degraded'
                  ? 'ML Degraded'
                  : 'Backend Offline'}
              </span>
            </div>

            {/* MySQL Database Status */}
            <div className={`status-pill ${dbConnected ? 'online' : 'offline'}`} title={dbStatusMsg || 'MySQL Port 3306'}>
              <span className="status-dot"></span>
              <span>
                {dbConnected ? 'MySQL Connected' : 'MySQL Offline'}
              </span>
            </div>

            {/* Gemini AI Status */}
            <div
              className={`status-pill ${geminiConfigured ? 'online' : 'checking'}`}
              title={geminiConfigured ? 'Gemini Visual Analysis Configured' : 'GEMINI_API_KEY unconfigured (backend .env)'}
            >
              <span className="status-dot"></span>
              <span>
                {geminiConfigured ? 'Gemini Ready' : 'Gemini (No Key)'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="main-content">
        {/* =========================================================================
            TAB 1: DIAGNOSE LEAF VIEW
           ========================================================================= */}
        {activeTab === 'diagnose' && (
          <>
            <div className="hero-header">
              <div className="hero-pill">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
                </svg>
                MobileNetV2 CNN • MySQL Persistent History
              </div>
              <h1 className="hero-title">Hibiscus Leaf Disease Detection</h1>
              <p className="hero-desc">
                Upload a hibiscus leaf photograph to diagnose foliage pathology and persist diagnostic records to relational MySQL storage.
              </p>
            </div>

            {/* Global Error Banner */}
            {errorMessage && (
              <div className="error-box">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="8" x2="12" y2="12"/>
                    <line x1="12" y1="16" x2="12.01" y2="16"/>
                  </svg>
                  <span>{errorMessage}</span>
                </div>
                <button className="btn btn-secondary" style={{ padding: '0.35rem 0.75rem', fontSize: '0.8rem' }} onClick={() => setErrorMessage(null)}>
                  Dismiss
                </button>
              </div>
            )}

            {/* 1. Dropzone State */}
            {!selectedFile && !predictionResult && !isLoading && (
              <div className="card">
                <div
                  className={`dropzone ${isDragActive ? 'drag-active' : ''}`}
                  onDragOver={(e) => { e.preventDefault(); setIsDragActive(true); }}
                  onDragLeave={() => setIsDragActive(false)}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".jpg,.jpeg,.png,.webp"
                    style={{ display: 'none' }}
                    onChange={(e) => handleFileSelection(e.target.files?.[0])}
                  />
                  <div className="upload-icon-circle">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2 2v-4"/>
                      <polyline points="17 8 12 3 7 8"/>
                      <line x1="12" y1="3" x2="12" y2="15"/>
                    </svg>
                  </div>
                  <h3 className="dropzone-heading">Drop your hibiscus leaf photo here</h3>
                  <p className="dropzone-subtext">or click to browse your computer</p>
                  <div className="format-tags">
                    <span className="format-tag">JPG</span>
                    <span className="format-tag">JPEG</span>
                    <span className="format-tag">PNG</span>
                    <span className="format-tag">WEBP</span>
                    <span className="format-tag">Up to 16MB</span>
                  </div>
                </div>
              </div>
            )}

            {/* 2. Selected File Preview State */}
            {selectedFile && !predictionResult && !isLoading && (
              <div className="card">
                <div className="preview-card">
                  <div className="preview-thumb-container">
                    <img src={previewUrl} alt="Leaf Preview" className="preview-thumb" />
                    <button className="btn-remove-thumb" title="Remove image" onClick={handleClearSelection}>
                      ✕
                    </button>
                  </div>

                  <div>
                    <h3 className="preview-meta-title">{fileDetails?.name || 'Selected Leaf Photo'}</h3>
                    <div className="preview-meta-specs">
                      <span>{fileDetails?.type}</span>
                      <span>•</span>
                      <span>{fileDetails?.size}</span>
                      {fileDetails?.width && (
                        <>
                          <span>•</span>
                          <span>{fileDetails.width} × {fileDetails.height} px</span>
                        </>
                      )}
                    </div>

                    <div className="preview-actions">
                      <button
                        className="btn btn-primary"
                        onClick={handleAnalyze}
                        disabled={backendStatus !== 'online' || !modelReady}
                      >
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
                        </svg>
                        Analyze Leaf
                      </button>
                      <button className="btn btn-secondary" onClick={() => fileInputRef.current?.click()}>
                        Change Photo
                      </button>
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept=".jpg,.jpeg,.png,.webp"
                        style={{ display: 'none' }}
                        onChange={(e) => handleFileSelection(e.target.files?.[0])}
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* 3. Scanning Animation State */}
            {isLoading && (
              <div className="card scanning-card">
                <div className="scanner-image-wrapper">
                  <img src={previewUrl} alt="Scanning" className="scanner-image" />
                  <div className="scanner-beam"></div>
                </div>
                <div className="scanning-spinner"></div>
                <div className="scanning-text">Analyzing Leaf Pathology</div>
                <div className="scanning-subtext">{loadingMessages[loadingStep]}</div>
              </div>
            )}

            {/* 4. Diagnostic Result Dashboard */}
            {predictionResult && !isLoading && (
              <div className="card">
                {/* Top Verdict Banner */}
                <div className={`result-banner ${predictionResult.prediction === 'Fresh' ? 'fresh' : 'diseased'}`}>
                  <div className="result-banner-left">
                    <div className="result-status-icon">
                      {predictionResult.prediction === 'Fresh' ? (
                        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                          <path d="M20 6L9 17l-5-5"/>
                        </svg>
                      ) : (
                        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                          <circle cx="12" cy="12" r="10"/>
                          <line x1="12" y1="8" x2="12" y2="12"/>
                          <line x1="12" y1="16" x2="12.01" y2="16"/>
                        </svg>
                      )}
                    </div>
                    <div>
                      <div className="result-class-headline">
                        {predictionResult.prediction === 'Fresh' ? 'Fresh / Healthy Foliage' : 'Diseased Foliage Detected'}
                      </div>
                      <div className="result-tagline">
                        Classification Decision at Threshold {predictionResult.threshold}
                      </div>
                    </div>
                  </div>

                  <button className="btn btn-secondary" onClick={handleClearSelection}>
                    Analyze Another Leaf
                  </button>
                </div>

                {/* Database Persistence Status Notification */}
                <div className={`db-persist-note ${predictionResult.saved_to_db ? 'saved' : 'unsaved'}`}>
                  {predictionResult.saved_to_db ? (
                    <>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--primary-600)" strokeWidth="2.5">
                        <path d="M20 6L9 17l-5-5"/>
                      </svg>
                      <span>
                        <strong>Saved to MySQL:</strong> Diagnostic record successfully persisted in table <code>scan_history</code> as <strong>Record #{predictionResult.scan_id}</strong>.
                      </span>
                    </>
                  ) : (
                    <>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#b45309" strokeWidth="2">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="12" y1="8" x2="12" y2="12"/>
                        <line x1="12" y1="16" x2="12.01" y2="16"/>
                      </svg>
                      <span>
                        <strong>Persistence Notice:</strong> Leaf diagnosis succeeded, but could not be recorded in MySQL history because the database server is currently offline.
                      </span>
                    </>
                  )}
                </div>

                {/* Two-Column Diagnostic Layout */}
                <div className="result-grid" style={{ marginTop: '1.25rem' }}>
                  {/* Left Column: Leaf Photo Preview */}
                  <div className="result-image-card">
                    <div className="result-image-box">
                      <img src={previewUrl} alt="Analyzed Leaf" />
                    </div>
                    <div className="result-image-specs">
                      <span>{fileDetails?.name || 'leaf_scan'}</span>
                      <span>{fileDetails?.size || ''}</span>
                    </div>
                  </div>

                  {/* Right Column: Metrics & Care Advice */}
                  <div className="diagnostic-cards">
                    {/* Metric Card */}
                    <div className="metric-panel">
                      <div className="metric-header">
                        <span className="metric-label">Raw Sigmoid Score</span>
                        <span className="metric-score-value">{predictionResult.raw_score?.toFixed(4)}</span>
                      </div>

                      <div className="progress-track" title={`Raw score: ${predictionResult.raw_score}`}>
                        <div
                          className="progress-fill"
                          style={{ width: `${Math.min(100, Math.max(0, predictionResult.raw_score * 100))}%` }}
                        ></div>
                      </div>

                      <div className="threshold-marker-text">
                        <span>0.00 (Diseased)</span>
                        <span style={{ fontWeight: 700, color: 'var(--text-main)' }}>
                          Threshold: {predictionResult.threshold}
                        </span>
                        <span>1.00 (Fresh)</span>
                      </div>
                    </div>

                    {/* Visible Provisional Threshold Disclaimer */}
                    <div className="provisional-notice-card">
                      <div className="provisional-notice-header">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                          <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                          <line x1="12" y1="9" x2="12" y2="13"/>
                          <line x1="12" y1="17" x2="12.01" y2="17"/>
                        </svg>
                        Provisional Threshold Notice ({predictionResult.threshold_status})
                      </div>
                      <div>
                        This decision threshold ({predictionResult.threshold}) is provisional, based on an empirical evaluation of 18 test images. The raw score is an uncalibrated sigmoid output and does not represent an absolute disease probability.
                      </div>
                    </div>

                    {/* Stage 2: Gemini Visual Symptom Category Analysis */}
                    <div className="gemini-card">
                      <div className="gemini-card-header">
                        <div className="gemini-card-header-left">
                          <div className="gemini-sparkle-icon">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                              <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
                            </svg>
                          </div>
                          <div>
                            <div className="gemini-card-title">
                              Gemini Visual Symptom Classification
                            </div>
                            <div className="gemini-card-subtitle">
                              Stage 2 Multimodal AI (Observable Damage Patterns)
                            </div>
                          </div>
                        </div>

                        {symptomResult?.model && (
                          <span className="gemini-model-badge">
                            {symptomResult.model}
                          </span>
                        )}
                      </div>

                      {/* Loading State */}
                      {symptomLoading && (
                        <div className="gemini-status-state loading">
                          <div className="gemini-mini-spinner"></div>
                          <div>
                            <div style={{ fontWeight: 600, fontSize: '0.88rem', color: 'var(--text-main)' }}>
                              Analyzing visual foliage symptoms with Gemini...
                            </div>
                            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                              Categorizing observable patterns into Healthy, Chewed, Dried, Leaf Spot, or Yellowing.
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Error State */}
                      {!symptomLoading && symptomError && (
                        <div className="gemini-status-state error">
                          <div style={{ display: 'flex', gap: '0.65rem', alignItems: 'flex-start', width: '100%' }}>
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#dc2626" strokeWidth="2.2">
                              <circle cx="12" cy="12" r="10"/>
                              <line x1="12" y1="8" x2="12" y2="12"/>
                              <line x1="12" y1="16" x2="12.01" y2="16"/>
                            </svg>
                            <div style={{ flex: 1 }}>
                              <div style={{ fontWeight: 700, fontSize: '0.85rem', color: '#991b1b' }}>
                                Gemini Visual Analysis Unavailable
                              </div>
                              <div style={{ fontSize: '0.8rem', color: '#b91c1c', marginTop: '0.2rem' }}>
                                {symptomError}
                              </div>
                              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>
                                Primary Keras ML binary diagnosis above remains active and operational.
                              </div>
                              <button
                                className="btn btn-secondary"
                                onClick={handleRetrySymptom}
                                style={{ marginTop: '0.6rem', fontSize: '0.78rem', padding: '0.3rem 0.75rem' }}
                              >
                                Retry Gemini Analysis
                              </button>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Success / Result State */}
                      {!symptomLoading && symptomResult && (
                        <div className="gemini-result-body">
                          {/* Suggested Category Badge & Confidence */}
                          <div className="gemini-category-row">
                            <div className="gemini-category-left">
                              <span className="gemini-label">Suggested Symptom Category:</span>
                              <span className={`gemini-category-pill ${getCategoryClass(symptomResult.category)}`}>
                                {symptomResult.category}
                              </span>
                            </div>
                            {symptomResult.confidence && (
                              <span className="gemini-confidence-pill">
                                Confidence: {symptomResult.confidence}
                              </span>
                            )}
                          </div>

                          {/* Uncertainty Warning Alert */}
                          {symptomResult.uncertain && (
                            <div className="gemini-uncertainty-alert">
                              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#d97706" strokeWidth="2.2">
                                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                                <line x1="12" y1="9" x2="12" y2="13"/>
                                <line x1="12" y1="17" x2="12.01" y2="17"/>
                              </svg>
                              <div>
                                <strong>Visual Uncertainty Notice:</strong> The model flagged uncertainty due to image resolution, blur, lighting, or ambiguous damage patterns. Do not treat as a conclusive assessment.
                              </div>
                            </div>
                          )}

                          {/* Visible Evidence Chips */}
                          {symptomResult.visible_symptoms && symptomResult.visible_symptoms.length > 0 && (
                            <div className="gemini-evidence-box">
                              <div className="gemini-evidence-heading">Visible Evidence Observed:</div>
                              <div className="gemini-chips">
                                {symptomResult.visible_symptoms.map((symptom, idx) => (
                                  <span key={idx} className="gemini-chip">
                                    • {symptom}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Short Explanation */}
                          {symptomResult.explanation && (
                            <div className="gemini-explanation">
                              <strong>Analysis:</strong> {symptomResult.explanation}
                            </div>
                          )}

                          {/* Disclaimer */}
                          <div className="gemini-disclaimer">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                              <circle cx="12" cy="12" r="10"/>
                              <line x1="12" y1="16" x2="12" y2="12"/>
                              <line x1="12" y1="8" x2="12.01" y2="8"/>
                            </svg>
                            <span>
                              {symptomResult.disclaimer || "Gemini-assisted analysis screens visible damage patterns only and is separate from the Keras binary prediction."}
                            </span>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Botanical Care Recommendations */}
                    <div className="care-card">
                      <div className="care-card-title">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--primary-600)" strokeWidth="2">
                          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                        </svg>
                        Botanical Care Guidance
                      </div>
                      {predictionResult.prediction === 'Fresh' ? (
                        <ul className="care-list">
                          <li>Maintain regular hydration: irrigate soil directly, avoiding leaves.</li>
                          <li>Ensure adequate 6–8 hours of daily direct sunlight exposure.</li>
                          <li>Apply balanced organic plant food to support robust hibiscus blooms.</li>
                        </ul>
                      ) : (
                        <ul className="care-list">
                          <li>Quarantine plant to prevent spores spreading to neighboring foliage.</li>
                          <li>Prune severely infected leaves using sanitized shears and dispose safely.</li>
                          <li>Apply cold-pressed organic neem oil spray or copper fungicide during morning hours.</li>
                          <li>Ensure proper spacing between branches for healthy air circulation.</li>
                        </ul>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </>
        )}

        {/* =========================================================================
            TAB 2: SQL SCAN HISTORY VIEW
           ========================================================================= */}
        {activeTab === 'history' && (
          <div className="history-section">
            <div className="history-header-bar">
              <div>
                <h2 className="history-title">Relational Database Scan History</h2>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                  Persistent diagnosis records retrieved directly from table <code>scan_history</code> in database <code>hibiscus_ai</code>.
                </p>
              </div>

              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button
                  className="btn btn-secondary"
                  onClick={fetchHistory}
                  disabled={historyLoading}
                  style={{ fontSize: '0.85rem', padding: '0.45rem 0.95rem' }}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="23 4 23 10 17 10"/>
                    <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
                  </svg>
                  {historyLoading ? 'Refreshing...' : 'Refresh Records'}
                </button>
              </div>
            </div>

            {/* State A: MySQL Offline Warning & Setup Guide */}
            {!dbConnected && (
              <div className="mysql-guide-card">
                <div className="mysql-guide-title">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="8" x2="12" y2="12"/>
                    <line x1="12" y1="16" x2="12.01" y2="16"/>
                  </svg>
                  MySQL Database Is Currently Offline
                </div>
                <p>
                  The backend could not establish a connection to MySQL at <code>127.0.0.1:3306</code>. Leaf prediction continues to function smoothly in memory, but persistent scan history requires an active MySQL server.
                </p>
                <div style={{ marginTop: '0.75rem', fontWeight: 700 }}>
                  Prerequisites to connect MySQL for DBS Viva / Evaluation:
                </div>
                <ol className="mysql-guide-steps">
                  <li>
                    Start <strong>MySQL</strong> in your <strong>XAMPP Control Panel</strong> (or launch your local MySQL Server service).
                  </li>
                  <li>
                    Execute the database DDL script in <span className="mysql-code-pill">backend/schema.sql</span> to initialize database <code>hibiscus_ai</code> and table <code>scan_history</code>.
                  </li>
                  <li>
                    Ensure credentials in <span className="mysql-code-pill">backend/.env</span> match your database user (default: <code>root</code> with blank password).
                  </li>
                  <li>
                    Click the <strong>Refresh Records</strong> button above to re-test the connection.
                  </li>
                </ol>
              </div>
            )}

            {/* State B: Connected but No Records */}
            {dbConnected && !historyLoading && historyRecords.length === 0 && (
              <div className="history-empty-card">
                <div className="history-empty-icon">
                  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="8" x2="12" y2="12"/>
                    <line x1="12" y1="16" x2="12.01" y2="16"/>
                  </svg>
                </div>
                <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: 'var(--text-main)', marginBottom: '0.35rem' }}>
                  No Leaf Scans Recorded Yet
                </h3>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.25rem' }}>
                  The <code>scan_history</code> table is connected and ready. Analyze a leaf photo in the Diagnose tab to create your first relational history record.
                </p>
                <button className="btn btn-primary" onClick={() => setActiveTab('diagnose')}>
                  Start First Diagnosis
                </button>
              </div>
            )}

            {/* State C: Data Table with Records */}
            {dbConnected && historyRecords.length > 0 && (
              <div className="history-table-wrapper">
                <table className="history-table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Timestamp</th>
                      <th>Filename</th>
                      <th>Predicted Class</th>
                      <th>Sigmoid Score</th>
                      <th>Threshold</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {historyRecords.map((item) => (
                      <tr key={item.id}>
                        <td>
                          <span className="badge-scan-id">#{item.id}</span>
                        </td>
                        <td style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                          {item.scan_timestamp || '—'}
                        </td>
                        <td style={{ fontWeight: 600 }}>
                          {item.filename}
                        </td>
                        <td>
                          <span className={`badge-class ${item.predicted_class?.toLowerCase() === 'fresh' ? 'fresh' : 'diseased'}`}>
                            {item.predicted_class === 'Fresh' ? '🌿 Fresh' : '⚠️ Diseased'}
                          </span>
                        </td>
                        <td>
                          <span className="badge-score">{item.raw_score.toFixed(4)}</span>
                        </td>
                        <td style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                          {item.threshold.toFixed(2)}
                        </td>
                        <td>
                          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'capitalize' }}>
                            {item.threshold_status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        DBS Course Project — Hibiscus Leaf Disease Detection • MobileNetV2 Keras 3 • MySQL History
      </footer>
    </div>
  );
}
