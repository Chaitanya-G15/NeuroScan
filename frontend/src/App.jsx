import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  AlertTriangle, 
  CheckCircle2, 
  Sliders, 
  Upload, 
  Layers, 
  Brain, 
  Info, 
  RefreshCw,
  Zap,
  ShieldCheck,
  TrendingUp,
  AlertCircle
} from 'lucide-react';
import { 
  ResponsiveContainer, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  CartesianGrid,
  LineChart,
  Line
} from 'recharts';

const API_BASE = "/api/v1";

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [threshold, setThreshold] = useState(0.28);
  const [blurKernel, setBlurKernel] = useState(5);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [demoSamples, setDemoSamples] = useState([]);
  const [systemHealth, setSystemHealth] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [activeTab, setActiveTab] = useState('inspection'); // 'inspection' | 'metrics'

  // Fetch initial system status and demo samples
  useEffect(() => {
    fetchHealth();
    fetchSamples();
    fetchMetrics();
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json();
        setSystemHealth(data);
      }
    } catch (e) {
      console.warn("Backend not running or unreachable:", e);
    }
  };

  const fetchSamples = async () => {
    try {
      const res = await fetch(`${API_BASE}/sample`);
      if (res.ok) {
        const data = await res.json();
        setDemoSamples(data.samples || []);
      }
    } catch (e) {
      console.warn("Could not fetch demo samples:", e);
    }
  };

  const fetchMetrics = async () => {
    try {
      const res = await fetch(`${API_BASE}/metrics`);
      if (res.ok) {
        const data = await res.json();
        setMetrics(data);
      }
    } catch (e) {
      console.warn("Could not fetch metrics:", e);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setAnalysisResult(null);
      setErrorMessage(null);
      // Auto run analysis on upload
      runAnalysis({ uploadedFile: file });
    }
  };

  const handleDemoSelect = (sample) => {
    setSelectedFile(null);
    setPreviewUrl(sample.preview);
    setErrorMessage(null);
    runAnalysis({ sampleId: sample.id });
  };

  const runAnalysis = async (options = {}) => {
    setLoading(true);
    setErrorMessage(null);
    const formData = new FormData();
    formData.append('threshold', threshold.toString());
    formData.append('blur_kernel', blurKernel.toString());

    if (options.sampleId) {
      formData.append('sample_id', options.sampleId);
    } else if (options.uploadedFile) {
      formData.append('file', options.uploadedFile);
    } else if (selectedFile) {
      formData.append('file', selectedFile);
    } else {
      setLoading(false);
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/detect`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setAnalysisResult(data);
      } else {
        const err = data.detail || "Analysis request failed from backend.";
        console.error("Backend error:", err);
        setErrorMessage(err);
      }
    } catch (err) {
      console.error("Detection connection error:", err);
      setErrorMessage("Could not connect to FastAPI server at http://127.0.0.1:8000. Ensure the Python backend is running.");
    } finally {
      setLoading(false);
    }
  };

  // Safe chart data extraction
  const latentChartData = (analysisResult && Array.isArray(analysisResult.latent_mu_preview))
    ? analysisResult.latent_mu_preview.map((val, idx) => ({
        dimension: `z[${idx}]`,
        activation: typeof val === 'number' ? Number(val.toFixed(3)) : 0,
      }))
    : [];

  const rocCurveData = (metrics && Array.isArray(metrics.fpr) && Array.isArray(metrics.tpr))
    ? metrics.fpr.map((fprVal, idx) => ({
        fpr: fprVal,
        tpr: metrics.tpr[idx] ?? 0,
      }))
    : [];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Navigation Bar */}
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur-md px-6 py-4 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-sky-500/20">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-sky-400 to-indigo-300 bg-clip-text text-transparent">
              NeuroScan-VAE
            </h1>
            <p className="text-xs text-slate-400">
              Unsupervised Medical Anomaly Detection System (VAE Engine)
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 bg-slate-800/80 px-3 py-1.5 rounded-full border border-slate-700/60 text-xs">
            <span className={`w-2 h-2 rounded-full ${systemHealth ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'}`} />
            <span className="text-slate-300">Backend:</span>
            <span className="font-medium text-slate-100 uppercase">{systemHealth ? (systemHealth.device || 'Online') : 'Offline (Run Server)'}</span>
          </div>

          <div className="flex bg-slate-800 rounded-lg p-0.5 border border-slate-700">
            <button
              onClick={() => setActiveTab('inspection')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                activeTab === 'inspection' ? 'bg-sky-600 text-white shadow-sm' : 'text-slate-400 hover:text-white'
              }`}
            >
              Diagnostic Inspection
            </button>
            <button
              onClick={() => setActiveTab('metrics')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                activeTab === 'metrics' ? 'bg-sky-600 text-white shadow-sm' : 'text-slate-400 hover:text-white'
              }`}
            >
              Model Benchmarks (ROC-AUC)
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Controls & Upload (4 Cols) */}
        <div className="lg:col-span-4 flex flex-col gap-6">
          
          {/* Card: Image Input & Upload */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                <Upload className="w-4 h-4 text-sky-400" /> Medical Scan Input
              </h2>
              <span className="text-xs text-slate-500">128x128 Axial MRI</span>
            </div>

            {/* Drag & Drop File Box */}
            <label className="border-2 border-dashed border-slate-700 hover:border-sky-500 rounded-xl p-5 flex flex-col items-center justify-center cursor-pointer transition-all bg-slate-950/40 hover:bg-slate-800/40 group">
              <input type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
              <Upload className="w-8 h-8 text-slate-500 group-hover:text-sky-400 mb-2 transition-colors" />
              <span className="text-xs font-medium text-slate-300">Click to upload Brain MRI Scan</span>
              <span className="text-[11px] text-slate-500 mt-1">PNG, JPG, or DICOM-derived slices</span>
            </label>

            {/* Pre-Loaded Benchmark Samples */}
            <div>
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2 block">
                Quick Benchmark Samples
              </span>
              <div className="grid grid-cols-2 gap-2">
                {demoSamples.length > 0 ? (
                  demoSamples.map((sample) => (
                    <button
                      key={sample.id}
                      onClick={() => handleDemoSelect(sample)}
                      disabled={loading}
                      className="flex flex-col items-center p-2 rounded-xl border border-slate-800 bg-slate-950/60 hover:border-sky-500/60 hover:bg-slate-800/50 transition-all text-left group disabled:opacity-50"
                    >
                      <img 
                        src={sample.preview} 
                        alt={sample.label} 
                        className="w-16 h-16 rounded-lg object-cover mb-2 border border-slate-800 group-hover:scale-105 transition-transform" 
                      />
                      <span className="text-[11px] font-medium text-slate-300 text-center line-clamp-2">
                        {sample.label}
                      </span>
                      <span className={`text-[9px] mt-1 px-1.5 py-0.5 rounded uppercase font-semibold ${
                        sample.type === 'healthy' ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' : 'bg-rose-950 text-rose-400 border border-rose-800'
                      }`}>
                        {sample.type}
                      </span>
                    </button>
                  ))
                ) : (
                  <div className="col-span-2 text-center p-4 text-xs text-slate-500 border border-dashed border-slate-800 rounded-xl">
                    Loading benchmark samples...
                  </div>
                )}
              </div>
            </div>

            {/* Run Button if user selected file manually */}
            {selectedFile && (
              <button
                onClick={() => runAnalysis()}
                disabled={loading}
                className="w-full py-2.5 bg-gradient-to-r from-sky-600 to-indigo-600 hover:from-sky-500 hover:to-indigo-500 text-white rounded-xl text-xs font-semibold flex items-center justify-center gap-2 shadow-lg shadow-sky-600/20 disabled:opacity-50 transition-all cursor-pointer"
              >
                {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
                Run Diagnostic VAE Analysis
              </button>
            )}
          </div>

          {/* Card: Hyperparameter & Threshold Controls */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col gap-4">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <Sliders className="w-4 h-4 text-sky-400" /> Decision Parameters
            </h2>

            {/* Anomaly Score Threshold Slider */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400 font-medium">Anomaly Threshold (τ)</span>
                <span className="font-mono text-sky-400 font-bold">{threshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.80"
                step="0.01"
                value={threshold}
                onChange={(e) => {
                  const val = parseFloat(e.target.value);
                  setThreshold(val);
                  if (analysisResult) {
                    setAnalysisResult({
                      ...analysisResult,
                      threshold: val,
                      is_anomalous: analysisResult.anomaly_score >= val,
                    });
                  }
                }}
                className="w-full accent-sky-500 bg-slate-800 h-1.5 rounded-lg appearance-none cursor-pointer"
              />
              <p className="text-[11px] text-slate-500 leading-tight">
                Pixels with normalized error above τ are flagged as pathological tissue deviations.
              </p>
            </div>

            {/* Gaussian Blur Kernel Slider */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400 font-medium">Spatial Smoothing Kernel</span>
                <span className="font-mono text-sky-400 font-bold">{blurKernel}x{blurKernel}</span>
              </div>
              <input
                type="range"
                min="1"
                max="11"
                step="2"
                value={blurKernel}
                onChange={(e) => setBlurKernel(parseInt(e.target.value))}
                className="w-full accent-sky-500 bg-slate-800 h-1.5 rounded-lg appearance-none cursor-pointer"
              />
              <p className="text-[11px] text-slate-500 leading-tight">
                Suppresses isolated high-frequency acquisition noise to cluster contiguous lesions.
              </p>
            </div>

            <div className="pt-2 border-t border-slate-800/80">
              <div className="flex items-start gap-2 text-[11px] text-slate-400 bg-slate-950/60 p-3 rounded-xl border border-slate-800">
                <Info className="w-4 h-4 text-sky-400 flex-shrink-0 mt-0.5" />
                <span>
                  The model was trained strictly on healthy brains. Lesions produce high reconstruction residuals because they lie outside the learned healthy manifold.
                </span>
              </div>
            </div>
          </div>

        </div>

        {/* Right Column: Diagnostic Visualizer & Results (8 Cols) */}
        <div className="lg:col-span-8 flex flex-col gap-6">

          {/* Error notification banner if any */}
          {errorMessage && (
            <div className="p-4 rounded-xl bg-rose-950/50 border border-rose-800 text-rose-300 text-xs flex items-center gap-3">
              <AlertCircle className="w-5 h-5 flex-shrink-0 text-rose-400" />
              <span>{errorMessage}</span>
            </div>
          )}

          {activeTab === 'inspection' ? (
            <>
              {/* Diagnostic Assessment Banner */}
              {loading ? (
                <div className="p-8 rounded-2xl border border-slate-800 bg-slate-900/50 flex flex-col items-center justify-center">
                  <RefreshCw className="w-8 h-8 text-sky-400 animate-spin mb-3" />
                  <span className="text-sm font-semibold text-slate-200">Reconstructing Image via Beta-VAE Engine...</span>
                  <span className="text-xs text-slate-500 mt-1">Passing through encoder bottleneck and residual calculation</span>
                </div>
              ) : analysisResult ? (
                <div className={`p-5 rounded-2xl border flex items-center justify-between transition-all ${
                  analysisResult.is_anomalous 
                    ? 'bg-rose-950/30 border-rose-800/80 text-rose-200' 
                    : 'bg-emerald-950/30 border-emerald-800/80 text-emerald-200'
                }`}>
                  <div className="flex items-center gap-4">
                    <div className={`p-3 rounded-xl ${
                      analysisResult.is_anomalous ? 'bg-rose-900/60 text-rose-400' : 'bg-emerald-900/60 text-emerald-400'
                    }`}>
                      {analysisResult.is_anomalous ? <AlertTriangle className="w-7 h-7" /> : <ShieldCheck className="w-7 h-7" />}
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs uppercase tracking-wider font-semibold opacity-75">Clinical Diagnostic Grade</span>
                        <span className={`text-[10px] px-2 py-0.5 rounded font-bold uppercase ${
                          analysisResult.is_anomalous ? 'bg-rose-900 text-rose-300' : 'bg-emerald-900 text-emerald-300'
                        }`}>
                          {analysisResult.is_anomalous ? 'Anomaly Detected' : 'Normal / Healthy'}
                        </span>
                      </div>
                      <h3 className="text-lg font-bold tracking-tight mt-0.5">{analysisResult.diagnostic_grade}</h3>
                    </div>
                  </div>

                  <div className="text-right">
                    <div className="text-xs opacity-75 uppercase tracking-wider font-medium">Anomaly Severity Score</div>
                    <div className="text-2xl font-black font-mono">
                      {typeof analysisResult.anomaly_score === 'number' 
                        ? (analysisResult.anomaly_score * 100).toFixed(1) 
                        : '0.0'}%
                    </div>
                  </div>
                </div>
              ) : (
                <div className="p-8 rounded-2xl border border-slate-800 bg-slate-900/40 text-center flex flex-col items-center justify-center">
                  <Activity className="w-10 h-10 text-slate-600 mb-2 animate-pulse" />
                  <h3 className="text-sm font-semibold text-slate-300">Awaiting Scan Input</h3>
                  <p className="text-xs text-slate-500 mt-1 max-w-sm">
                    Select a benchmark sample from the left or upload an axial brain MRI scan to view VAE reconstruction and thermal error heatmaps.
                  </p>
                </div>
              )}

              {/* 4-Panel Medical Image Visualizer */}
              {analysisResult && analysisResult.images && (
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                      <Layers className="w-4 h-4 text-sky-400" />Reconstruction & Heatmap Pipeline
                    </h2>
                    <span className="text-xs text-slate-500">Dual-Pass Analysis</span>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {/* Panel 1: Original */}
                    <div className="bg-slate-950 rounded-xl p-3 border border-slate-800 flex flex-col items-center">
                      <span className="text-[11px] font-semibold text-slate-400 mb-2 uppercase tracking-wide">
                        1. Input Scan (X)
                      </span>
                      {analysisResult.images.original ? (
                        <img 
                          src={analysisResult.images.original} 
                          alt="Input" 
                          className="w-full aspect-square rounded-lg object-contain border border-slate-800/80 bg-black"
                        />
                      ) : <div className="w-full aspect-square bg-slate-900 rounded-lg" />}
                      <span className="text-[10px] text-slate-500 mt-2">Axial Brain MRI</span>
                    </div>

                    {/* Panel 2: VAE Healthy Reconstruction */}
                    <div className="bg-slate-950 rounded-xl p-3 border border-slate-800 flex flex-col items-center">
                      <span className="text-[11px] font-semibold text-sky-400 mb-2 uppercase tracking-wide">
                        2. Recon (Healthy)
                      </span>
                      {analysisResult.images.reconstruction ? (
                        <img 
                          src={analysisResult.images.reconstruction} 
                          alt="Reconstructed" 
                          className="w-full aspect-square rounded-lg object-contain border border-slate-800/80 bg-black"
                        />
                      ) : <div className="w-full aspect-square bg-slate-900 rounded-lg" />}
                      <span className="text-[10px] text-slate-500 mt-2">Healthy Manifold Projection</span>
                    </div>

                    {/* Panel 3: Thermal Jet Overlay */}
                    <div className="bg-slate-950 rounded-xl p-3 border border-slate-800 flex flex-col items-center">
                      <span className="text-[11px] font-semibold text-amber-400 mb-2 uppercase tracking-wide">
                        3. Residual Heatmap
                      </span>
                      {analysisResult.images.heatmap_overlay ? (
                        <img 
                          src={analysisResult.images.heatmap_overlay} 
                          alt="Residual Overlay" 
                          className="w-full aspect-square rounded-lg object-contain border border-slate-800/80 bg-black"
                        />
                      ) : <div className="w-full aspect-square bg-slate-900 rounded-lg" />}
                      <span className="text-[10px] text-slate-500 mt-2">Error Colormap</span>
                    </div>

                    {/* Panel 4: Binary Anomaly Mask */}
                    <div className="bg-slate-950 rounded-xl p-3 border border-slate-800 flex flex-col items-center">
                      <span className="text-[11px] font-semibold text-rose-400 mb-2 uppercase tracking-wide">
                        4. Lesion Mask
                      </span>
                      {analysisResult.images.binary_mask ? (
                        <img 
                          src={analysisResult.images.binary_mask} 
                          alt="Binary Mask" 
                          className="w-full aspect-square rounded-lg object-contain border border-slate-800/80 bg-black"
                        />
                      ) : <div className="w-full aspect-square bg-slate-900 rounded-lg" />}
                      <span className="text-[10px] text-slate-500 mt-2">Threshold Cutoff (τ)</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Latent Space Bottleneck Activations */}
              {analysisResult && analysisResult.latent_mu_preview && latentChartData.length > 0 && (
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
                  <div className="flex items-center justify-between mb-2">
                    <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                      <Brain className="w-4 h-4 text-sky-400" /> Latent Space Coordinate Vector (μ in R^64)
                    </h2>
                    <span className="text-xs text-slate-500">First 16 Latent Dimensions</span>
                  </div>
                  <p className="text-xs text-slate-400 mb-4">
                    Distribution coordinates learned under the standard Gaussian prior p(z) ~ N(0, I).
                  </p>
                  
                  <div className="h-44 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={latentChartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="dimension" stroke="#64748b" tick={{ fontSize: 10 }} />
                        <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }}
                        />
                        <Bar dataKey="activation" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </>
          ) : (
            /* Tab 2: Model Performance & ROC-AUC Benchmarks */
            <div className="flex flex-col gap-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
                  <span className="text-xs text-slate-400 uppercase tracking-wider font-semibold">Validation ROC-AUC</span>
                  <div className="text-3xl font-extrabold text-sky-400 mt-1">
                    {metrics ? `${(metrics.roc_auc * 100).toFixed(1)}%` : 'Loading...'}
                  </div>
                  <span className="text-[11px] text-slate-500 mt-1 block">Separation of healthy vs tumor slices</span>
                </div>

                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
                  <span className="text-xs text-slate-400 uppercase tracking-wider font-semibold">Precision-Recall AUC</span>
                  <div className="text-3xl font-extrabold text-indigo-400 mt-1">
                    {metrics ? `${(metrics.pr_auc * 100).toFixed(1)}%` : 'Loading...'}
                  </div>
                  <span className="text-[11px] text-slate-500 mt-1 block">Rare disease positive detection power</span>
                </div>

                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
                  <span className="text-xs text-slate-400 uppercase tracking-wider font-semibold">Optimal Youden's J</span>
                  <div className="text-3xl font-extrabold text-emerald-400 mt-1">
                    {metrics ? metrics.optimal_threshold : '0.35'}
                  </div>
                  <span className="text-[11px] text-slate-500 mt-1 block">Calibrated anomaly decision threshold</span>
                </div>
              </div>

              {/* ROC Curve Chart */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-sky-400" /> Receiver Operating Characteristic (ROC) Curve
                  </h2>
                  <span className="text-xs text-slate-500">True Positive vs False Positive Rate</span>
                </div>

                <div className="h-64 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={rocCurveData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis 
                        dataKey="fpr" 
                        stroke="#64748b" 
                        label={{ value: 'False Positive Rate (FPR)', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 11 }} 
                      />
                      <YAxis 
                        dataKey="tpr" 
                        stroke="#64748b" 
                        label={{ value: 'True Positive Rate (TPR)', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }} 
                      />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }}
                      />
                      <Line type="monotone" dataKey="tpr" stroke="#0ea5e9" strokeWidth={2.5} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

        </div>

      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 px-6 py-4 text-center text-xs text-slate-500">
        Medical Anomaly Detection System &bull; PyTorch Beta-VAE Engine &bull; React & Tailwind Web Application
      </footer>
    </div>
  );
}
