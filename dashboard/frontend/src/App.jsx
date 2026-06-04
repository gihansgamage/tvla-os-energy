import React, { useState, useEffect } from 'react';
import { Activity, Play, Image as ImageIcon, BarChart2, AlertTriangle, CheckCircle, Database } from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

function App() {
  const [analyses, setAnalyses] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [summary, setSummary] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const fetchAnalyses = async () => {
    try {
      const res = await fetch(`${API_BASE}/analyses`);
      const data = await res.json();
      setAnalyses(data);
      if (data.length > 0 && !selectedId) {
        setSelectedId(data[0].id);
      }
    } catch (e) {
      console.error("Failed to fetch analyses", e);
    }
  };

  useEffect(() => {
    fetchAnalyses();
  }, []);

  useEffect(() => {
    if (selectedId) {
      fetch(`${API_BASE}/analyses/${selectedId}/summary`)
        .then(res => res.json())
        .then(data => setSummary(data))
        .catch(e => console.error("Failed to fetch summary", e));
    }
  }, [selectedId]);

  const handleRunAnalysis = async () => {
    setIsAnalyzing(true);
    try {
      const res = await fetch(`${API_BASE}/analyze`, { method: 'POST' });
      if (res.ok) {
        await fetchAnalyses();
      } else {
        alert("Analysis failed. Check server logs.");
      }
    } catch (e) {
      console.error(e);
      alert("Failed to run analysis.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getMetricValue = (key) => {
    if (!summary || !summary.quantitative_metrics || !summary.quantitative_metrics.raw) return "N/A";
    return summary.quantitative_metrics.raw[key];
  };

  const formatDate = (timestamp) => {
    return new Date(timestamp * 1000).toLocaleString();
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar">
        <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--border-color)' }}>
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1.25rem' }}>
            <Activity color="var(--accent-primary)" />
            TVLA Runs
          </h2>
        </div>
        <div style={{ flex: 1, overflowY: 'auto', padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {analyses.map(a => (
            <div 
              key={a.id} 
              onClick={() => setSelectedId(a.id)}
              style={{
                padding: '1rem',
                borderRadius: '8px',
                cursor: 'pointer',
                border: `1px solid ${selectedId === a.id ? 'var(--accent-primary)' : 'transparent'}`,
                backgroundColor: selectedId === a.id ? 'rgba(59, 130, 246, 0.1)' : 'transparent',
                transition: 'all 0.2s'
              }}
              onMouseEnter={(e) => {
                if (selectedId !== a.id) e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)';
              }}
              onMouseLeave={(e) => {
                if (selectedId !== a.id) e.currentTarget.style.backgroundColor = 'transparent';
              }}
            >
              <div style={{ fontWeight: 500, marginBottom: '0.25rem' }}>{a.id}</div>
              <div className="text-xs text-muted" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Database size={12} />
                {formatDate(a.created_at)}
              </div>
            </div>
          ))}
          {analyses.length === 0 && <div className="text-muted text-sm">No analysis runs found.</div>}
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
          <div>
            <h1>Dashboard</h1>
            <p className="text-muted">Interpret latest TVLA statistical results.</p>
          </div>
          <button 
            className="btn btn-primary glass-panel" 
            onClick={handleRunAnalysis}
            disabled={isAnalyzing}
            style={{ opacity: isAnalyzing ? 0.7 : 1 }}
          >
            {isAnalyzing ? (
              <><div className="loading-indicator" /> Running...</>
            ) : (
              <><Play size={18} /> Run Analysis on Latest Data</>
            )}
          </button>
        </div>

        {selectedId && summary ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            
            {/* Metric Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
              
              <div className="glass-panel" style={{ padding: '1.5rem' }}>
                <div className="text-sm text-muted" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                  <AlertTriangle size={16} color="var(--accent-danger)" />
                  Exceedance Rate (Raw)
                </div>
                <div style={{ fontSize: '2rem', fontWeight: 700 }}>
                  {(getMetricValue('exceedance_percent') || 0).toFixed(2)}%
                </div>
              </div>

              <div className="glass-panel" style={{ padding: '1.5rem' }}>
                <div className="text-sm text-muted" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                  <BarChart2 size={16} color="var(--accent-primary)" />
                  Max T-Statistic (Raw)
                </div>
                <div style={{ fontSize: '2rem', fontWeight: 700 }}>
                  {(getMetricValue('max_abs_t_statistic') || 0).toFixed(2)}
                </div>
              </div>

              <div className="glass-panel" style={{ padding: '1.5rem' }}>
                <div className="text-sm text-muted" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                  <Activity size={16} color="var(--accent-success)" />
                  Power SNR
                </div>
                <div style={{ fontSize: '2rem', fontWeight: 700 }}>
                  {(getMetricValue('power_snr') || 0).toFixed(4)}
                </div>
              </div>

              <div className="glass-panel" style={{ padding: '1.5rem' }}>
                <div className="text-sm text-muted" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                  <CheckCircle size={16} color="#a855f7" />
                  Total Samples
                </div>
                <div style={{ fontSize: '2rem', fontWeight: 700 }}>
                  {getMetricValue('samples')}
                </div>
              </div>

            </div>

            {/* Verdict Box */}
            {summary.decision_matrix && summary.decision_matrix.decision_1_environment_control && (
               <div className="glass-panel" style={{ padding: '1.5rem', borderLeft: '4px solid var(--accent-primary)' }}>
                 <h3 style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>Environment Control Verdict</h3>
                 <p style={{ color: 'var(--text-secondary)' }}>
                   {summary.decision_matrix.decision_1_environment_control.verdict}
                 </p>
               </div>
            )}

            {/* Plots Gallery */}
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
                <ImageIcon size={20} color="var(--accent-primary)" />
                Analysis Plots
              </h2>
              
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '2rem' }}>
                
                {[
                  { file: 'tvla.png', title: 'TVLA (Raw)' },
                  { file: 'tvla_regression_residual.png', title: 'TVLA (Regression Residual)' },
                  { file: 'tvla_moving_average.png', title: 'TVLA (Moving Average)' },
                  { file: 'tvla_wavelet.png', title: 'TVLA (Wavelet)' },
                  { file: 'tvla_migration_overlay.png', title: 'Migration Overlay' },
                  { file: 'migration_effect.png', title: 'Migration Effect' },
                ].map(plot => (
                  <div key={plot.file} style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    <div style={{ fontWeight: 500 }}>{plot.title}</div>
                    <img 
                      src={`${API_BASE}/analyses/${selectedId}/plot/${plot.file}`} 
                      alt={plot.title}
                      style={{ 
                        width: '100%', 
                        borderRadius: '8px', 
                        border: '1px solid var(--border-color)',
                        backgroundColor: '#fff' // Matplotlib plots are often transparent or white
                      }} 
                      onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'block'; }}
                    />
                    <div style={{ display: 'none', padding: '2rem', textAlign: 'center', backgroundColor: 'rgba(0,0,0,0.2)', borderRadius: '8px', color: 'var(--text-secondary)' }}>
                      Plot not generated
                    </div>
                  </div>
                ))}

              </div>
            </div>

          </div>
        ) : (
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px', color: 'var(--text-secondary)' }}>
            {selectedId ? "Loading analysis data..." : "Select an analysis from the sidebar"}
          </div>
        )}

      </div>
    </div>
  );
}

export default App;
