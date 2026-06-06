import React, { useState } from 'react';
import { Play, Database, Terminal, AlertTriangle, CheckCircle, Sliders } from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

export default function RunAnalysisTab({ onSuccess }) {
  const [allTraces, setAllTraces] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [log, setLog] = useState('');
  const [status, setStatus] = useState(null); // 'success' | 'error' | null

  // Parameter states
  const [medianWindow, setMedianWindow] = useState('5');
  const [movingAverageWindow, setMovingAverageWindow] = useState('5');
  const [savgolAutoTune, setSavgolAutoTune] = useState(true);
  const [savgolWindow, setSavgolWindow] = useState('11');

  // Validation
  const getValidationErrors = () => {
    const errors = [];
    const medVal = parseInt(medianWindow, 10);
    if (isNaN(medVal) || medVal <= 0 || medVal % 2 === 0) {
      errors.push('Median window size must be a positive odd integer (e.g. 3, 5, 7).');
    }
    const movVal = parseInt(movingAverageWindow, 10);
    if (isNaN(movVal) || movVal <= 0) {
      errors.push('Moving average window size must be a positive integer.');
    }
    if (!savgolAutoTune) {
      const savVal = parseInt(savgolWindow, 10);
      if (isNaN(savVal) || savVal <= 0 || savVal % 2 === 0) {
        errors.push('Savitzky-Golay window size must be a positive odd integer (e.g. 5, 11, 15).');
      }
    }
    return errors;
  };

  const validationErrors = getValidationErrors();
  const isValid = validationErrors.length === 0;

  const handleRun = async () => {
    if (!isValid) return;
    setIsRunning(true);
    setLog('');
    setStatus(null);
    setLog(`Starting analysis${allTraces ? ' on ALL traces' : ' on latest data'}...\n`);

    try {
      const body = {
        all_traces: allTraces,
        median_window: parseInt(medianWindow, 10),
        moving_average_window: parseInt(movingAverageWindow, 10),
        savgol_window: savgolAutoTune ? null : parseInt(savgolWindow, 10),
      };

      const res = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();

      if (res.ok) {
        setLog(data.output || 'Analysis completed successfully.');
        setStatus('success');
        if (onSuccess) onSuccess();
      } else {
        setLog(data.detail || 'Analysis failed.');
        setStatus('error');
      }
    } catch (e) {
      setLog(`Error: ${e.message}`);
      setStatus('error');
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div style={{ maxWidth: 700, display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

      <div>
        <h2 style={{ fontSize: '1.1rem', fontWeight: 800, letterSpacing: '-0.02em', marginBottom: '0.4rem' }}>
          Run New Analysis
        </h2>
        <p className="text-sm text-secondary">
          Trigger your <code style={{ color: 'var(--accent-primary)', background: 'rgba(59,130,246,0.1)', padding: '1px 5px', borderRadius: 4, fontSize: '0.75rem' }}>analyze_traces.py</code> script to process traces from the <code style={{ color: 'var(--accent-cyan)', background: 'rgba(6,182,212,0.1)', padding: '1px 5px', borderRadius: 4, fontSize: '0.75rem' }}>data/</code> directory. Results will appear in the sidebar automatically.
        </p>
      </div>

      {/* Mode Selector */}
      <div className="glass-panel" style={{ padding: '1.5rem' }}>
        <div style={{ fontWeight: 700, marginBottom: '1rem' }}>Select Analysis Mode</div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <label style={{
            display: 'flex', alignItems: 'flex-start', gap: '0.875rem',
            padding: '1rem', borderRadius: 'var(--radius-lg)',
            border: `1px solid ${!allTraces ? 'rgba(59,130,246,0.4)' : 'var(--border-color)'}`,
            background: !allTraces ? 'rgba(59,130,246,0.07)' : 'transparent',
            cursor: 'pointer', transition: 'all 0.15s'
          }}>
            <input type="radio" name="mode" checked={!allTraces} onChange={() => setAllTraces(false)}
              style={{ marginTop: 2, accentColor: 'var(--accent-primary)' }} />
            <div>
              <div style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Play size={15} color="var(--accent-primary)" /> Latest Data Only
              </div>
              <div className="text-xs text-muted" style={{ marginTop: '0.25rem' }}>
                Analyzes only the most recent <code>fixed_*</code> and <code>random_*</code> folder pair in <code>data/</code>. Fast.
              </div>
            </div>
          </label>

          <label style={{
            display: 'flex', alignItems: 'flex-start', gap: '0.875rem',
            padding: '1rem', borderRadius: 'var(--radius-lg)',
            border: `1px solid ${allTraces ? 'rgba(168,85,247,0.4)' : 'var(--border-color)'}`,
            background: allTraces ? 'rgba(168,85,247,0.07)' : 'transparent',
            cursor: 'pointer', transition: 'all 0.15s'
          }}>
            <input type="radio" name="mode" checked={allTraces} onChange={() => setAllTraces(true)}
              style={{ marginTop: 2, accentColor: 'var(--accent-purple)' }} />
            <div>
              <div style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Database size={15} color="var(--accent-purple)" /> All Traces (Aggregated)
              </div>
              <div className="text-xs text-muted" style={{ marginTop: '0.25rem' }}>
                Merges <b>every</b> <code>fixed_*</code> and <code>random_*</code> folder in <code>data/</code> into one massive analysis. May take a few minutes.
              </div>
            </div>
          </label>
        </div>
      </div>

      {/* Filter Parameters */}
      <div className="glass-panel" style={{ padding: '1.5rem' }}>
        <div style={{ fontWeight: 700, marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Sliders size={16} color="var(--accent-cyan)" /> Customize Filter Parameters
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
          {/* Median Window */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
            <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              Median Window Size
            </label>
            <input
              type="number"
              value={medianWindow}
              onChange={(e) => setMedianWindow(e.target.value)}
              disabled={isRunning}
              style={{
                background: 'var(--bg-primary)',
                border: '1px solid var(--border-color)',
                borderRadius: 'var(--radius-sm)',
                padding: '0.5rem 0.75rem',
                color: 'var(--text-primary)',
                fontFamily: 'inherit',
                fontSize: '0.85rem',
                outline: 'none',
                transition: 'border-color 0.2s',
              }}
            />
            <span className="text-xs text-muted">Must be a positive odd integer (e.g. 3, 5, 7). Default: 5</span>
          </div>

          {/* Moving Average Window */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
            <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              Moving Average Window Size
            </label>
            <input
              type="number"
              value={movingAverageWindow}
              onChange={(e) => setMovingAverageWindow(e.target.value)}
              disabled={isRunning}
              style={{
                background: 'var(--bg-primary)',
                border: '1px solid var(--border-color)',
                borderRadius: 'var(--radius-sm)',
                padding: '0.5rem 0.75rem',
                color: 'var(--text-primary)',
                fontFamily: 'inherit',
                fontSize: '0.85rem',
                outline: 'none',
                transition: 'border-color 0.2s',
              }}
            />
            <span className="text-xs text-muted">Must be a positive integer. Default: 5</span>
          </div>
        </div>

        <div style={{ height: '1px', background: 'var(--border-color)', margin: '1.25rem 0' }} />

        {/* Savitzky-Golay */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', userSelect: 'none' }}>
            <input
              type="checkbox"
              checked={savgolAutoTune}
              onChange={(e) => setSavgolAutoTune(e.target.checked)}
              disabled={isRunning}
              style={{ accentColor: 'var(--accent-cyan)' }}
            />
            <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              Auto-Tune Savitzky-Golay Window Size
            </span>
          </label>
          <span className="text-xs text-muted" style={{ marginLeft: '1.5rem', marginTop: '-0.25rem' }}>
            When enabled, automatically estimates the optimal window length based on frequency response.
          </span>

          {!savgolAutoTune && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', marginLeft: '1.5rem', marginTop: '0.25rem', maxWidth: '300px' }}>
              <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                Savitzky-Golay Window Size
              </label>
              <input
                type="number"
                value={savgolWindow}
                onChange={(e) => setSavgolWindow(e.target.value)}
                disabled={isRunning}
                style={{
                  background: 'var(--bg-primary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '0.5rem 0.75rem',
                  color: 'var(--text-primary)',
                  fontFamily: 'inherit',
                  fontSize: '0.85rem',
                  outline: 'none',
                  transition: 'border-color 0.2s',
                }}
              />
              <span className="text-xs text-muted">Must be a positive odd integer (e.g. 5, 11, 15).</span>
            </div>
          )}
        </div>
      </div>

      {/* Validation Errors */}
      {!isValid && (
        <div className="glass-panel" style={{ padding: '1rem', borderColor: 'var(--accent-danger)', background: 'rgba(239, 68, 68, 0.05)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--accent-danger)', fontWeight: 600, marginBottom: '0.5rem' }}>
            <AlertTriangle size={16} /> Please resolve validation errors:
          </div>
          <ul style={{ paddingLeft: '1.25rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            {validationErrors.map((err, idx) => (
              <li key={idx} style={{ marginBottom: '0.25rem' }}>{err}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Run Button */}
      <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
        <button
          className={`btn ${allTraces ? 'btn-purple' : 'btn-primary'}`}
          onClick={handleRun}
          disabled={isRunning || !isValid}
          style={{ fontSize: '0.9rem', padding: '0.65rem 1.5rem' }}
        >
          {isRunning ? (
            <><div className="spinner" /> Running Analysis...</>
          ) : allTraces ? (
            <><Database size={16} /> Run All Traces</>
          ) : (
            <><Play size={16} /> Run Latest Data</>
          )}
        </button>
        {status === 'success' && (
          <span className="badge badge-success"><CheckCircle size={12} /> Success</span>
        )}
        {status === 'error' && (
          <span className="badge badge-danger"><AlertTriangle size={12} /> Failed</span>
        )}
      </div>

      {/* Log Output */}
      {log && (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', fontWeight: 600, fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            <Terminal size={14} /> Output
          </div>
          <div className="run-log">{log}</div>
        </div>
      )}

    </div>
  );
}
