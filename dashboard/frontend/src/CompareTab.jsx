import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  GitCompare, Download, RefreshCw, ChevronDown, ChevronUp,
  BarChart2, Activity, AlertTriangle, Table2, Layers, Zap, Info,
  Maximize2, X, CheckSquare, Square, Search, Eye, Filter, Cpu, TrendingUp, CheckCircle, AlertOctagon
} from 'lucide-react';
import { MultiLineChart } from './MultiLineChart';

const API_BASE = 'http://localhost:8000/api';

const PALETTE = [
  '#3b82f6', '#f59e0b', '#22c55e', '#ec4899',
  '#8b5cf6', '#06b6d4', '#eab308', '#ef4444',
  '#10b981', '#6366f1', '#f43f5e', '#14b8a6'
];

const FILTER_META = [
  { key: 'raw',            label: 'Raw TVLA',           color: '#3b82f6' },
  { key: 'median',         label: 'Median Filter',      color: '#8b5cf6' },
  { key: 'moving_average', label: 'Moving Average',     color: '#06b6d4' },
  { key: 'wavelet',        label: 'Wavelet Denoising',  color: '#f59e0b' },
  { key: 'savitzky_golay', label: 'Savitzky–Golay',     color: '#ec4899' },
  { key: 'lowpass',        label: 'Lowpass Butterworth',color: '#eab308' },
];

function fmt(v, decimals = 3) {
  if (v == null || v === '') return '—';
  const n = parseFloat(v);
  return isNaN(n) ? String(v) : n.toFixed(decimals);
}

function leakColor(maxT) {
  if (maxT == null) return 'var(--text-muted)';
  if (maxT > 4.5) return 'var(--accent-danger, #ef4444)';
  return 'var(--accent-success, #22c55e)';
}

// ─── Collapsible Section ────────────────────────────────────────────────
function Section({ icon, title, badge, children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="page-section" style={{
      background: 'var(--bg-secondary)',
      border: '1px solid var(--border-color)',
      borderRadius: 'var(--radius-lg)',
      padding: '1.25rem',
      marginBottom: '1.25rem'
    }}>
      <div
        style={{
          cursor: 'pointer', userSelect: 'none', display: 'flex',
          justifyContent: 'space-between', alignItems: 'center'
        }}
        onClick={() => setOpen(o => !o)}
      >
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', fontWeight: 700, fontSize: '0.92rem' }}>
          {icon} {title}
          {badge && (
            <span style={{
              fontSize: '0.65rem', padding: '0.15rem 0.5rem', borderRadius: 'var(--radius-sm)',
              background: 'rgba(59,130,246,0.12)', color: 'var(--accent-primary)',
              border: '1px solid rgba(59,130,246,0.25)', fontWeight: 600
            }}>
              {badge}
            </span>
          )}
        </span>
        <button className="btn btn-ghost" style={{ padding: '0.2rem 0.4rem', pointerEvents: 'none' }}>
          {open ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
        </button>
      </div>
      {open && <div style={{ marginTop: '1rem' }}>{children}</div>}
    </div>
  );
}

// ─── Lightbox Modal ─────────────────────────────────────────────────────
function ImageModal({ src, title, onClose }) {
  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 1000,
        background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(8px)',
        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
        padding: '1.5rem'
      }}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{
          background: 'var(--bg-secondary)', border: '1px solid var(--border-color)',
          borderRadius: 'var(--radius-lg)', overflow: 'hidden', maxWidth: '94vw', maxHeight: '92vh',
          display: 'flex', flexDirection: 'column', boxShadow: '0 10px 40px rgba(0,0,0,0.6)'
        }}
      >
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '0.75rem 1.25rem', borderBottom: '1px solid var(--border-color)',
          background: 'var(--bg-tertiary)'
        }}>
          <span style={{ fontWeight: 600, fontSize: '0.85rem', color: 'var(--text-primary)' }}>{title}</span>
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            <a href={src} download={title || 'plot.png'}>
              <button className="btn btn-ghost" style={{ fontSize: '0.72rem', padding: '0.3rem 0.6rem' }}>
                <Download size={13} /> Download
              </button>
            </a>
            <button className="btn btn-ghost" onClick={onClose} style={{ padding: '0.3rem 0.5rem' }}>
              <X size={15} />
            </button>
          </div>
        </div>
        <div style={{ padding: '1rem', overflow: 'auto', textAlign: 'center' }}>
          <img src={src} alt={title} style={{ maxWidth: '100%', maxHeight: '78vh', borderRadius: 'var(--radius-md)' }} />
        </div>
      </div>
    </div>
  );
}

// ─── Generic CSV Table ──────────────────────────────────────────────────
function CsvTable({ rows, downloadUrl, downloadName, maxHeight = 340 }) {
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(0);
  const pageSize = 50;

  if (!rows || rows.length === 0)
    return <div className="text-xs text-muted" style={{ padding: '0.75rem' }}>No data available.</div>;

  const cols = Object.keys(rows[0]);
  const filtered = useMemo(() => {
    if (!search.trim()) return rows;
    const q = search.toLowerCase();
    return rows.filter(r => Object.values(r).some(v => String(v).toLowerCase().includes(q)));
  }, [rows, search]);

  const paged = filtered.slice(page * pageSize, (page + 1) * pageSize);
  const totalPages = Math.ceil(filtered.length / pageSize);

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <input
            type="text"
            placeholder="Search rows…"
            value={search}
            onChange={e => { setSearch(e.target.value); setPage(0); }}
            style={{
              background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
              borderRadius: 'var(--radius-sm)', padding: '0.35rem 0.65rem',
              color: 'var(--text-primary)', fontSize: '0.75rem', outline: 'none'
            }}
          />
          <span className="text-xs text-muted">
            {filtered.length} {filtered.length === 1 ? 'row' : 'rows'}
          </span>
        </div>
        {downloadUrl && (
          <a href={downloadUrl} download={downloadName || 'data.csv'}>
            <button className="btn btn-primary" style={{ fontSize: '0.72rem', padding: '0.35rem 0.75rem' }}>
              <Download size={12} /> Download CSV
            </button>
          </a>
        )}
      </div>

      <div style={{
        overflowX: 'auto', maxHeight, overflowY: 'auto',
        borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)'
      }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.73rem' }}>
          <thead style={{ position: 'sticky', top: 0, zIndex: 1 }}>
            <tr style={{ background: 'var(--bg-tertiary)' }}>
              {cols.map(c => (
                <th key={c} style={{
                  padding: '0.5rem 0.75rem', textAlign: 'left',
                  color: 'var(--text-muted)', fontWeight: 600,
                  borderBottom: '1px solid var(--border-color)', whiteSpace: 'nowrap'
                }}>
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paged.map((row, i) => (
              <tr key={i} style={{ background: i % 2 === 0 ? 'var(--bg-primary)' : 'var(--bg-secondary)' }}>
                {cols.map(c => {
                  const v = row[c];
                  const isT = c.includes('t_') || c === 'max_abs_t_statistic';
                  const isVerdict = c === 'verdict';
                  const isExceed = c.includes('exceed');
                  let color = undefined;
                  if (isT && typeof v === 'number') color = leakColor(Math.abs(v));
                  if (isVerdict) color = v === 'CONTROL_PASSED' ? 'var(--accent-success)' : 'var(--accent-danger)';
                  return (
                    <td key={c} style={{
                      padding: '0.4rem 0.75rem', borderBottom: '1px solid var(--border-color)',
                      color: color || 'var(--text-secondary)',
                      fontFamily: typeof v === 'number' ? 'JetBrains Mono, monospace' : 'inherit',
                      whiteSpace: 'nowrap',
                      fontWeight: (isT || isVerdict) ? 600 : 'normal'
                    }}>
                      {typeof v === 'number' ? fmt(v, isExceed ? 1 : 3) : String(v ?? '—')}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.5rem', fontSize: '0.72rem' }}>
          <span className="text-muted">Page {page + 1} of {totalPages}</span>
          <div style={{ display: 'flex', gap: '0.3rem' }}>
            <button className="btn btn-ghost" disabled={page === 0} onClick={() => setPage(p => p - 1)} style={{ padding: '0.2rem 0.5rem' }}>Prev</button>
            <button className="btn btn-ghost" disabled={page >= totalPages - 1} onClick={() => setPage(p => p + 1)} style={{ padding: '0.2rem 0.5rem' }}>Next</button>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Metric Card (Same as IndividualTab) ────────────────────────────────
function MetricCard({ label, icon, iconColor, value, sub, highlight }) {
  return (
    <div className="metric-card" style={highlight ? { borderColor: `${iconColor}50` } : {}}>
      <div className="metric-label" style={{ color: iconColor ? `${iconColor}99` : undefined }}>
        {icon && React.cloneElement(icon, { size: 13, color: iconColor })}
        {label}
      </div>
      <div className="metric-value" style={highlight ? { color: iconColor } : {}}>
        {value}
      </div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  );
}

// ─── Main CompareTab Component ──────────────────────────────────────────
export default function CompareTab() {
  const [allRuns, setAllRuns] = useState([]);
  const [overviewList, setOverviewList] = useState([]);
  const [controlRuns, setControlRuns] = useState([]);
  const [activeRunId, setActiveRunId] = useState(null);
  const [currentSummary, setCurrentSummary] = useState(null);

  // Setup inputs
  const [mode, setMode] = useState('fixed'); // 'fixed' | 'random'
  const [dataset1, setDataset1] = useState('');
  const [dataset2, setDataset2] = useState('');

  // Tuning parameters
  const [medianWindow, setMedianWindow] = useState(5);
  const [movingAvgWindow, setMovingAvgWindow] = useState(5);
  const [savgolWindow, setSavgolWindow] = useState(15);

  // Execution state
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState(null);

  // Active chart tab: 'tstat' | 'power'
  const [chartTab, setChartTab] = useState('tstat');

  // Multi-line chart series
  const [tstatSeries, setTstatSeries] = useState([]);
  const [powerSeries, setPowerSeries] = useState([]);

  // CSV viewer tab
  const [activeCsv, setActiveCsv] = useState('quantitative_metrics.csv');
  const [csvRows, setCsvRows] = useState(null);

  // Lightbox Modal
  const [modalImage, setModalImage] = useState(null);

  // Load all analyses & existing control runs
  const fetchInitialData = useCallback(async () => {
    try {
      const [runsRes, ovRes, ctrlRes] = await Promise.all([
        fetch(`${API_BASE}/analyses`),
        fetch(`${API_BASE}/analyses/overview`),
        fetch(`${API_BASE}/control-tvla/runs`)
      ]);
      const runs = await runsRes.json();
      const ov = await ovRes.json();
      const ctrls = await ctrlRes.json();

      setAllRuns(runs || []);
      setOverviewList(ov || []);
      setControlRuns(ctrls || []);

      if (ctrls && ctrls.length > 0) {
        setActiveRunId(ctrls[0].id);
      }
    } catch (e) {
      console.error(e);
    }
  }, []);

  useEffect(() => { fetchInitialData(); }, [fetchInitialData]);

  // Overview lookup map for trace counts
  const overviewMap = useMemo(() => {
    const map = {};
    overviewList.forEach(r => { map[r.id] = r; });
    return map;
  }, [overviewList]);

  // Get trace volume for a dataset
  const getVolume = (id) => {
    const ov = overviewMap[id];
    if (!ov) return null;
    return mode === 'fixed' ? ov.fixed_traces : ov.random_traces;
  };

  const vol1 = dataset1 ? getVolume(dataset1) : null;
  const vol2 = dataset2 ? getVolume(dataset2) : null;
  const volumeMatches = vol1 != null && vol2 != null && vol1 === vol2;
  const volumeMismatch = vol1 != null && vol2 != null && vol1 !== vol2;

  // Auto-select initial pair if available with matching trace count
  useEffect(() => {
    if (allRuns.length >= 2 && !dataset1 && !dataset2) {
      // Find two runs that share identical trace count
      const groups = {};
      allRuns.forEach(r => {
        const v = getVolume(r.id);
        if (v != null) {
          groups[v] = groups[v] || [];
          groups[v].push(r.id);
        }
      });
      for (const v in groups) {
        if (groups[v].length >= 2) {
          setDataset1(groups[v][0]);
          setDataset2(groups[v][1]);
          break;
        }
      }
    }
  }, [allRuns, overviewMap, mode]);

  // Load active control run data (summary, tstat series, power series, CSVs)
  const loadActiveRun = useCallback(async (runId) => {
    if (!runId) return;
    try {
      const sumRes = await fetch(`${API_BASE}/control-tvla/${runId}/summary`);
      const sumData = await sumRes.json();
      setCurrentSummary(sumData);

      // Load master t-stat CSV for interactive multi-line chart
      const tRes = await fetch(`${API_BASE}/control-tvla/${runId}/csv/all_t_stat_filters.csv`);
      if (tRes.ok) {
        const tRows = await tRes.json();
        const cols = Object.keys(tRows[0] || {}).filter(k => k !== 'index');
        const series = cols.map((col, idx) => {
          const meta = FILTER_META.find(f => col.includes(f.key));
          return {
            key: col,
            label: meta ? meta.label : col.replace('t_', '').replace(/_/g, ' '),
            data: tRows.map(r => r[col] ?? 0),
            color: meta ? meta.color : PALETTE[idx % PALETTE.length],
            yAxis: 'left'
          };
        });
        setTstatSeries(series);
      }

      // Load power traces CSV for interactive chart
      const pRes = await fetch(`${API_BASE}/control-tvla/${runId}/csv/power_traces_${sumData.mode?.includes('random') ? 'random' : 'fixed'}.csv`);
      if (pRes.ok) {
        const pRows = await pRes.json();
        const pCols = Object.keys(pRows[0] || {}).filter(k => k !== 'index');
        const series = pCols.map((col, idx) => ({
          key: col,
          label: col.replace(/_/g, ' '),
          data: pRows.map(r => r[col] ?? 0),
          color: idx === 0 ? '#3b82f6' : idx === 1 ? '#f59e0b' : '#22c55e',
          yAxis: 'left'
        }));
        setPowerSeries(series);
      }

      // Load active CSV table
      loadActiveCsvTable(runId, 'quantitative_metrics.csv');
    } catch (e) {
      console.error(e);
    }
  }, []);

  const loadActiveCsvTable = async (runId, csvName) => {
    setActiveCsv(csvName);
    try {
      const res = await fetch(`${API_BASE}/control-tvla/${runId}/csv/${csvName}`);
      if (res.ok) {
        setCsvRows(await res.json());
      } else {
        setCsvRows(null);
      }
    } catch (e) {
      setCsvRows(null);
    }
  };

  useEffect(() => {
    if (activeRunId) {
      loadActiveRun(activeRunId);
    }
  }, [activeRunId, loadActiveRun]);

  // Execute TVLA calculation via backend API
  const handleCalculateTvla = async () => {
    if (!volumeMatches) {
      setRunError(`Cannot calculate: Trace volume mismatch (${vol1} vs ${vol2}). Volumes must be equal!`);
      return;
    }
    setRunning(true);
    setRunError(null);
    try {
      const res = await fetch(`${API_BASE}/control-tvla/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode,
          dataset1,
          dataset2,
          median_window: medianWindow,
          moving_average_window: movingAvgWindow,
          savgol_window: savgolWindow
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'TVLA calculation failed');
      }
      await fetchInitialData();
      setActiveRunId(data.run_id);
    } catch (e) {
      setRunError(e.message);
    } finally {
      setRunning(false);
    }
  };

  // Find active control run object
  const activeRunObj = controlRuns.find(r => r.id === activeRunId);
  const qm = currentSummary?.quantitative_metrics;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

      {/* Header bar */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        flexWrap: 'wrap', gap: '0.75rem', paddingBottom: '0.5rem',
        borderBottom: '1px solid var(--border-color)'
      }}>
        <div>
          <h1 style={{
            fontSize: '1.25rem', fontWeight: 800, letterSpacing: '-0.02em',
            display: 'flex', alignItems: 'center', gap: '0.6rem'
          }}>
            <GitCompare size={22} color="var(--accent-primary)" />
            TVLA Null-Hypothesis Studio (Fixed vs Fixed / Random vs Random)
          </h1>
          <p className="text-xs text-muted" style={{ marginTop: '0.2rem' }}>
            Calculate new TVLA Welch t-values between 2 Fixed value sets (or 2 Random value sets) with matching trace volumes, apply filters, and evaluate environment baseline noise.
          </p>
        </div>
        <button className="btn btn-ghost" onClick={fetchInitialData} disabled={running}>
          <RefreshCw size={14} className={running ? 'spin' : ''} /> Refresh
        </button>
      </div>

      {/* ── SECTION 1: TVLA CALCULATION & TRACE VOLUME VALIDATOR ── */}
      <Section
        icon={<Zap size={17} color="var(--accent-warning)" />}
        title="Control TVLA Setup: Select 2 Datasets"
        badge={volumeMatches ? `Valid: ${vol1} traces each` : volumeMismatch ? 'Volume Mismatch' : 'Select Datasets'}
        defaultOpen={true}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>

          {/* Mode Switcher */}
          <div>
            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
              1. CONTROL MODE
            </div>
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              <button
                className={`btn ${mode === 'fixed' ? 'btn-primary' : 'btn-ghost'}`}
                style={{ padding: '0.5rem 1rem', fontSize: '0.78rem' }}
                onClick={() => setMode('fixed')}
              >
                Fixed vs Fixed (Null Hypothesis Test)
              </button>
              <button
                className={`btn ${mode === 'random' ? 'btn-primary' : 'btn-ghost'}`}
                style={{ padding: '0.5rem 1rem', fontSize: '0.78rem' }}
                onClick={() => setMode('random')}
              >
                Random vs Random (Null Hypothesis Test)
              </button>
            </div>
          </div>

          {/* Dataset 1 & Dataset 2 Pickers */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
            {/* Dataset 1 */}
            <div style={{
              background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
              borderRadius: 'var(--radius-md)', padding: '0.85rem'
            }}>
              <label style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--accent-primary)', display: 'block', marginBottom: '0.4rem' }}>
                Dataset 1 ({mode.toUpperCase()} Set A)
              </label>
              <select
                value={dataset1}
                onChange={e => setDataset1(e.target.value)}
                style={{
                  width: '100%', background: 'var(--bg-secondary)', border: '1px solid var(--border-color)',
                  color: 'var(--text-primary)', padding: '0.45rem 0.65rem', borderRadius: 'var(--radius-sm)',
                  fontSize: '0.78rem', outline: 'none', fontFamily: 'JetBrains Mono, monospace'
                }}
              >
                <option value="">-- Choose Dataset 1 --</option>
                {allRuns.map(r => {
                  const v = getVolume(r.id);
                  return (
                    <option key={r.id} value={r.id}>
                      {r.id.replace('analysis_', '')} — ({v != null ? `${v.toLocaleString()} traces` : 'N/A'})
                    </option>
                  );
                })}
              </select>
              {vol1 != null && (
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>
                  Trace Volume: <b style={{ color: 'var(--text-secondary)' }}>{vol1.toLocaleString()} traces</b>
                </div>
              )}
            </div>

            {/* Dataset 2 */}
            <div style={{
              background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
              borderRadius: 'var(--radius-md)', padding: '0.85rem'
            }}>
              <label style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--accent-warning)', display: 'block', marginBottom: '0.4rem' }}>
                Dataset 2 ({mode.toUpperCase()} Set B)
              </label>
              <select
                value={dataset2}
                onChange={e => setDataset2(e.target.value)}
                style={{
                  width: '100%', background: 'var(--bg-secondary)', border: '1px solid var(--border-color)',
                  color: 'var(--text-primary)', padding: '0.45rem 0.65rem', borderRadius: 'var(--radius-sm)',
                  fontSize: '0.78rem', outline: 'none', fontFamily: 'JetBrains Mono, monospace'
                }}
              >
                <option value="">-- Choose Dataset 2 --</option>
                {allRuns.map(r => {
                  const v = getVolume(r.id);
                  return (
                    <option key={r.id} value={r.id}>
                      {r.id.replace('analysis_', '')} — ({v != null ? `${v.toLocaleString()} traces` : 'N/A'})
                    </option>
                  );
                })}
              </select>
              {vol2 != null && (
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>
                  Trace Volume: <b style={{ color: 'var(--text-secondary)' }}>{vol2.toLocaleString()} traces</b>
                </div>
              )}
            </div>
          </div>

          {/* Trace Volume Live Validation Banner */}
          {dataset1 && dataset2 && (
            <div>
              {volumeMatches ? (
                <div style={{
                  background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.3)',
                  borderRadius: 'var(--radius-md)', padding: '0.6rem 0.9rem', display: 'flex',
                  alignItems: 'center', gap: '0.5rem', color: 'var(--accent-success)', fontSize: '0.78rem'
                }}>
                  <CheckCircle size={15} />
                  <span>
                    <b>Trace Volumes Match:</b> Both datasets have exactly <b>{vol1.toLocaleString()} traces</b>. Ready to calculate TVLA t-values!
                  </span>
                </div>
              ) : (
                <div style={{
                  background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.3)',
                  borderRadius: 'var(--radius-md)', padding: '0.6rem 0.9rem', display: 'flex',
                  alignItems: 'center', gap: '0.5rem', color: 'var(--accent-danger)', fontSize: '0.78rem'
                }}>
                  <AlertOctagon size={16} />
                  <span>
                    <b>Trace Volume Mismatch Error:</b> Dataset 1 has <b>{vol1?.toLocaleString() ?? '?'} traces</b>, but Dataset 2 has <b>{vol2?.toLocaleString() ?? '?'} traces</b>.
                    For TVLA {mode}-vs-{mode} calculation, trace volumes MUST be identical! Please select datasets with matching trace counts.
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Filter Parameters Tuning */}
          <div style={{
            display: 'flex', gap: '1rem', flexWrap: 'wrap',
            paddingTop: '0.5rem', borderTop: '1px solid var(--border-color)', alignItems: 'center'
          }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>Filter Windows:</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.72rem' }}>
              <span>Median:</span>
              <input
                type="number" min="3" step="2" value={medianWindow}
                onChange={e => setMedianWindow(parseInt(e.target.value) || 5)}
                style={{ width: 48, background: 'var(--bg-primary)', border: '1px solid var(--border-color)', color: 'white', padding: '0.2rem 0.4rem', borderRadius: 4 }}
              />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.72rem' }}>
              <span>Moving Avg:</span>
              <input
                type="number" min="2" value={movingAvgWindow}
                onChange={e => setMovingAvgWindow(parseInt(e.target.value) || 5)}
                style={{ width: 48, background: 'var(--bg-primary)', border: '1px solid var(--border-color)', color: 'white', padding: '0.2rem 0.4rem', borderRadius: 4 }}
              />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.72rem' }}>
              <span>Savitzky-Golay:</span>
              <input
                type="number" min="5" step="2" value={savgolWindow}
                onChange={e => setSavgolWindow(parseInt(e.target.value) || 15)}
                style={{ width: 48, background: 'var(--bg-primary)', border: '1px solid var(--border-color)', color: 'white', padding: '0.2rem 0.4rem', borderRadius: 4 }}
              />
            </div>

            <button
              className="btn btn-primary"
              disabled={running || !volumeMatches || !dataset1 || !dataset2 || dataset1 === dataset2}
              onClick={handleCalculateTvla}
              style={{ marginLeft: 'auto', padding: '0.55rem 1.25rem', fontSize: '0.82rem', fontWeight: 700 }}
            >
              {running ? (
                <>
                  <div className="spinner" style={{ width: 14, height: 14 }} />
                  Calculating TVLA &amp; Filters…
                </>
              ) : (
                <>
                  <Zap size={14} /> Calculate New TVLA &amp; Apply Filters
                </>
              )}
            </button>
          </div>

          {runError && (
            <div style={{
              background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
              borderRadius: 'var(--radius-md)', padding: '0.75rem', color: 'var(--accent-danger)', fontSize: '0.75rem'
            }}>
              {runError}
            </div>
          )}

        </div>
      </Section>

      {/* ── SECTION 2: RESULTS VIEWPORT ── */}
      {currentSummary && (
        <>
          {/* Verdict Banner */}
          <div style={{
            background: currentSummary.control_test_passed ? 'rgba(34,197,94,0.1)' : 'rgba(239,68,68,0.1)',
            border: `1px solid ${currentSummary.control_test_passed ? 'rgba(34,197,94,0.3)' : 'rgba(239,68,68,0.3)'}`,
            borderRadius: 'var(--radius-lg)', padding: '1rem 1.25rem', display: 'flex',
            alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem'
          }}>
            <div>
              <div style={{
                fontSize: '0.95rem', fontWeight: 800,
                color: currentSummary.control_test_passed ? 'var(--accent-success)' : 'var(--accent-danger)',
                display: 'flex', alignItems: 'center', gap: '0.5rem'
              }}>
                {currentSummary.control_test_passed ? <CheckCircle size={18} /> : <AlertTriangle size={18} />}
                {currentSummary.control_test_passed
                  ? '✓ CONTROL TEST PASSED — Environment Valid (No Baseline Noise Detected)'
                  : '⚠ CONTROL TEST FAILED — Baseline Noise / Jitter Detected between Identical Inputs'}
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.3rem' }}>
                {currentSummary.verdict}
              </div>
            </div>

            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: '0.62rem', color: 'var(--text-muted)' }}>Raw Max |t|</div>
                <div style={{
                  fontSize: '1.15rem', fontWeight: 800, fontFamily: 'JetBrains Mono, monospace',
                  color: leakColor(currentSummary.raw_max_abs_t_statistic)
                }}>
                  {currentSummary.raw_max_abs_t_statistic?.toFixed(3)}
                </div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: '0.62rem', color: 'var(--text-muted)' }}>Exceedances</div>
                <div style={{ fontSize: '1.15rem', fontWeight: 800 }}>
                  {currentSummary.raw_samples_exceeding_threshold} <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>samples</span>
                </div>
              </div>
            </div>
          </div>

          {/* Filter Metric Cards (Matching IndividualTab) */}
          {qm && (
            <Section
              icon={<Cpu size={17} color="var(--accent-primary)" />}
              title="Filter Metrics (All 6 Types on New TVLA Calculation)"
              defaultOpen={true}
            >
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '0.85rem' }}>
                {FILTER_META.map(f => {
                  const m = qm[f.key];
                  if (!m) return null;
                  const maxT = m.max_abs_t_statistic ?? 0;
                  const passed = maxT <= 4.5;
                  return (
                    <div key={f.key} style={{
                      background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                      borderLeft: `4px solid ${f.color}`, borderRadius: 'var(--radius-md)', padding: '1rem'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.6rem' }}>
                        <span style={{ fontWeight: 700, fontSize: '0.82rem', color: f.color }}>
                          {f.label}
                        </span>
                        <span className={`badge ${passed ? 'badge-success' : 'badge-danger'}`} style={{ fontSize: '0.62rem' }}>
                          {passed ? '✓ Clean (≤ 4.5)' : '⚠ Noise (> 4.5)'}
                        </span>
                      </div>

                      <div className="metrics-grid" style={{ gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                        <div>
                          <div style={{ fontSize: '0.62rem', color: 'var(--text-muted)' }}>Max |t|</div>
                          <div style={{ fontWeight: 800, fontSize: '1rem', color: leakColor(maxT), fontFamily: 'JetBrains Mono, monospace' }}>
                            {maxT.toFixed(3)}
                          </div>
                        </div>
                        <div>
                          <div style={{ fontSize: '0.62rem', color: 'var(--text-muted)' }}>Exceed %</div>
                          <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>
                            {(m.exceedance_percent ?? 0).toFixed(1)}%
                          </div>
                        </div>
                        <div>
                          <div style={{ fontSize: '0.62rem', color: 'var(--text-muted)' }}>Power SNR</div>
                          <div style={{ fontWeight: 600, color: '#22c55e', fontSize: '0.8rem' }}>
                            {fmt(m.power_snr, 4)}
                          </div>
                        </div>
                        <div>
                          <div style={{ fontSize: '0.62rem', color: 'var(--text-muted)' }}>Δ Power</div>
                          <div style={{ fontWeight: 600, color: '#f59e0b', fontSize: '0.8rem' }}>
                            {fmt(m.mean_power_difference_mw, 1)} mW
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </Section>
          )}

          {/* Interactive MultiLineChart (Matching Dashboard Results) */}
          {(tstatSeries.length > 0 || powerSeries.length > 0) && (
            <Section
              icon={<TrendingUp size={17} color="var(--accent-cyan)" />}
              title="Interactive Per-Sample Charts"
              defaultOpen={true}
            >
              <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
                <button
                  className={`btn ${chartTab === 'tstat' ? 'btn-primary' : 'btn-ghost'}`}
                  style={{ fontSize: '0.75rem', padding: '0.35rem 0.8rem' }}
                  onClick={() => setChartTab('tstat')}
                >
                  <BarChart2 size={13} /> Newly Calculated TVLA t-Statistic Overlay (±4.5 Threshold)
                </button>
                <button
                  className={`btn ${chartTab === 'power' ? 'btn-primary' : 'btn-ghost'}`}
                  style={{ fontSize: '0.75rem', padding: '0.35rem 0.8rem' }}
                  onClick={() => setChartTab('power')}
                >
                  <Activity size={13} /> Compared Power Signals Overlay
                </button>
              </div>

              <div style={{
                background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                borderRadius: 'var(--radius-md)', padding: '1rem'
              }}>
                {chartTab === 'tstat' && tstatSeries.length > 0 && (
                  <div>
                    <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                      TVLA t-Statistics for each filter technique (Red dashed lines at ±4.5 threshold)
                    </div>
                    <MultiLineChart
                      series={tstatSeries}
                      leftTitle="t-statistic"
                      thresholds={[
                        { value: 4.5, color: '#ef4444', dasharray: '4 4', label: '+4.5 threshold', yAxis: 'left' },
                        { value: -4.5, color: '#ef4444', dasharray: '4 4', label: '-4.5 threshold', yAxis: 'left' },
                        { value: 0, color: '#555555', dasharray: '2 2', label: '0.0', yAxis: 'left' }
                      ]}
                      height={260}
                      downloadTitle={`Control TVLA ${currentSummary.mode}`}
                      downloadSubtitle={`${currentSummary.dataset_1} vs ${currentSummary.dataset_2}`}
                    />
                  </div>
                )}

                {chartTab === 'power' && powerSeries.length > 0 && (
                  <div>
                    <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                      Average Power Signals for Dataset 1 vs Dataset 2
                    </div>
                    <MultiLineChart
                      series={powerSeries}
                      leftTitle="Power (mW)"
                      height={260}
                      downloadTitle={`Power Signals - ${currentSummary.dataset_1} vs ${currentSummary.dataset_2}`}
                    />
                  </div>
                )}
              </div>
            </Section>
          )}

          {/* Generated Plots Gallery with Lightbox */}
          {activeRunObj?.plots && activeRunObj.plots.length > 0 && (
            <Section
              icon={<Eye size={17} color="var(--accent-primary)" />}
              title="Generated Report Figures"
              badge={`${activeRunObj.plots.length} figures`}
              defaultOpen={true}
            >
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1rem' }}>
                {activeRunObj.plots.map(p => {
                  const url = `${API_BASE}/control-tvla/${activeRunId}/plot/${p}`;
                  const title = p.replace('.png', '').replace(/_/g, ' ');
                  return (
                    <div key={p} style={{
                      background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                      borderRadius: 'var(--radius-md)', padding: '0.75rem', display: 'flex', flexDirection: 'column'
                    }}>
                      <div style={{
                        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                        marginBottom: '0.5rem', fontSize: '0.78rem'
                      }}>
                        <span style={{ fontWeight: 700, textTransform: 'capitalize' }}>{title}</span>
                        <div style={{ display: 'flex', gap: '0.35rem' }}>
                          <button
                            className="btn btn-ghost"
                            onClick={() => setModalImage({ src: url, title })}
                            style={{ padding: '0.25rem 0.5rem', fontSize: '0.68rem' }}
                          >
                            <Maximize2 size={12} /> Fullscreen
                          </button>
                          <a href={url} download={p}>
                            <button className="btn btn-ghost" style={{ padding: '0.25rem 0.5rem', fontSize: '0.68rem' }}>
                              <Download size={12} />
                            </button>
                          </a>
                        </div>
                      </div>
                      <div
                        onClick={() => setModalImage({ src: url, title })}
                        style={{ cursor: 'pointer', textAlign: 'center', overflow: 'hidden', borderRadius: 'var(--radius-sm)' }}
                      >
                        <img
                          src={url}
                          alt={title}
                          style={{ width: '100%', height: '220px', objectFit: 'contain', background: '#0F1117' }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </Section>
          )}

          {/* CSV Tables & Downloads */}
          {activeRunObj?.csvs && activeRunObj.csvs.length > 0 && (
            <Section
              icon={<Table2 size={17} color="var(--accent-purple)" />}
              title="Calculated TVLA CSV Data Tables &amp; Downloads"
              badge={`${activeRunObj.csvs.length} CSV files`}
              defaultOpen={true}
            >
              <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap', marginBottom: '1rem' }}>
                {activeRunObj.csvs.map(csvName => {
                  const active = activeCsv === csvName;
                  return (
                    <button
                      key={csvName}
                      className={`btn ${active ? 'btn-primary' : 'btn-ghost'}`}
                      style={{ fontSize: '0.72rem', padding: '0.3rem 0.65rem' }}
                      onClick={() => loadActiveCsvTable(activeRunId, csvName)}
                    >
                      <Table2 size={12} /> {csvName.replace('.csv', '').replace(/_/g, ' ')}
                    </button>
                  );
                })}
              </div>

              {csvRows && (
                <CsvTable
                  rows={csvRows}
                  downloadUrl={`${API_BASE}/control-tvla/${activeRunId}/download/${activeCsv}`}
                  downloadName={activeCsv}
                  maxHeight={320}
                />
              )}
            </Section>
          )}
        </>
      )}

      {/* History of Previous Control Runs */}
      {controlRuns.length > 1 && (
        <Section
          icon={<Layers size={17} color="var(--accent-muted)" />}
          title={`Previous Control TVLA Runs (${controlRuns.length})`}
          defaultOpen={false}
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
            {controlRuns.map(r => (
              <div
                key={r.id}
                onClick={() => setActiveRunId(r.id)}
                style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  padding: '0.55rem 0.85rem', borderRadius: 'var(--radius-sm)',
                  background: activeRunId === r.id ? 'rgba(59,130,246,0.12)' : 'var(--bg-primary)',
                  border: activeRunId === r.id ? '1px solid var(--accent-primary)' : '1px solid var(--border-color)',
                  cursor: 'pointer'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                  <span style={{ fontWeight: 600, fontSize: '0.78rem', fontFamily: 'JetBrains Mono, monospace' }}>
                    {r.id.replace('control_', '')}
                  </span>
                  <span className="badge badge-neutral" style={{ fontSize: '0.62rem' }}>
                    {r.summary?.trace_volume ? `${r.summary.trace_volume} traces` : ''}
                  </span>
                </div>
                {r.summary?.raw_max_abs_t_statistic != null && (
                  <span style={{
                    fontSize: '0.68rem', fontWeight: 700,
                    color: leakColor(r.summary.raw_max_abs_t_statistic)
                  }}>
                    max |t| = {r.summary.raw_max_abs_t_statistic.toFixed(3)}
                  </span>
                )}
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Lightbox Modal */}
      {modalImage && (
        <ImageModal
          src={modalImage.src}
          title={modalImage.title}
          onClose={() => setModalImage(null)}
        />
      )}

    </div>
  );
}
