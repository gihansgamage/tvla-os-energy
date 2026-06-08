import React, { useState, useRef, useEffect } from 'react';
import { TrendingUp, Zap, AlertTriangle, CheckCircle, Activity, BarChart2 } from 'lucide-react';

const FILTERS = ['raw', 'median', 'moving_avg', 'wavelet', 'regr'];
const FILTER_LABELS = {
  raw: 'Raw', median: 'Median', moving_avg: 'Moving Avg', wavelet: 'Wavelet', regr: 'Regression'
};

function formatDate(ts) {
  return new Date(ts * 1000).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: '2-digit' });
}

function formatDateFull(ts) {
  return new Date(ts * 1000).toLocaleString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' });
}

// -----------------------------------------------------------
// Mini SVG Trend Chart
// -----------------------------------------------------------
function TrendChart({ data, keyT, keyE, onHover, onClick }) {
  const svgRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);
  const W = 800, H = 200, PAD = { top: 16, right: 16, bottom: 30, left: 48 };
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;

  const validData = data.filter(d => d[keyT] != null);
  if (validData.length < 2) return <div className="empty-state" style={{ height: 200 }}>Not enough data for chart</div>;

  const tVals = validData.map(d => d[keyT]);
  const eVals = validData.map(d => d[keyE] ?? 0);
  const maxE = Math.max(...eVals);

  // Build a consistent domain that always includes the 4.5 threshold
  const THRESHOLD = 4.5;
  const dataMin = Math.min(...tVals);
  const dataMax = Math.max(...tVals);
  const pad = Math.max((dataMax - dataMin) * 0.12, 0.3);
  const domainMin = Math.min(dataMin, THRESHOLD) - pad;
  const domainMax = Math.max(dataMax, THRESHOLD) + pad;
  const domainRange = domainMax - domainMin;

  // Single source of truth for y mapping
  const xScale = (i) => PAD.left + (i / (validData.length - 1)) * innerW;
  const yScaleT = (v) => PAD.top + innerH - ((v - domainMin) / domainRange) * innerH;
  const yScaleE = (v) => PAD.top + innerH - (v / (maxE || 1)) * innerH;

  const tPath = validData.map((d, i) => `${i === 0 ? 'M' : 'L'}${xScale(i).toFixed(1)},${yScaleT(d[keyT]).toFixed(1)}`).join(' ');
  const ePath = validData.map((d, i) => `${i === 0 ? 'M' : 'L'}${xScale(i).toFixed(1)},${yScaleE(d[keyE] ?? 0).toFixed(1)}`).join(' ');

  // Grid lines: derived from the same yScaleT so labels match pixel positions exactly
  const gridLines = [0, 0.2, 0.4, 0.6, 0.8, 1].map(f => {
    const val = domainMin + f * domainRange;
    return { y: yScaleT(val), label: val.toFixed(1) };
  });

  const thresholdY = yScaleT(THRESHOLD);

  return (
    <div className="trend-chart-container" style={{ position: 'relative' }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }} ref={svgRef}>
        <defs>
          <linearGradient id="tGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.3" />
            <stop offset="100%" stopColor="#3b82f6" stopOpacity="0" />
          </linearGradient>
          <linearGradient id="eGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#ef4444" stopOpacity="0.2" />
            <stop offset="100%" stopColor="#ef4444" stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* Grid lines */}
        {gridLines.map((g, i) => (
          <g key={i}>
            <line x1={PAD.left} y1={g.y} x2={W - PAD.right} y2={g.y} stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
            <text x={PAD.left - 6} y={g.y + 4} textAnchor="end" fontSize="9" fill="rgba(255,255,255,0.25)">{g.label}</text>
          </g>
        ))}

        {/* Threshold line */}
        {thresholdY > PAD.top && thresholdY < PAD.top + innerH && (
          <g>
            <line x1={PAD.left} y1={thresholdY} x2={W - PAD.right} y2={thresholdY}
              stroke="#ef4444" strokeWidth="1" strokeDasharray="4 4" opacity="0.5" />
            <text x={W - PAD.right + 2} y={thresholdY + 4} fontSize="8" fill="#ef4444" opacity="0.7">|t|=4.5</text>
          </g>
        )}

        {/* Exceedance area */}
        <path
          d={`${ePath} L${xScale(validData.length - 1).toFixed(1)},${PAD.top + innerH} L${xScale(0).toFixed(1)},${PAD.top + innerH} Z`}
          fill="url(#eGrad)"
        />

        {/* T-stat area */}
        <path
          d={`${tPath} L${xScale(validData.length - 1).toFixed(1)},${PAD.top + innerH} L${xScale(0).toFixed(1)},${PAD.top + innerH} Z`}
          fill="url(#tGrad)"
        />

        {/* Lines */}
        <path d={ePath} stroke="#ef4444" strokeWidth="1.5" fill="none" opacity="0.7" />
        <path d={tPath} stroke="#3b82f6" strokeWidth="2" fill="none" />

        {/* Dots */}
        {validData.map((d, i) => (
          <circle
            key={i}
            cx={xScale(i)}
            cy={yScaleT(d[keyT])}
            r="4"
            fill={d[keyT] > 4.5 ? '#ef4444' : '#3b82f6'}
            stroke={d[keyT] > 4.5 ? '#ef4444' : '#3b82f6'}
            strokeWidth="2"
            fillOpacity="0.8"
            style={{ cursor: 'pointer' }}
            onClick={() => onClick && onClick(d)}
            onMouseEnter={(e) => {
              const rect = svgRef.current?.getBoundingClientRect();
              setTooltip({ x: e.clientX, y: e.clientY, d });
            }}
            onMouseLeave={() => setTooltip(null)}
          />
        ))}

        {/* X-axis dates (sample) */}
        {validData.filter((_, i) => i % Math.max(1, Math.floor(validData.length / 6)) === 0).map((d, i, arr) => {
          const origIdx = validData.indexOf(d);
          return (
            <text key={i} x={xScale(origIdx)} y={H - 4} textAnchor="middle" fontSize="8" fill="rgba(255,255,255,0.3)">
              {formatDate(d.created_at)}
            </text>
          );
        })}
      </svg>

      {tooltip && (
        <div className="chart-tooltip" style={{ left: tooltip.x + 12, top: tooltip.y - 40 }}>
          <div style={{ fontWeight: 600, marginBottom: 2 }}>{tooltip.d.id}</div>
          <div>Max |t|: <span style={{ color: '#3b82f6', fontWeight: 700 }}>{tooltip.d[keyT]?.toFixed(3)}</span></div>
          <div>Exceedance: <span style={{ color: '#ef4444', fontWeight: 700 }}>{tooltip.d[keyE]?.toFixed(1)}%</span></div>
          <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.65rem', marginTop: 2 }}>{formatDateFull(tooltip.d.created_at)}</div>
        </div>
      )}
    </div>
  );
}

// -----------------------------------------------------------
// Heatmap
// -----------------------------------------------------------
function Heatmap({ data, onSelect }) {
  const filterKeys = [
    { key: 'exceedance_pct_raw', label: 'Raw' },
    { key: 'exceedance_pct_median', label: 'Median' },
    { key: 'exceedance_pct_moving_avg', label: 'Mov.Avg' },
    { key: 'exceedance_pct_savgol', label: 'Sav.Gol' },
    { key: 'exceedance_pct_wavelet', label: 'Wavelet' },
    { key: 'exceedance_pct_regr', label: 'Regr.' },
  ];

  const validRows = data.filter(d => d.has_new_schema);
  const allVals = validRows.flatMap(d => filterKeys.map(f => d[f.key] ?? 0));
  const maxVal = Math.max(...allVals, 1);

  const getColor = (val) => {
    if (val == null) return 'rgba(255,255,255,0.04)';
    const intensity = val / maxVal;
    if (intensity < 0.001) return 'rgba(34, 197, 94, 0.15)';
    if (intensity < 0.3) return `rgba(245, 158, 11, ${0.2 + intensity * 0.6})`;
    return `rgba(239, 68, 68, ${0.3 + intensity * 0.7})`;
  };

  if (validRows.length === 0) {
    return <div className="empty-state" style={{ height: 100 }}>No data with filter metrics found.</div>;
  }

  return (
    <div style={{ overflowX: 'auto' }}>
      <div style={{ display: 'grid', gridTemplateColumns: `90px repeat(${validRows.length}, 28px)`, gap: '3px', alignItems: 'center', minWidth: 200 }}>
        {/* Header row */}
        <div />
        {validRows.map((d, i) => (
          <div key={i} style={{ fontSize: '0.55rem', color: 'var(--text-muted)', textAlign: 'center', transform: 'rotate(-45deg)', transformOrigin: 'center', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', width: 28 }}>
            {d.id.replace('analysis_', '').slice(0, 6)}
          </div>
        ))}

        {/* Filter rows */}
        {filterKeys.map(f => (
          <React.Fragment key={f.key}>
            <div className="heatmap-label">{f.label}</div>
            {validRows.map((d, i) => {
              const val = d[f.key];
              return (
                <div
                  key={i}
                  className="heatmap-cell"
                  style={{ background: getColor(val) }}
                  title={`${d.id}\n${f.label}: ${val?.toFixed(1) ?? 'N/A'}%`}
                  onClick={() => onSelect(d.id)}
                />
              );
            })}
          </React.Fragment>
        ))}
      </div>
      <div style={{ marginTop: '0.5rem', display: 'flex', gap: '1.5rem', fontSize: '0.68rem', color: 'var(--text-muted)' }}>
        <span>■ <span style={{ color: 'var(--accent-success)' }}>0%</span></span>
        <span>■ <span style={{ color: 'var(--accent-warning)' }}>Low</span></span>
        <span>■ <span style={{ color: 'var(--accent-danger)' }}>High</span></span>
      </div>
    </div>
  );
}

// -----------------------------------------------------------
// Sortable Table
// -----------------------------------------------------------
function RunsTable({ data, selectedId, onSelect }) {
  const [sortKey, setSortKey] = useState('created_at');
  const [sortDir, setSortDir] = useState('desc');

  const handleSort = (key) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('desc'); }
  };

  const sorted = [...data].sort((a, b) => {
    const av = a[sortKey] ?? -Infinity, bv = b[sortKey] ?? -Infinity;
    return sortDir === 'asc' ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1);
  });

  const Th = ({ k, label }) => (
    <th className={sortKey === k ? 'sorted' : ''} onClick={() => handleSort(k)}>
      {label} {sortKey === k ? (sortDir === 'asc' ? '↑' : '↓') : ''}
    </th>
  );

  const tClass = (val) => {
    if (val == null) return '';
    if (val > 4.5) return 'val-danger';
    if (val > 3.5) return 'val-warning';
    return 'val-success';
  };

  const eClass = (val) => {
    if (val == null) return '';
    if (val > 5) return 'val-danger';
    if (val > 1) return 'val-warning';
    return 'val-success';
  };

  return (
    <div style={{ overflowX: 'auto', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border-color)' }}>
      <table className="data-table">
        <thead>
          <tr>
            <Th k="created_at" label="Date" />
            <Th k="max_t_raw" label="Max |t| Raw" />
            <Th k="max_t_wavelet" label="Max |t| Wavelet" />
            <Th k="exceedance_pct_raw" label="Exceed. %" />
            <Th k="power_snr_raw" label="Power SNR" />
            <Th k="power_diff_mw_raw" label="Δ Power (mW)" />
            <Th k="fixed_traces" label="Fixed" />
            <Th k="random_traces" label="Random" />
            <Th k="fixed_input" label="Fixed Input" />
          </tr>
        </thead>
        <tbody>
          {sorted.map(d => (
            <tr key={d.id} onClick={() => onSelect(d.id)} className={selectedId === d.id ? 'row-active' : ''}>
              <td>
                <div className="mono" style={{ fontSize: '0.72rem', color: 'var(--text-primary)' }}>{d.id.replace('analysis_', '')}</div>
                <div className="text-xs text-muted">{formatDate(d.created_at)}</div>
              </td>
              <td className={`mono ${tClass(d.max_t_raw)}`}>{d.max_t_raw != null ? d.max_t_raw.toFixed(3) : '—'}</td>
              <td className={`mono ${tClass(d.max_t_wavelet)}`}>{d.max_t_wavelet != null ? d.max_t_wavelet.toFixed(3) : '—'}</td>
              <td className={`mono ${eClass(d.exceedance_pct_raw)}`}>{d.exceedance_pct_raw != null ? `${d.exceedance_pct_raw.toFixed(1)}%` : '—'}</td>
              <td className="mono">{d.power_snr_raw != null ? d.power_snr_raw.toFixed(4) : '—'}</td>
              <td className="mono">{d.power_diff_mw_raw != null ? `${d.power_diff_mw_raw.toFixed(0)} mW` : '—'}</td>
              <td className="mono">{d.fixed_traces ?? '—'}</td>
              <td className="mono">{d.random_traces ?? '—'}</td>
              <td className="mono" style={{ fontSize: '0.7rem' }}>{d.fixed_input ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// -----------------------------------------------------------
// Overview Tab
// -----------------------------------------------------------
export default function OverviewTab({ overviewData, selectedId, onSelectId }) {
  const validRuns = overviewData.filter(d => d.has_new_schema);
  const totalRuns = overviewData.length;
  const leakingRuns = validRuns.filter(d => (d.max_t_raw ?? 0) > 4.5).length;
  const avgMaxT = validRuns.length > 0
    ? (validRuns.reduce((s, d) => s + (d.max_t_raw ?? 0), 0) / validRuns.length).toFixed(3)
    : '—';
  const avgExceed = validRuns.length > 0
    ? (validRuns.reduce((s, d) => s + (d.exceedance_pct_raw ?? 0), 0) / validRuns.length).toFixed(1)
    : '—';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>

      {/* Stat Bar */}
      <div className="stat-bar">
        <div className="stat-bar-item">
          <div className="stat-val">{totalRuns}</div>
          <div className="stat-lbl">Total Runs</div>
        </div>
        <div className="stat-bar-item">
          <div className="stat-val" style={{ color: 'var(--accent-danger)' }}>{leakingRuns}</div>
          <div className="stat-lbl">Runs w/ Leakage (|t|&gt;4.5)</div>
        </div>
        <div className="stat-bar-item">
          <div className="stat-val">{totalRuns - leakingRuns}</div>
          <div className="stat-lbl">Clean Runs</div>
        </div>
        <div className="stat-bar-item">
          <div className="stat-val" style={{ color: 'var(--accent-primary)' }}>{avgMaxT}</div>
          <div className="stat-lbl">Avg Max |t|</div>
        </div>
        <div className="stat-bar-item">
          <div className="stat-val" style={{ color: 'var(--accent-warning)' }}>{avgExceed}%</div>
          <div className="stat-lbl">Avg Exceedance</div>
        </div>
        <div className="stat-bar-item">
          <div className="stat-val" style={{ color: 'var(--accent-success)' }}>{validRuns.length}</div>
          <div className="stat-lbl">Runs w/ Full Data</div>
        </div>
      </div>

      {/* Trend Chart */}
      <div className="glass-panel" style={{ padding: '1.5rem' }}>
        <div className="section-header">
          <TrendingUp size={18} color="var(--accent-primary)" />
          Max |t| Trend Over Time
        </div>
        <p className="section-desc">
          Blue line = Max |t| (raw). Red line = Exceedance %. Dashed red line = |t|=4.5 threshold.
          Click a dot to inspect that run.
        </p>
        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', marginBottom: '1rem' }}>
          <span className="badge badge-blue">— Max |t| (raw)</span>
          <span className="badge badge-danger">— Exceedance %</span>
          <span className="badge badge-neutral">⋯ Threshold (4.5)</span>
        </div>
        <TrendChart
          data={overviewData}
          keyT="max_t_raw"
          keyE="exceedance_pct_raw"
          onClick={(d) => onSelectId(d.id)}
        />
      </div>

      {/* Heatmap */}
      <div className="glass-panel" style={{ padding: '1.5rem' }}>
        <div className="section-header">
          <BarChart2 size={18} color="var(--accent-purple)" />
          Filter Exceedance Heatmap
        </div>
        <p className="section-desc">
          Each column is an analysis run. Each row is a filter type. Color intensity = exceedance rate.
          Click any cell to view that run.
        </p>
        <Heatmap data={overviewData} onSelect={onSelectId} />
      </div>

      {/* All Runs Table */}
      <div>
        <div className="section-header">
          <Activity size={18} color="var(--accent-cyan)" />
          All Analysis Runs
        </div>
        <p className="section-desc">
          Click any row to inspect in the Individual tab. Click column headers to sort.
        </p>
        <RunsTable data={overviewData} selectedId={selectedId} onSelect={onSelectId} />
      </div>

    </div>
  );
}
