import React, { useState, useEffect, useRef } from 'react';
import { Download } from 'lucide-react';
import { MultiLineChart } from './MultiLineChart';
import { downloadSvgAsPng } from './downloadSvg';

const API_BASE = 'http://localhost:8000/api';

// Column name mapping for each TVLA filter
const FILTER_CSV_MAP = {
  raw: {
    t_stat: { file: 'tvla_t_stat.csv', col: 't_stat' },
    p_value: { file: 'tvla_p_value.csv', col: 'p_value' },
    label: 'Raw',
    color: '#3b82f6',
  },
  median: {
    t_stat: { file: 'tvla_t_stat_median.csv', col: 't_stat_median' },
    p_value: { file: 'tvla_p_value_median.csv', col: 'p_value_median' },
    label: 'Median',
    color: '#8b5cf6',
  },
  moving_average: {
    t_stat: { file: 'tvla_t_stat_moving_average.csv', col: 't_stat_moving_average' },
    p_value: { file: 'tvla_p_value_moving_average.csv', col: 'p_value_moving_average' },
    label: 'Moving Average',
    color: '#06b6d4',
  },
  wavelet: {
    t_stat: { file: 'tvla_t_stat_wavelet.csv', col: 't_stat_wavelet' },
    p_value: { file: 'tvla_p_value_wavelet.csv', col: 'p_value_wavelet' },
    label: 'Wavelet',
    color: '#f59e0b',
  },
  savitzky_golay: {
    t_stat: { file: 'tvla_t_stat_savitzky_golay.csv', col: 't_stat_savitzky_golay' },
    p_value: { file: 'tvla_p_value_savitzky_golay.csv', col: 'p_value_savitzky_golay' },
    label: 'Savitzky–Golay',
    color: '#ec4899',
  },
  regression_residual: {
    t_stat: { file: 'tvla_t_stat_regression_residual.csv', col: 't_stat_regression_residual' },
    p_value: { file: 'tvla_p_value_regression_residual.csv', col: 'p_value_regression_residual' },
    label: 'Regression Residual',
    color: '#22c55e',
  },
};

// Colors for all 8 filters (used in MultiLineChart)
const ALL_FILTERS_CONFIG = [
  { key: 'raw',                  label: 'Raw',                  color: '#3b82f6' },
  { key: 'median',               label: 'Median',               color: '#8b5cf6' },
  { key: 'moving_average',       label: 'Moving Average',       color: '#06b6d4' },
  { key: 'lowpass',              label: 'Lowpass',              color: '#eab308' },
  { key: 'savitzky_golay',       label: 'Savitzky–Golay',       color: '#ec4899' },
  { key: 'wavelet',              label: 'Wavelet',              color: '#f97316' },
  { key: 'regression_predicted', label: 'Regression Predicted', color: '#a855f7' },
  { key: 'regression_residual',  label: 'Regression Residual',  color: '#22c55e' },
];

// -----------------------------------------------------------
// Inline SVG bar chart for t-stat values per sample
// -----------------------------------------------------------
function TStatChart({ data, color, threshold = 4.5, title = '', subtitle = '' }) {
  const svgRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);
  const W = 800, H = 140;
  const PAD = { top: 14, right: 20, bottom: 20, left: 55 };
  const iW = W - PAD.left - PAD.right;
  const iH = H - PAD.top - PAD.bottom;

  if (!data || data.length === 0) return null;

  const vals = data.map(d => Math.abs(d));
  const maxVal = Math.max(...vals, threshold + 0.5);
  const n = vals.length;
  const barW = Math.max(1.5, iW / n - 1);

  const xPos = (i) => PAD.left + (i / n) * iW;
  const yPos = (v) => PAD.top + iH - (v / maxVal) * iH;
  const threshY = yPos(threshold);

  const gridVals = [0, threshold, maxVal * 0.5, maxVal].filter((v, i, arr) =>
    arr.indexOf(v) === i && v >= 0
  ).sort((a, b) => a - b);

  return (
    <div style={{ position: 'relative' }}>
      <button
        onClick={() => downloadSvgAsPng(svgRef.current, 't-statistic-chart.png', title, subtitle)}
        className="btn btn-ghost"
        style={{ position: 'absolute', top: -5, right: 0, padding: '4px', zIndex: 10 }}
        title="Download Graph"
      >
        <Download size={14} />
      </button>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }} ref={svgRef}>
        {gridVals.map((v, i) => {
          const y = yPos(v);
          if (y < PAD.top || y > PAD.top + iH) return null;
          return (
            <g key={i}>
              <line x1={PAD.left} y1={y} x2={W - PAD.right} y2={y}
                stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
              <text x={PAD.left - 6} y={y + 3.5} textAnchor="end" fontSize="8" fill="var(--text-muted)" fontFamily="monospace">
                {v.toFixed(1)}
              </text>
            </g>
          );
        })}

        <line x1={PAD.left} y1={threshY} x2={W - PAD.right} y2={threshY}
          stroke="#ef4444" strokeWidth="1.2" strokeDasharray="4 3" opacity="0.7" />
        <text x={W - PAD.right + 4} y={threshY + 3.5} fontSize="7.5" fill="#ef4444" opacity="0.8">
          {threshold}
        </text>

        {vals.map((v, i) => {
          const x = xPos(i);
          const y = yPos(v);
          const h = Math.max(1, iH - (y - PAD.top));
          const exceeds = v > threshold;
          return (
            <rect
              key={i}
              x={x}
              y={y}
              width={barW}
              height={h}
              fill={exceeds ? '#ef4444' : color}
              opacity={exceeds ? 0.9 : 0.65}
              rx="1"
              style={{ cursor: 'pointer' }}
              onMouseEnter={(e) => setTooltip({ x: e.clientX, y: e.clientY, idx: i, val: data[i] })}
              onMouseLeave={() => setTooltip(null)}
            />
          );
        })}

        {[0, Math.floor(n / 4), Math.floor(n / 2), Math.floor(3 * n / 4), n - 1].map(i => (
          <text key={i} x={xPos(i) + barW / 2} y={H - 4} textAnchor="middle"
            fontSize="8" fill="var(--text-muted)" fontFamily="monospace">
            {i}
          </text>
        ))}
      </svg>

      {tooltip && (
        <div style={{
          position: 'fixed', left: tooltip.x + 12, top: tooltip.y - 36,
          background: 'var(--bg-active)', border: '1px solid var(--border-hover)',
          borderRadius: 6, padding: '4px 8px', fontSize: '0.73rem',
          pointerEvents: 'none', zIndex: 9999, whiteSpace: 'nowrap',
          boxShadow: '0 4px 12px rgba(0,0,0,0.4)', color: 'var(--text-primary)'
        }}>
          Sample <b>{tooltip.idx}</b>: |t| = <span style={{ color, fontWeight: 700 }}>
            {Math.abs(tooltip.val).toFixed(4)}
          </span>
          {Math.abs(tooltip.val) > threshold && (
            <span style={{ color: '#ef4444', marginLeft: 6 }}>⚠ exceeds {threshold}</span>
          )}
        </div>
      )}
    </div>
  );
}

// -----------------------------------------------------------
// Inline SVG scatter for p-value (log scale)
// -----------------------------------------------------------
function PValueChart({ data, color, title = '', subtitle = '' }) {
  const svgRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);
  const W = 800, H = 130;
  const PAD = { top: 14, right: 20, bottom: 20, left: 55 };
  const iW = W - PAD.left - PAD.right;
  const iH = H - PAD.top - PAD.bottom;

  if (!data || data.length === 0) return null;

  const SIG_LINE = 0.05;
  const n = data.length;
  const dotR = Math.max(2, Math.min(3.5, iW / n / 2));

  const LOG_MIN = -10, LOG_MAX = 0;
  const logRange = LOG_MAX - LOG_MIN;
  const safeLog = (p) => Math.max(LOG_MIN, Math.log10(Math.max(1e-15, p)));
  const xPos = (i) => PAD.left + ((i + 0.5) / n) * iW;
  const yPos = (p) => PAD.top + iH - ((safeLog(p) - LOG_MIN) / logRange) * iH;
  const sigY = yPos(SIG_LINE);

  const gridPs = [1, 0.05, 0.001, 1e-6, 1e-8, 1e-10];

  return (
    <div style={{ position: 'relative' }}>
      <button
        onClick={() => downloadSvgAsPng(svgRef.current, 'p-value-chart.png', title, subtitle)}
        className="btn btn-ghost"
        style={{ position: 'absolute', top: -5, right: 0, padding: '4px', zIndex: 10 }}
        title="Download Graph"
      >
        <Download size={14} />
      </button>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }} ref={svgRef}>
        {gridPs.map((p, i) => {
          const y = yPos(p);
          if (y < PAD.top - 2 || y > PAD.top + iH + 2) return null;
          return (
            <g key={i}>
              <line x1={PAD.left} y1={y} x2={W - PAD.right} y2={y}
                stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
              <text x={PAD.left - 6} y={y + 3.5} textAnchor="end" fontSize="8" fill="var(--text-muted)" fontFamily="monospace">
                {p < 0.001 ? p.toExponential(0) : p}
              </text>
            </g>
          );
        })}

        <line x1={PAD.left} y1={sigY} x2={W - PAD.right} y2={sigY}
          stroke="#f59e0b" strokeWidth="1.2" strokeDasharray="4 3" opacity="0.8" />
        <text x={W - PAD.right + 4} y={sigY + 3.5} fontSize="7.5" fill="#f59e0b" opacity="0.9">
          α=0.05
        </text>

        {data.map((p, i) => {
          const sig = p < SIG_LINE;
          return (
            <circle
              key={i}
              cx={xPos(i)}
              cy={yPos(p)}
              r={dotR}
              fill={sig ? '#ef4444' : color}
              opacity={sig ? 0.9 : 0.55}
              style={{ cursor: 'pointer' }}
              onMouseEnter={(e) => setTooltip({ x: e.clientX, y: e.clientY, idx: i, val: p })}
              onMouseLeave={() => setTooltip(null)}
            />
          );
        })}

        {[0, Math.floor(n / 4), Math.floor(n / 2), Math.floor(3 * n / 4), n - 1].map(i => (
          <text key={i} x={xPos(i)} y={H - 4} textAnchor="middle"
            fontSize="8" fill="var(--text-muted)" fontFamily="monospace">
            {i}
          </text>
        ))}
      </svg>

      {tooltip && (
        <div style={{
          position: 'fixed', left: tooltip.x + 12, top: tooltip.y - 36,
          background: 'var(--bg-active)', border: '1px solid var(--border-hover)',
          borderRadius: 6, padding: '4px 8px', fontSize: '0.73rem',
          pointerEvents: 'none', zIndex: 9999, whiteSpace: 'nowrap',
          boxShadow: '0 4px 12px rgba(0,0,0,0.4)', color: 'var(--text-primary)'
        }}>
          Sample <b>{tooltip.idx}</b>: p = <span style={{ color, fontWeight: 700 }}>
            {tooltip.val < 0.001 ? tooltip.val.toExponential(3) : tooltip.val.toFixed(4)}
          </span>
          {tooltip.val < 0.05 && (
            <span style={{ color: '#ef4444', marginLeft: 6 }}>significant</span>
          )}
        </div>
      )}
    </div>
  );
}

// -----------------------------------------------------------
// Significance Summary Row
// -----------------------------------------------------------
function SigSummary({ tData, pData, color, threshold = 4.5 }) {
  if (!tData || !pData) return null;
  const tExceed = tData.filter(v => Math.abs(v) > threshold).length;
  const pSig = pData.filter(v => v < 0.05).length;
  const n = tData.length;

  return (
    <div style={{
      display: 'flex', gap: '1rem', flexWrap: 'wrap',
      background: 'var(--bg-primary)', borderRadius: 8,
      padding: '0.6rem 0.875rem', marginBottom: '0.75rem', fontSize: '0.78rem'
    }}>
      <span>
        Samples: <b style={{ color: 'var(--text-primary)' }}>{n}</b>
      </span>
      <span>
        |t| &gt; {threshold}:{' '}
        <b style={{ color: tExceed > 0 ? '#ef4444' : '#22c55e' }}>
          {tExceed} ({((tExceed / n) * 100).toFixed(1)}%)
        </b>
      </span>
      <span>
        p &lt; 0.05:{' '}
        <b style={{ color: pSig > 0 ? '#f59e0b' : '#22c55e' }}>
          {pSig} ({((pSig / n) * 100).toFixed(1)}%)
        </b>
      </span>
      <span>
        Max |t|:{' '}
        <b style={{ color }}>
          {Math.max(...tData.map(Math.abs)).toFixed(4)}
        </b>
      </span>
      <span>
        Min p:{' '}
        <b style={{ color }}>
          {Math.min(...pData).toExponential(3)}
        </b>
      </span>
    </div>
  );
}

// -----------------------------------------------------------
// TVLA Filter panel (collapsible)
// -----------------------------------------------------------
function FilterPanel({ analysisId, filterKey, meta, summary }) {
  const [tData, setTData] = useState(null);
  const [pData, setPData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(false);

  const filterParams = summary?.filter_parameters || {};
  const medWindow = filterParams.median_window ?? 5;
  const movAvgWindow = filterParams.moving_average_window ?? 5;
  const savgolWindow = filterParams.savitzky_golay_window ?? summary?.auto_tuned_parameters?.savitzky_golay_window ?? 11;
  const savgolAutoTuned = filterParams.savitzky_golay_auto_tuned ?? (summary?.auto_tuned_parameters ? true : false);

  const getFilterParamsText = () => {
    switch (filterKey) {
      case 'raw': return 'no filter';
      case 'median': return `window = ${medWindow}`;
      case 'moving_average': return `window = ${movAvgWindow}`;
      case 'savitzky_golay': return savgolAutoTuned ? `auto-tuned window = ${savgolWindow}` : `window = ${savgolWindow}`;
      case 'wavelet': return 'wavelet = sym4, level = 1';
      case 'regression_residual': return 'regression filter';
      default: return '';
    }
  };

  const paramText = getFilterParamsText();

  const load = async () => {
    if (tData !== null) return;
    setLoading(true);
    setError(null);
    try {
      const [tRes, pRes] = await Promise.all([
        fetch(`${API_BASE}/analyses/${analysisId}/csv-data/${meta.t_stat.file}`),
        fetch(`${API_BASE}/analyses/${analysisId}/csv-data/${meta.p_value.file}`),
      ]);
      
      if (!tRes.ok || !pRes.ok) throw new Error('TVLA CSV files not found');
      
      const tRows = await tRes.json();
      const pRows = await pRes.json();
      
      setTData(tRows.map(r => r[meta.t_stat.col] ?? 0));
      setPData(pRows.map(r => r[meta.p_value.col] ?? 1));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const toggle = () => {
    if (!expanded) load();
    setExpanded(e => !e);
  };

  return (
    <div style={{
      border: `1px solid ${expanded ? `${meta.color}40` : 'var(--border-color)'}`,
      borderLeft: `3px solid ${meta.color}`,
      borderRadius: 'var(--radius-lg)',
      background: 'var(--bg-tertiary)',
      overflow: 'hidden',
      transition: 'border-color 0.2s',
    }}>
      <button
        onClick={toggle}
        style={{
          width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '0.875rem 1.1rem', background: 'transparent', border: 'none',
          cursor: 'pointer', color: 'var(--text-primary)', fontFamily: 'inherit',
          textAlign: 'left',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
          <span style={{ width: 10, height: 10, borderRadius: '50%', background: meta.color, display: 'inline-block', flexShrink: 0 }} />
          <span style={{ fontWeight: 700, fontSize: '0.88rem' }}>{meta.label}</span>
          {paramText && (
            <span style={{ fontSize: '0.65rem', padding: '2px 6px', background: 'rgba(255,255,255,0.06)', borderRadius: '4px', color: 'var(--text-muted)', fontWeight: 500 }}>
              {paramText}
            </span>
          )}
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace' }}>
            {meta.t_stat.file}  •  {meta.p_value.file}
          </span>
        </div>
        <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem', transition: 'transform 0.2s', transform: expanded ? 'rotate(180deg)' : 'none' }}>▾</span>
      </button>

      {expanded && (
        <div style={{ padding: '0 1.1rem 1.1rem' }}>
          {loading && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', padding: '1rem 0', fontSize: '0.8rem' }}>
              <div className="spinner" /> Loading CSV data…
            </div>
          )}
          {error && (
            <div style={{ padding: '0.75rem', background: 'rgba(239,68,68,0.08)', borderRadius: 8, color: '#ef4444', fontSize: '0.78rem' }}>
              {error} — this CSV may not have been generated for this run.
            </div>
          )}
          {tData && pData && (
            <>
              <SigSummary tData={tData} pData={pData} color={meta.color} />

              <div style={{ fontWeight: 600, fontSize: '0.78rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <span style={{ width: 8, height: 8, background: meta.color, borderRadius: 2, display: 'inline-block' }} />
                T-Statistic per Sample
                <span style={{ fontWeight: 400, color: 'var(--text-muted)', fontSize: '0.7rem' }}>(|t| shown • red = exceeds 4.5)</span>
              </div>
              <TStatChart 
                data={tData} 
                color={meta.color} 
                threshold={4.5} 
                title={`T-Statistic per Sample (${meta.label})`} 
                subtitle="|t| shown • red = exceeds 4.5"
              />

              <div style={{ height: '1rem' }} />

              <div style={{ fontWeight: 600, fontSize: '0.78rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <span style={{ width: 8, height: 8, background: meta.color, borderRadius: 2, display: 'inline-block' }} />
                P-Value per Sample
                <span style={{ fontWeight: 400, color: 'var(--text-muted)', fontSize: '0.7rem' }}>(log₁₀ scale • red = p&lt;0.05 significant • dashed = α=0.05)</span>
              </div>
              <PValueChart 
                data={pData} 
                color={meta.color} 
                title={`P-Value per Sample (${meta.label})`} 
                subtitle="log₁₀ scale • red = p<0.05 significant • dashed = α=0.05"
              />
            </>
          )}
        </div>
      )}
    </div>
  );
}

// -----------------------------------------------------------
// All Filters overlay panel (collapsible)
// -----------------------------------------------------------
function AllFiltersPanel({ analysisId, type, label, color }) {
  const [series, setSeries] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(false);

  const load = async () => {
    if (series !== null) return;
    setLoading(true);
    setError(null);
    try {
      const results = await Promise.all(
        ALL_FILTERS_CONFIG.map(async (c) => {
          const res = await fetch(`${API_BASE}/analyses/${analysisId}/csv-data/filtered/${type}/${c.key}.csv`);
          if (!res.ok) return null;
          const rows = await res.json();
          return {
            key: c.key,
            label: c.label,
            color: c.color,
            data: rows.map(r => r.power_mw ?? 0),
          };
        })
      );
      const valid = results.filter(r => r !== null);
      if (valid.length === 0) throw new Error('No filter CSVs found');
      setSeries(valid);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const toggle = () => {
    if (!expanded) load();
    setExpanded(e => !e);
  };

  return (
    <div style={{
      border: `1px solid ${expanded ? `${color}40` : 'var(--border-color)'}`,
      borderLeft: `3px solid ${color}`,
      borderRadius: 'var(--radius-lg)',
      background: 'var(--bg-tertiary)',
      overflow: 'hidden',
      transition: 'border-color 0.2s',
    }}>
      <button
        onClick={toggle}
        style={{
          width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '0.875rem 1.1rem', background: 'transparent', border: 'none',
          cursor: 'pointer', color: 'var(--text-primary)', fontFamily: 'inherit',
          textAlign: 'left',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <span style={{ width: 10, height: 10, borderRadius: '50%', background: color, display: 'inline-block', flexShrink: 0 }} />
          <span style={{ fontWeight: 700, fontSize: '0.88rem' }}>{label}</span>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace' }}>
            filtered/{type}/*.csv
          </span>
        </div>
        <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem', transition: 'transform 0.2s', transform: expanded ? 'rotate(180deg)' : 'none' }}>▾</span>
      </button>

      {expanded && (
        <div style={{ padding: '0 1.1rem 1.1rem' }}>
          {loading && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', padding: '1rem 0', fontSize: '0.8rem' }}>
              <div className="spinner" /> Loading filters data…
            </div>
          )}
          {error && (
            <div style={{ padding: '0.75rem', background: 'rgba(239,68,68,0.08)', borderRadius: 8, color: '#ef4444', fontSize: '0.78rem' }}>
              {error} — filter files not available for this run.
            </div>
          )}
          {series && (
            <div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                Toggle legends to view specific filters. Displays signal amplitude over time.
              </div>
              <MultiLineChart 
                series={series} 
                leftTitle="Power Amplitude (mW)" 
                height={250} 
                downloadTitle={label}
                downloadSubtitle="Displays signal amplitude over time."
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// -----------------------------------------------------------
// Migration Effect Panel (collapsible)
// -----------------------------------------------------------
function MigrationEffectPanel({ analysisId, color }) {
  const [series, setSeries] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(false);

  const load = async () => {
    if (series !== null) return;
    setLoading(true);
    setError(null);
    try {
      const [fixedRes, randomRes] = await Promise.all([
        fetch(`${API_BASE}/analyses/${analysisId}/csv-data/migration_fixed.csv`),
        fetch(`${API_BASE}/analyses/${analysisId}/csv-data/migration_random.csv`),
      ]);
      if (!fixedRes.ok || !randomRes.ok) throw new Error('Migration CSV files not found');
      
      const fixedRows = await fixedRes.json();
      const randomRows = await randomRes.json();

      setSeries([
        { key: 'fixed', label: 'fixed avg migration rate', color: '#22c55e', data: fixedRows.map(r => r.migration_rate ?? 0) },
        { key: 'random', label: 'random avg migration rate', color: '#f59e0b', data: randomRows.map(r => r.migration_rate ?? 0) }
      ]);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const toggle = () => {
    if (!expanded) load();
    setExpanded(e => !e);
  };

  return (
    <div style={{
      border: `1px solid ${expanded ? `${color}40` : 'var(--border-color)'}`,
      borderLeft: `3px solid ${color}`,
      borderRadius: 'var(--radius-lg)',
      background: 'var(--bg-tertiary)',
      overflow: 'hidden',
      transition: 'border-color 0.2s',
    }}>
      <button
        onClick={toggle}
        style={{
          width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '0.875rem 1.1rem', background: 'transparent', border: 'none',
          cursor: 'pointer', color: 'var(--text-primary)', fontFamily: 'inherit',
          textAlign: 'left',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <span style={{ width: 10, height: 10, borderRadius: '50%', background: color, display: 'inline-block', flexShrink: 0 }} />
          <span style={{ fontWeight: 700, fontSize: '0.88rem' }}>Migration Effect</span>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace' }}>
            migration_fixed.csv  •  migration_random.csv
          </span>
        </div>
        <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem', transition: 'transform 0.2s', transform: expanded ? 'rotate(180deg)' : 'none' }}>▾</span>
      </button>

      {expanded && (
        <div style={{ padding: '0 1.1rem 1.1rem' }}>
          {loading && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', padding: '1rem 0', fontSize: '0.8rem' }}>
              <div className="spinner" /> Loading migration rates…
            </div>
          )}
          {error && (
            <div style={{ padding: '0.75rem', background: 'rgba(239,68,68,0.08)', borderRadius: 8, color: '#ef4444', fontSize: '0.78rem' }}>
              {error} — migration files not available.
            </div>
          )}
          {series && (
            <div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                Average task migration rate per sample index (average across traces).
              </div>
              <MultiLineChart 
                series={series} 
                leftTitle="Avg Migration Event Rate" 
                forceZeroMin={true} 
                height={220} 
                downloadTitle="Migration Effect"
                downloadSubtitle="Average task migration rate per sample index (average across traces)."
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// -----------------------------------------------------------
// TVLA vs Migration Overlay Panel (collapsible)
// -----------------------------------------------------------
function MigrationOverlayPanel({ analysisId, color }) {
  const [series, setSeries] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(false);

  const load = async () => {
    if (series !== null) return;
    setLoading(true);
    setError(null);
    try {
      const [tRes, fixedRes, randomRes] = await Promise.all([
        fetch(`${API_BASE}/analyses/${analysisId}/csv-data/tvla_t_stat_regression_residual.csv`),
        fetch(`${API_BASE}/analyses/${analysisId}/csv-data/migration_fixed.csv`),
        fetch(`${API_BASE}/analyses/${analysisId}/csv-data/migration_random.csv`),
      ]);
      if (!tRes.ok || !fixedRes.ok || !randomRes.ok) throw new Error('CSVs required for overlay not found');
      
      const tRows = await tRes.json();
      const fixedRows = await fixedRes.json();
      const randomRows = await randomRes.json();

      const commonLen = Math.min(tRows.length, fixedRows.length, randomRows.length);

      const tStatsAbs = tRows.slice(0, commonLen).map(r => Math.abs(r.t_stat_regression_residual ?? 0));
      const fixedRate = fixedRows.slice(0, commonLen).map(r => r.migration_rate ?? 0);
      const randomRate = randomRows.slice(0, commonLen).map(r => r.migration_rate ?? 0);
      const migrationGap = fixedRate.map((v, i) => Math.abs(v - randomRate[i]));

      setSeries([
        { key: 't_stat', label: '|t-statistic| (Residual)', color: '#3b82f6', data: tStatsAbs, yAxis: 'left' },
        { key: 'fixed_rate', label: 'fixed migration rate', color: '#22c55e', data: fixedRate, yAxis: 'right' },
        { key: 'random_rate', label: 'random migration rate', color: '#f59e0b', data: randomRate, yAxis: 'right' },
        { key: 'gap', label: '|fixed-random| migration gap', color: '#ef4444', data: migrationGap, yAxis: 'right', isDashed: true }
      ]);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const toggle = () => {
    if (!expanded) load();
    setExpanded(e => !e);
  };

  return (
    <div style={{
      border: `1px solid ${expanded ? `${color}40` : 'var(--border-color)'}`,
      borderLeft: `3px solid ${color}`,
      borderRadius: 'var(--radius-lg)',
      background: 'var(--bg-tertiary)',
      overflow: 'hidden',
      transition: 'border-color 0.2s',
    }}>
      <button
        onClick={toggle}
        style={{
          width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '0.875rem 1.1rem', background: 'transparent', border: 'none',
          cursor: 'pointer', color: 'var(--text-primary)', fontFamily: 'inherit',
          textAlign: 'left',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <span style={{ width: 10, height: 10, borderRadius: '50%', background: color, display: 'inline-block', flexShrink: 0 }} />
          <span style={{ fontWeight: 700, fontSize: '0.88rem' }}>Migration Overlay (TVLA vs Migration)</span>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace' }}>
            tvla_t_stat_regression_residual.csv  •  migration_*.csv
          </span>
        </div>
        <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem', transition: 'transform 0.2s', transform: expanded ? 'rotate(180deg)' : 'none' }}>▾</span>
      </button>

      {expanded && (
        <div style={{ padding: '0 1.1rem 1.1rem' }}>
          {loading && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', padding: '1rem 0', fontSize: '0.8rem' }}>
              <div className="spinner" /> Loading overlay data…
            </div>
          )}
          {error && (
            <div style={{ padding: '0.75rem', background: 'rgba(239,68,68,0.08)', borderRadius: 8, color: '#ef4444', fontSize: '0.78rem' }}>
              {error} — overlay files not available.
            </div>
          )}
          {series && (
            <div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                Overlay showing absolute TVLA t-statistic (left axis) vs migration rates (right axis).
              </div>
              <MultiLineChart
                series={series}
                leftTitle="|t-statistic|"
                rightTitle="Migration Event Rate"
                forceZeroMin={true}
                thresholds={[
                  { value: 4.5, color: '#3b82f6', dasharray: '4 3', label: 'TVLA threshold (4.5)', yAxis: 'left' }
                ]}
                height={260}
                downloadTitle="Migration Overlay (TVLA vs Migration)"
                downloadSubtitle="Overlay showing absolute TVLA t-statistic (left axis) vs migration rates (right axis)."
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// -----------------------------------------------------------
// T-Statistic Comparison Panel (collapsible)
// -----------------------------------------------------------
function TStatComparisonPanel({ analysisId, color }) {
  const [series, setSeries] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(false);

  const load = async () => {
    if (series !== null) return;
    setLoading(true);
    setError(null);
    try {
      const results = await Promise.all(
        Object.entries(FILTER_CSV_MAP).map(async ([key, meta]) => {
          const res = await fetch(`${API_BASE}/analyses/${analysisId}/csv-data/${meta.t_stat.file}`);
          if (!res.ok) return null;
          const rows = await res.json();
          return {
            key,
            label: meta.label,
            color: meta.color,
            data: rows.map(r => Math.abs(r[meta.t_stat.col] ?? 0)),
          };
        })
      );
      const valid = results.filter(r => r !== null);
      if (valid.length === 0) throw new Error('No t-statistic CSVs found');
      setSeries(valid);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const toggle = () => {
    if (!expanded) load();
    setExpanded(e => !e);
  };

  return (
    <div style={{
      border: `1px solid ${expanded ? `${color}40` : 'var(--border-color)'}`,
      borderLeft: `3px solid ${color}`,
      borderRadius: 'var(--radius-lg)',
      background: 'var(--bg-tertiary)',
      overflow: 'hidden',
      transition: 'border-color 0.2s',
    }}>
      <button
        onClick={toggle}
        style={{
          width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '0.875rem 1.1rem', background: 'transparent', border: 'none',
          cursor: 'pointer', color: 'var(--text-primary)', fontFamily: 'inherit',
          textAlign: 'left',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <span style={{ width: 10, height: 10, borderRadius: '50%', background: color, display: 'inline-block', flexShrink: 0 }} />
          <span style={{ fontWeight: 700, fontSize: '0.88rem' }}>T-Statistic Comparison (All Filters)</span>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace' }}>
            tvla_t_stat*.csv
          </span>
        </div>
        <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem', transition: 'transform 0.2s', transform: expanded ? 'rotate(180deg)' : 'none' }}>▾</span>
      </button>

      {expanded && (
        <div style={{ padding: '0 1.1rem 1.1rem' }}>
          {loading && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', padding: '1rem 0', fontSize: '0.8rem' }}>
              <div className="spinner" /> Loading comparison data…
            </div>
          )}
          {error && (
            <div style={{ padding: '0.75rem', background: 'rgba(239,68,68,0.08)', borderRadius: 8, color: '#ef4444', fontSize: '0.78rem' }}>
              {error} — comparison files not available.
            </div>
          )}
          {series && (
            <div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                Compare absolute t-statistic values (|t|) across all 5 filters. Dashed red line = TVLA threshold (4.5).
              </div>
              <MultiLineChart
                series={series}
                leftTitle="|t-statistic|"
                forceZeroMin={true}
                thresholds={[
                  { value: 4.5, color: '#ef4444', dasharray: '4 3', label: 'Threshold (4.5)', yAxis: 'left' }
                ]}
                height={260}
                downloadTitle="T-Statistic Comparison (All Filters)"
                downloadSubtitle="Compare absolute t-statistic values (|t|) across all 5 filters."
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// -----------------------------------------------------------
// Main export: Per-Sample Charts Section
// -----------------------------------------------------------
export default function PerSampleCharts({ analysisId, summary }) {
  if (!analysisId) return null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      {Object.entries(FILTER_CSV_MAP).map(([key, meta]) => (
        <FilterPanel key={key} analysisId={analysisId} filterKey={key} meta={meta} summary={summary} />
      ))}
      <TStatComparisonPanel analysisId={analysisId} color="#f43f5e" />
      <AllFiltersPanel analysisId={analysisId} type="fixed" label="Fixed – Filters" color="#a855f7" />
      <AllFiltersPanel analysisId={analysisId} type="random" label="Random – Filters" color="#ec4899" />
      <MigrationEffectPanel analysisId={analysisId} color="#14b8a6" />
      <MigrationOverlayPanel analysisId={analysisId} color="#3b82f6" />
    </div>
  );
}
