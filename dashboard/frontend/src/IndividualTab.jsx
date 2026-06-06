import React, { useState, useEffect } from 'react';
import {
  Activity, AlertTriangle, BarChart2, CheckCircle, Image as ImageIcon,
  Download, Info, Cpu, ArrowUpDown, TrendingUp, Zap, TableProperties
} from 'lucide-react';
import PerSampleCharts from './PerSampleCharts';

const API_BASE = 'http://localhost:8000/api';

// ============================================================
// Metric Card
// ============================================================
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

// ============================================================
// Filter metadata
// ============================================================
const FILTER_META = [
  { key: 'raw',                label: 'Raw',                color: '#3b82f6' },
  { key: 'median',             label: 'Median',             color: '#8b5cf6' },
  { key: 'moving_average',     label: 'Moving Average',     color: '#06b6d4' },
  { key: 'savitzky_golay',     label: 'Savitzky–Golay',     color: '#ec4899' },
  { key: 'wavelet',            label: 'Wavelet',            color: '#f59e0b' },
  { key: 'regression_residual',label: 'Regression Residual',color: '#22c55e' },
];

// ============================================================
// Filter Metric Section
// ============================================================
function FilterMetricSection({ qm }) {
  if (!qm) return (
    <div className="empty-state" style={{ padding: '1.5rem' }}>
      <Info size={24} className="empty-state-icon" />
      <span className="empty-state-desc">Quantitative metrics not available for this run (older schema).</span>
    </div>
  );
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {FILTER_META.map(f => {
        const m = qm[f.key];
        if (!m) return null;
        const leaking = (m.max_abs_t_statistic ?? 0) > 4.5;
        return (
          <div key={f.key} style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--border-color)', borderLeft: `3px solid ${f.color}`, borderRadius: 'var(--radius-lg)', padding: '1.25rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <div style={{ fontWeight: 700, fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ width: 10, height: 10, borderRadius: '50%', background: f.color, display: 'inline-block' }} />
                {f.label}
              </div>
              <span className={`badge ${leaking ? 'badge-danger' : 'badge-success'}`}>
                {leaking ? '⚠ Leakage Detected' : '✓ Clean'}
              </span>
            </div>
            <div className="metrics-grid">
              <MetricCard label="Max |t|" icon={<BarChart2 />} iconColor={f.color} value={(m.max_abs_t_statistic ?? 0).toFixed(3)} highlight={leaking} />
              <MetricCard label="Exceedance" icon={<AlertTriangle />} iconColor="#ef4444" value={`${(m.exceedance_percent ?? 0).toFixed(1)}%`} sub={`${m.samples_exceeding_threshold ?? 0} samples`} />
              <MetricCard label="Power SNR" icon={<Activity />} iconColor="#22c55e" value={(m.power_snr ?? 0).toFixed(4)} />
              <MetricCard label="Δ Power (mW)" icon={<Zap />} iconColor="#f59e0b" value={m.mean_power_difference_mw != null ? m.mean_power_difference_mw.toFixed(0) : '—'} sub="Fixed vs Random mean" />
              <MetricCard label="Samples" icon={<CheckCircle />} iconColor="#a855f7" value={m.samples ?? '—'} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ============================================================
// Decision Matrix
// ============================================================
function DecisionMatrix({ dm }) {
  if (!dm) return (
    <div className="empty-state" style={{ padding: '1.5rem' }}>
      <Info size={24} className="empty-state-icon" />
      <span className="empty-state-desc">Decision matrix not available for this run.</span>
    </div>
  );
  const decisions = [
    { key: 'decision_1_environment_control', label: 'D1 – Environment Control', icon: '🌡️' },
    { key: 'decision_2_filter_rq5', label: 'D2 – Filter Impact (RQ5)', icon: '🔬' },
    { key: 'decision_3_data_vs_migration_rq1_rq4', label: 'D3 – Data vs Migration (RQ1/RQ4)', icon: '🔀' },
    { key: 'decision_4_big_vs_little_rq3', label: 'D4 – Big vs Little Core (RQ3)', icon: '💻' },
  ];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.875rem' }}>
      {decisions.map(({ key, label, icon }) => {
        const d = dm[key];
        if (!d) return null;
        if (key === 'decision_2_filter_rq5') {
          return (
            <div key={key} className="decision-card">
              <div className="decision-title">{icon} {label}</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
                {d.comparisons && Object.entries(d.comparisons).map(([filter, comp]) => (
                  <div key={filter} style={{ background: 'var(--bg-primary)', borderRadius: 'var(--radius-sm)', padding: '0.6rem 0.875rem', display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.5rem' }}>
                    <span style={{ fontWeight: 600, fontSize: '0.78rem', minWidth: 90, textTransform: 'capitalize' }}>{filter.replace('_', ' ')}</span>
                    <span className={`badge ${comp.delta_exceedance_rate > 0 ? 'badge-danger' : 'badge-success'}`}>
                      Δ {comp.delta_exceedance_rate > 0 ? '+' : ''}{(comp.delta_exceedance_rate * 100).toFixed(1)}%
                    </span>
                    <span className="decision-rule" style={{ flex: 1 }}>{comp.decision}</span>
                  </div>
                ))}
              </div>
            </div>
          );
        }
        return (
          <div key={key} className="decision-card">
            <div className="decision-title">{icon} {label}</div>
            {d.rule && <div className="decision-rule">{d.rule}</div>}
            {d.verdict && (
              <div style={{ marginTop: '0.25rem' }}>
                <span className={`badge ${d.applicable === false ? 'badge-neutral' : d.verdict.includes('noisy') ? 'badge-danger' : d.verdict.includes('valid') ? 'badge-success' : 'badge-neutral'}`}>
                  {d.verdict}
                </span>
              </div>
            )}
            {d.migration_alignment && (
              <div style={{ display: 'flex', gap: '1rem', marginTop: '0.25rem', flexWrap: 'wrap' }}>
                <span className="text-xs text-muted">Corr(|t|, migration): <b style={{ color: 'var(--text-secondary)' }}>{d.migration_alignment.correlation_abs_t_vs_migration_gap?.toFixed(4) ?? '—'}</b></span>
                <span className="text-xs text-muted">Max migration gap: <b style={{ color: 'var(--text-secondary)' }}>{d.migration_alignment.max_migration_rate_gap?.toFixed(4) ?? '—'}</b></span>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ============================================================
// Main IndividualTab
// ============================================================
export default function IndividualTab({ analysisId, summary }) {
  if (!analysisId) {
    return (
      <div className="empty-state">
        <Activity size={40} className="empty-state-icon" />
        <div className="empty-state-title">No Run Selected</div>
        <div className="empty-state-desc">Select an analysis run from the sidebar or click a row in the Overview tab.</div>
      </div>
    );
  }
  if (!summary) {
    return (
      <div className="empty-state">
        <div className="spinner" />
        <div className="empty-state-title">Loading analysis data…</div>
      </div>
    );
  }

  const qm = summary.quantitative_metrics ?? null;
  const dm = summary.decision_matrix ?? null;
  const migration = summary.migration_alignment ?? null;
  const autoTuned = summary.auto_tuned_parameters ?? null;
  const downloadCsv = (name) => window.open(`${API_BASE}/analyses/${analysisId}/csv/${name}`, '_blank');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 800, letterSpacing: '-0.02em' }}>{analysisId}</h2>
          <div className="text-xs text-muted" style={{ marginTop: '0.25rem', display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
            {summary.fixed_traces != null && <span>Fixed: <b>{summary.fixed_traces}</b> traces</span>}
            {summary.random_traces != null && <span>Random: <b>{summary.random_traces}</b> traces</span>}
            {summary.tvla_threshold != null && <span>Threshold: <b>|t| = {summary.tvla_threshold}</b></span>}
          </div>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
          <button className="btn btn-ghost" onClick={() => downloadCsv('quantitative_metrics.csv')}>
            <Download size={14} /> Metrics CSV
          </button>
          <button className="btn btn-ghost" onClick={() => downloadCsv('tvla_t_stat.csv')}>
            <Download size={14} /> T-Stat CSV
          </button>
        </div>
      </div>

      {/* Filter Metrics */}
      <div className="page-section">
        <div className="section-header"><BarChart2 size={18} color="var(--accent-primary)" />Filter Metrics (All 5 Types)</div>
        <FilterMetricSection qm={qm} />
      </div>

      {/* Per-Sample & Multi-Line Analysis Charts */}
      <div className="page-section">
        <div className="section-header"><TableProperties size={18} color="var(--accent-warning)" />Per-Sample & Multi-Line Analysis</div>
        <p className="section-desc">Interactive charts drawn directly from raw analysis data. Expand panels to inspect TVLA t-statistics, p-values, filtered signals, task migration profiles, and overlays with togglable legends.</p>
        <PerSampleCharts analysisId={analysisId} summary={summary} />
      </div>

      {/* Decision Matrix */}
      <div className="page-section">
        <div className="section-header"><CheckCircle size={18} color="var(--accent-success)" />Decision Matrix</div>
        <DecisionMatrix dm={dm} />
      </div>

      {/* Migration Info */}
      {migration && (
        <div className="page-section">
          <div className="section-header"><ArrowUpDown size={18} color="var(--accent-cyan)" />Migration Analysis Summary</div>
          <div className="metrics-grid">
            <MetricCard label="Samples Analyzed" value={migration.samples ?? '—'} icon={<Activity />} iconColor="var(--accent-cyan)" />
            <MetricCard label="Corr |t| vs Migration" value={(migration.correlation_abs_t_vs_migration_gap ?? 0).toFixed(4)} icon={<TrendingUp />} iconColor="var(--accent-purple)" />
            <MetricCard label="Max Migration Gap" value={(migration.max_migration_rate_gap ?? 0).toFixed(4)} icon={<ArrowUpDown />} iconColor="var(--accent-warning)" />
            {summary.mean_fixed_migration_events != null && <MetricCard label="Fixed Migrations (avg)" value={summary.mean_fixed_migration_events?.toFixed(2)} icon={<Cpu />} iconColor="var(--accent-primary)" />}
            {summary.mean_random_migration_events != null && <MetricCard label="Random Migrations (avg)" value={summary.mean_random_migration_events?.toFixed(2)} icon={<Cpu />} iconColor="var(--accent-purple)" />}
          </div>
        </div>
      )}

      {/* Auto-tuned params */}
      {autoTuned && (
        <div className="page-section">
          <div className="section-header"><Zap size={18} color="var(--accent-warning)" />Auto-Tuned Parameters</div>
          <div className="metrics-grid">
            <MetricCard label="Lowpass Cutoff Ratio" value={autoTuned.lowpass_cutoff_ratio} icon={<Activity />} iconColor="var(--accent-warning)" />
            <MetricCard label="Savitzky–Golay Window" value={autoTuned.savitzky_golay_window} icon={<BarChart2 />} iconColor="var(--accent-cyan)" />
          </div>
        </div>
      )}

    </div>
  );
}
