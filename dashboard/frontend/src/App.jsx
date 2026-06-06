import React, { useState, useEffect, useCallback } from 'react';
import {
  Activity, BarChart2, Play, Database, ChevronDown,
  LayoutDashboard, Search, Zap
} from 'lucide-react';
import OverviewTab from './OverviewTab';
import IndividualTab from './IndividualTab';
import RunAnalysisTab from './RunAnalysisTab';

const API_BASE = 'http://localhost:8000/api';

function formatDate(ts) {
  return new Date(ts * 1000).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

const TABS = [
  { id: 'overview',    label: 'Overview',      icon: LayoutDashboard },
  { id: 'individual',  label: 'Individual Run', icon: Search },
  { id: 'run',         label: 'Run Analysis',   icon: Zap },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [analyses, setAnalyses] = useState([]);
  const [overviewData, setOverviewData] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [summary, setSummary] = useState(null);
  const [sidebarSearch, setSidebarSearch] = useState('');

  // Fetch sidebar list and overview
  const fetchAll = useCallback(async () => {
    try {
      const [listRes, overviewRes] = await Promise.all([
        fetch(`${API_BASE}/analyses`),
        fetch(`${API_BASE}/analyses/overview`),
      ]);
      const list = await listRes.json();
      const overview = await overviewRes.json();
      setAnalyses(list);
      setOverviewData(overview);
      if (list.length > 0 && !selectedId) {
        setSelectedId(list[0].id);
      }
    } catch (e) {
      console.error('Failed to fetch analyses', e);
    }
  }, [selectedId]);

  useEffect(() => { fetchAll(); }, []);

  // Fetch summary whenever selectedId changes
  useEffect(() => {
    if (!selectedId) { setSummary(null); return; }
    setSummary(null);
    fetch(`${API_BASE}/analyses/${selectedId}/summary`)
      .then(r => r.json())
      .then(d => setSummary(d))
      .catch(() => setSummary(null));
  }, [selectedId]);

  const handleSelectId = (id) => {
    setSelectedId(id);
    setActiveTab('individual');
  };

  const filteredAnalyses = analyses.filter(a =>
    a.id.toLowerCase().includes(sidebarSearch.toLowerCase())
  );

  // Determine leakage badge for sidebar items
  const overviewMap = {};
  overviewData.forEach(d => { overviewMap[d.id] = d; });

  const getBadge = (id) => {
    const d = overviewMap[id];
    if (!d || !d.has_new_schema) return 'unknown';
    return (d.max_t_raw ?? 0) > 4.5 ? 'leaking' : 'clean';
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <div className="sidebar-logo-icon">
              <Activity size={16} color="white" />
            </div>
            TVLA Dashboard
          </div>
          <div className="sidebar-subtitle">{analyses.length} analysis runs</div>
          <input
            type="text"
            placeholder="Search runs…"
            value={sidebarSearch}
            onChange={e => setSidebarSearch(e.target.value)}
            style={{
              background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
              borderRadius: 'var(--radius-sm)', padding: '0.4rem 0.7rem',
              color: 'var(--text-primary)', fontSize: '0.78rem', width: '100%',
              outline: 'none', fontFamily: 'inherit'
            }}
          />
        </div>

        <div className="sidebar-runs-list">
          {filteredAnalyses.map(a => {
            const badge = getBadge(a.id);
            const row = overviewMap[a.id];
            return (
              <div
                key={a.id}
                className={`run-item ${selectedId === a.id ? 'active' : ''}`}
                onClick={() => handleSelectId(a.id)}
              >
                <div className="run-item-id">{a.id.replace('analysis_', '')}</div>
                <div className="run-item-date">{formatDate(a.created_at)}</div>
                <div style={{ display: 'flex', gap: '0.35rem', marginTop: '4px', flexWrap: 'wrap' }}>
                  <span className={`run-item-badge ${badge}`}>
                    {badge === 'leaking' ? '⚠ Leakage' : badge === 'clean' ? '✓ Clean' : '○ Legacy'}
                  </span>
                  {row?.max_t_raw != null && (
                    <span className="run-item-badge" style={{
                      background: 'rgba(59,130,246,0.1)', color: 'var(--accent-primary)',
                      border: '1px solid rgba(59,130,246,0.2)', fontSize: '0.6rem', fontWeight: 700
                    }}>
                      |t|={row.max_t_raw.toFixed(2)}
                    </span>
                  )}
                </div>
              </div>
            );
          })}
          {filteredAnalyses.length === 0 && (
            <div className="text-muted text-xs" style={{ padding: '1rem', textAlign: 'center' }}>
              No runs match search.
            </div>
          )}
        </div>

        <div className="sidebar-footer">
          <button
            className="btn btn-ghost w-full"
            style={{ justifyContent: 'center' }}
            onClick={() => { setActiveTab('run'); }}
          >
            <Zap size={14} /> New Analysis
          </button>
        </div>
      </div>

      {/* Main */}
      <div className="main-content">
        {/* Tab Bar */}
        <div className="tab-bar">
          {TABS.map(tab => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => setActiveTab(tab.id)}
              >
                <Icon size={15} />
                {tab.label}
              </button>
            );
          })}
          {selectedId && activeTab === 'individual' && (
            <div style={{
              marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '0.5rem',
              fontSize: '0.72rem', color: 'var(--text-muted)'
            }}>
              <span>Viewing:</span>
              <span style={{ fontFamily: 'JetBrains Mono, monospace', color: 'var(--text-secondary)', fontWeight: 600 }}>
                {selectedId.replace('analysis_', '')}
              </span>
            </div>
          )}
        </div>

        {/* Tab Content */}
        <div className="tab-content">
          {activeTab === 'overview' && (
            <OverviewTab
              overviewData={overviewData}
              selectedId={selectedId}
              onSelectId={handleSelectId}
            />
          )}
          {activeTab === 'individual' && (
            <IndividualTab
              analysisId={selectedId}
              summary={summary}
            />
          )}
          {activeTab === 'run' && (
            <RunAnalysisTab onSuccess={() => { fetchAll(); setActiveTab('overview'); }} />
          )}
        </div>
      </div>
    </div>
  );
}
