import React, { useState, useRef, useEffect } from 'react';

export function MultiLineChart({
  series, // Array of { key, label, data (array of numbers), color, yAxis: 'left' | 'right', isDashed?: boolean }
  leftTitle = '',
  rightTitle = '',
  thresholds = [], // Array of { value, color, dasharray, label, yAxis: 'left' | 'right' }
  height = 220,
  forceZeroMin = false,
}) {
  const svgRef = useRef(null);
  const [visibleKeys, setVisibleKeys] = useState({});
  const [tooltip, setTooltip] = useState(null);

  // Initialize visibility state when series changes
  useEffect(() => {
    const initial = {};
    series.forEach(s => {
      initial[s.key] = true;
    });
    setVisibleKeys(initial);
  }, [series]);

  const toggleLine = (key) => {
    setVisibleKeys(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  const activeSeries = series.filter(s => visibleKeys[s.key]);
  const hasRightAxis = activeSeries.some(s => s.yAxis === 'right');

  // Compute domains
  const getDomain = (yAxisName, fallbackMin = 0, fallbackMax = 1) => {
    const axisSeries = activeSeries.filter(s => (s.yAxis || 'left') === yAxisName);
    if (axisSeries.length === 0) return [fallbackMin, fallbackMax];

    let minVal = Infinity;
    let maxVal = -Infinity;
    axisSeries.forEach(s => {
      s.data.forEach(v => {
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      });
    });

    // include thresholds in domain if applicable
    thresholds.forEach(t => {
      if ((t.yAxis || 'left') === yAxisName) {
        if (t.value < minVal) minVal = t.value;
        if (t.value > maxVal) maxVal = t.value;
      }
    });

    if (minVal === Infinity) return [fallbackMin, fallbackMax];
    
    const isZeroForced = forceZeroMin && minVal >= 0;
    const effectiveMin = isZeroForced ? 0 : minVal;

    const diff = maxVal - effectiveMin;
    const margin = diff === 0 ? 1 : diff * 0.15;
    
    return [
      isZeroForced ? 0 : effectiveMin - margin,
      maxVal + margin
    ];
  };

  const domainL = getDomain('left', -5, 5);
  const domainR = hasRightAxis ? getDomain('right', 0, 1) : [0, 1];

  const W = 800;
  const H = height;
  const PAD = { top: 20, right: hasRightAxis ? 65 : 20, bottom: 25, left: 65 };
  const iW = W - PAD.left - PAD.right;
  const iH = H - PAD.top - PAD.bottom;

  // Data length (assume all series have same length)
  const n = series[0]?.data.length || 0;

  // Position helpers
  const xPos = (i) => PAD.left + (i / Math.max(1, n - 1)) * iW;
  
  const yPosL = (val) => {
    const [minY, maxY] = domainL;
    return PAD.top + iH - ((val - minY) / (maxY - minY)) * iH;
  };

  const yPosR = (val) => {
    const [minY, maxY] = domainR;
    return PAD.top + iH - ((val - minY) / (maxY - minY)) * iH;
  };

  const getY = (val, yAxis) => {
    return yAxis === 'right' ? yPosR(val) : yPosL(val);
  };

  // Mouse move handler for interactive tooltip
  const handleMouseMove = (e) => {
    if (!svgRef.current || n === 0) return;
    const rect = svgRef.current.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const clientY = e.clientY - rect.top;

    // Convert pixel position to sample index
    const pct = (clientX - (PAD.left / W) * rect.width) / ((iW / W) * rect.width);
    let idx = Math.round(pct * (n - 1));
    idx = Math.max(0, Math.min(n - 1, idx));

    // Gather values at this index
    const vals = series.map(s => ({
      key: s.key,
      label: s.label,
      val: s.data[idx],
      color: s.color,
      visible: visibleKeys[s.key],
      yAxis: s.yAxis || 'left'
    }));

    setTooltip({
      idx,
      x: e.clientX,
      y: e.clientY,
      vals
    });
  };

  const handleMouseLeave = () => {
    setTooltip(null);
  };

  // Grid tick generation
  const getTicks = ([minVal, maxVal]) => {
    const ticks = [];
    const count = 5;
    for (let i = 0; i < count; i++) {
      ticks.push(minVal + (i / (count - 1)) * (maxVal - minVal));
    }
    return ticks;
  };

  const ticksL = getTicks(domainL);
  const ticksR = hasRightAxis ? getTicks(domainR) : [];

  return (
    <div style={{ background: 'var(--bg-secondary)', borderRadius: 8, padding: '1rem', border: '1px solid var(--border-color)' }}>
      <div style={{ position: 'relative' }}>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          style={{ width: '100%', height: 'auto', display: 'block', overflow: 'visible' }}
          ref={svgRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        >
          {/* Grid lines & Left Axis */}
          {ticksL.map((val, i) => {
            const y = yPosL(val);
            if (isNaN(y)) return null;
            return (
              <g key={`grid-l-${i}`}>
                <line x1={PAD.left} y1={y} x2={W - PAD.right} y2={y} stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
                <text x={PAD.left - 6} y={y + 3} textAnchor="end" fontSize="8" fill="var(--text-muted)" fontFamily="monospace">
                  {val.toFixed(2)}
                </text>
              </g>
            );
          })}

          {/* Left axis title */}
          {leftTitle && (
            <text
              x={10}
              y={PAD.top + iH / 2}
              textAnchor="middle"
              fontSize="9"
              fontWeight="600"
              fill="var(--text-muted)"
              transform={`rotate(-90, 10, ${PAD.top + iH / 2})`}
            >
              {leftTitle}
            </text>
          )}

          {/* Right Axis */}
          {hasRightAxis && ticksR.map((val, i) => {
            const y = yPosR(val);
            if (isNaN(y)) return null;
            return (
              <g key={`grid-r-${i}`}>
                <text x={W - PAD.right + 6} y={y + 3} textAnchor="start" fontSize="8" fill="var(--text-muted)" fontFamily="monospace">
                  {val.toFixed(3)}
                </text>
              </g>
            );
          })}

          {/* Right axis title */}
          {hasRightAxis && rightTitle && (
            <text
              x={W - 10}
              y={PAD.top + iH / 2}
              textAnchor="middle"
              fontSize="9"
              fontWeight="600"
              fill="var(--text-muted)"
              transform={`rotate(90, ${W - 10}, ${PAD.top + iH / 2})`}
            >
              {rightTitle}
            </text>
          )}

          {/* Threshold lines */}
          {thresholds.map((t, idx) => {
            const y = getY(t.value, t.yAxis || 'left');
            if (isNaN(y)) return null;
            return (
              <g key={`thresh-${idx}`}>
                <line
                  x1={PAD.left}
                  y1={y}
                  x2={W - PAD.right}
                  y2={y}
                  stroke={t.color}
                  strokeWidth="1.2"
                  strokeDasharray={t.dasharray || '4 3'}
                  opacity="0.8"
                />
                <text
                  x={t.yAxis === 'right' ? W - PAD.right - 2 : PAD.left + 2}
                  y={y - 4}
                  fontSize="7.5"
                  fill={t.color}
                  textAnchor={t.yAxis === 'right' ? 'end' : 'start'}
                >
                  {t.label}
                </text>
              </g>
            );
          })}

          {/* Render active lines */}
          {activeSeries.map(s => {
            const points = s.data.map((val, i) => `${xPos(i)},${getY(val, s.yAxis || 'left')}`).join(' ');
            return (
              <polyline
                key={s.key}
                fill="none"
                stroke={s.color}
                strokeWidth="1.5"
                strokeDasharray={s.isDashed ? '4 3' : undefined}
                points={points}
                opacity="0.85"
              />
            );
          })}

          {/* Hover indicator line */}
          {tooltip && (
            <line
              x1={xPos(tooltip.idx)}
              y1={PAD.top}
              x2={xPos(tooltip.idx)}
              y2={PAD.top + iH}
              stroke="rgba(255,255,255,0.25)"
              strokeWidth="1"
              strokeDasharray="3 3"
            />
          )}

          {/* X axis index labels */}
          {[0, Math.floor(n / 4), Math.floor(n / 2), Math.floor(3 * n / 4), n - 1].map(i => {
            const x = xPos(i);
            if (isNaN(x)) return null;
            return (
              <text key={`x-lbl-${i}`} x={x} y={H - 5} textAnchor="middle" fontSize="8" fill="var(--text-muted)" fontFamily="monospace">
                {i}
              </text>
            );
          })}
        </svg>

        {/* Hover Tooltip Box */}
        {tooltip && (
          <div style={{
            position: 'fixed',
            left: tooltip.x + 15,
            top: tooltip.y - 45,
            background: 'var(--bg-active)',
            border: '1px solid var(--border-hover)',
            borderRadius: 6,
            padding: '8px 10px',
            fontSize: '0.73rem',
            pointerEvents: 'none',
            zIndex: 9999,
            boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
            color: 'var(--text-primary)',
            display: 'flex',
            flexDirection: 'column',
            gap: '4px',
            minWidth: 160
          }}>
            <div style={{ fontWeight: 700, borderBottom: '1px solid var(--border-color)', paddingBottom: '3px', marginBottom: '3px', color: 'var(--text-muted)' }}>
              Sample Index: {tooltip.idx}
            </div>
            {tooltip.vals.map(v => (
              <div key={v.key} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '10px', opacity: v.visible ? 1 : 0.4 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <span style={{ width: 6, height: 6, borderRadius: '50%', background: v.color, display: 'inline-block' }} />
                  <span>{v.label}</span>
                </div>
                <span style={{ fontWeight: 700, fontFamily: 'monospace' }}>
                  {v.val != null ? v.val.toFixed(v.yAxis === 'right' ? 4 : 2) : '—'}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Legend with interactive toggles */}
      <div style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: '0.5rem',
        marginTop: '0.75rem',
        justifyContent: 'center',
        borderTop: '1px solid var(--border-color)',
        paddingTop: '0.75rem'
      }}>
        {series.map(s => {
          const isVisible = visibleKeys[s.key] !== false;
          return (
            <button
              key={s.key}
              onClick={() => toggleLine(s.key)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.35rem',
                padding: '4px 8px',
                borderRadius: '4px',
                background: isVisible ? 'var(--bg-tertiary)' : 'transparent',
                border: `1px solid ${isVisible ? 'var(--border-hover)' : 'transparent'}`,
                color: isVisible ? 'var(--text-primary)' : 'var(--text-muted)',
                fontSize: '0.7rem',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
                userSelect: 'none'
              }}
            >
              <span style={{
                width: 7,
                height: 7,
                borderRadius: '50%',
                background: s.color,
                display: 'inline-block',
                opacity: isVisible ? 1 : 0.3
              }} />
              <span style={{ textDecoration: isVisible ? 'none' : 'line-through' }}>{s.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
