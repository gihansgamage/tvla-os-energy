import React, { useEffect, useRef, useState } from 'react';

// ============================================================
// LIGHTBOX MODAL
// ============================================================
export function PlotModal({ items, startIndex, onClose }) {
  const [idx, setIdx] = useState(startIndex ?? 0);
  const current = items?.[idx];

  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'Escape') onClose();
      if (e.key === 'ArrowRight') setIdx(i => Math.min(i + 1, items.length - 1));
      if (e.key === 'ArrowLeft') setIdx(i => Math.max(i - 1, 0));
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [onClose, items]);

  if (!current) return null;

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 1000,
        background: 'rgba(0,0,0,0.88)',
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        backdropFilter: 'blur(8px)',
        padding: '2rem',
      }}
    >
      {/* Header */}
      <div
        onClick={e => e.stopPropagation()}
        style={{
          width: '100%', maxWidth: 1100,
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          marginBottom: '0.875rem',
        }}
      >
        <div>
          <div style={{ fontWeight: 700, fontSize: '1rem', color: 'var(--text-primary)' }}>
            {current.label}
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '2px' }}>
            {idx + 1} / {items.length} — use ← → arrow keys or buttons to navigate
          </div>
        </div>
        <button
          onClick={onClose}
          style={{
            background: 'rgba(255,255,255,0.1)', border: '1px solid rgba(255,255,255,0.15)',
            borderRadius: 8, padding: '0.4rem 0.8rem', color: 'var(--text-primary)',
            cursor: 'pointer', fontFamily: 'inherit', fontSize: '0.8rem',
          }}
        >
          ✕ Close
        </button>
      </div>

      {/* Content */}
      <div
        onClick={e => e.stopPropagation()}
        style={{
          width: '100%', maxWidth: 1100,
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
          borderRadius: 14,
          overflow: 'auto',
          maxHeight: 'calc(100vh - 10rem)',
          padding: '1.5rem',
        }}
      >
        {current.renderContent({ isModal: true })}
      </div>

      {/* Nav arrows */}
      {items.length > 1 && (
        <div
          onClick={e => e.stopPropagation()}
          style={{ display: 'flex', gap: '0.75rem', marginTop: '1rem' }}
        >
          <button
            onClick={() => setIdx(i => Math.max(i - 1, 0))}
            disabled={idx === 0}
            style={{
              background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.15)',
              borderRadius: 8, padding: '0.4rem 1.1rem', color: 'var(--text-primary)',
              cursor: 'pointer', fontFamily: 'inherit', fontSize: '0.85rem',
              opacity: idx === 0 ? 0.3 : 1,
            }}
          >
            ← Prev
          </button>
          {items.map((item, i) => (
            <button
              key={i}
              onClick={() => setIdx(i)}
              style={{
                width: 8, height: 8, borderRadius: '50%', border: 'none',
                cursor: 'pointer', padding: 0,
                background: i === idx ? 'var(--accent-primary)' : 'rgba(255,255,255,0.2)',
              }}
            />
          ))}
          <button
            onClick={() => setIdx(i => Math.min(i + 1, items.length - 1))}
            disabled={idx === items.length - 1}
            style={{
              background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.15)',
              borderRadius: 8, padding: '0.4rem 1.1rem', color: 'var(--text-primary)',
              cursor: 'pointer', fontFamily: 'inherit', fontSize: '0.85rem',
              opacity: idx === items.length - 1 ? 0.3 : 1,
            }}
          >
            Next →
          </button>
        </div>
      )}
    </div>
  );
}

// ============================================================
// INTERACTIVE TVLA T-STAT CHART (signed, with ±4.5 bands)
// ============================================================
export function TVLALineChart({ tData, pData, color, label, threshold = 4.5, isModal = false }) {
  const svgRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);

  const W = isModal ? 900 : 600;
  const H = isModal ? 240 : 160;
  const PAD = { top: 20, right: isModal ? 20 : 16, bottom: 30, left: isModal ? 50 : 42 };
  const iW = W - PAD.left - PAD.right;
  const iH = H - PAD.top - PAD.bottom;

  if (!tData || tData.length === 0) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: H, color: 'var(--text-muted)', fontSize: '0.78rem' }}>
        No t-stat data available
      </div>
    );
  }

  const n = tData.length;
  const absMax = Math.max(...tData.map(Math.abs), threshold + 0.5);
  const domainMax = absMax + absMax * 0.08;
  const domainMin = -domainMax;

  const xPos = (i) => PAD.left + (i / (n - 1)) * iW;
  const yPos = (v) => PAD.top + iH - ((v - domainMin) / (domainMax - domainMin)) * iH;
  const yZero = yPos(0);
  const yThreshPos = yPos(threshold);
  const yThreshNeg = yPos(-threshold);

  // Build the line path
  const linePath = tData.map((v, i) =>
    `${i === 0 ? 'M' : 'L'}${xPos(i).toFixed(1)},${yPos(v).toFixed(1)}`
  ).join(' ');

  // Exceeding segments for coloring
  const exceedingDots = tData
    .map((v, i) => ({ v, i }))
    .filter(({ v }) => Math.abs(v) > threshold);

  // Grid
  const gridYVals = [-threshold, 0, threshold];
  // Add min/max rounded labels
  const roundedMax = Math.ceil(domainMax);
  if (roundedMax > threshold) gridYVals.push(roundedMax, -roundedMax);

  // p-value secondary info (dot opacity map)
  const pMap = pData ? pData.map(p => p < 0.05) : null;

  // Sample labels on x-axis
  const xTicks = [0, Math.floor(n / 4), Math.floor(n / 2), Math.floor(3 * n / 4), n - 1];

  return (
    <div style={{ position: 'relative', width: '100%' }}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: '100%', height: 'auto', display: 'block' }}
        ref={svgRef}
      >
        <defs>
          <clipPath id={`clip-${label}`}>
            <rect x={PAD.left} y={PAD.top} width={iW} height={iH} />
          </clipPath>
          <linearGradient id={`lineGrad-${label}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.25" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* Threshold band (safe zone background) */}
        <rect
          x={PAD.left} y={yThreshPos}
          width={iW} height={yThreshNeg - yThreshPos}
          fill="rgba(34,197,94,0.05)"
          clipPath={`url(#clip-${label})`}
        />

        {/* Grid lines + labels */}
        {gridYVals.map((v, i) => {
          const y = yPos(v);
          if (y < PAD.top - 2 || y > PAD.top + iH + 2) return null;
          const isThresh = Math.abs(v) === threshold;
          const isZero = v === 0;
          return (
            <g key={i}>
              <line
                x1={PAD.left} y1={y} x2={W - PAD.right} y2={y}
                stroke={isThresh ? 'rgba(239,68,68,0.35)' : isZero ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.05)'}
                strokeWidth={isZero ? 1 : 0.75}
                strokeDasharray={isThresh ? '5 3' : undefined}
              />
              <text
                x={PAD.left - 4} y={y + 3.5}
                textAnchor="end" fontSize={isModal ? 10 : 8}
                fill={isThresh ? 'rgba(239,68,68,0.7)' : 'rgba(255,255,255,0.3)'}
              >
                {v > 0 ? `+${v.toFixed(1)}` : v.toFixed(1)}
              </text>
            </g>
          );
        })}

        {/* Threshold labels on right */}
        <text x={W - PAD.right + 3} y={yThreshPos + 3.5} fontSize="8" fill="rgba(239,68,68,0.7)">+4.5</text>
        <text x={W - PAD.right + 3} y={yThreshNeg + 3.5} fontSize="8" fill="rgba(239,68,68,0.7)">−4.5</text>

        {/* Zero line (thicker) */}
        <line
          x1={PAD.left} y1={yZero} x2={W - PAD.right} y2={yZero}
          stroke="rgba(255,255,255,0.18)" strokeWidth="1"
        />

        {/* Area fill (gradient) */}
        <path
          d={`${linePath} L${xPos(n - 1).toFixed(1)},${yZero} L${xPos(0).toFixed(1)},${yZero} Z`}
          fill={`url(#lineGrad-${label})`}
          clipPath={`url(#clip-${label})`}
        />

        {/* Main line */}
        <path
          d={linePath}
          stroke={color} strokeWidth={isModal ? 1.8 : 1.4}
          fill="none"
          clipPath={`url(#clip-${label})`}
        />

        {/* Exceeding dots (red) */}
        {exceedingDots.map(({ v, i }) => (
          <circle
            key={i}
            cx={xPos(i)} cy={yPos(v)}
            r={isModal ? 4 : 3}
            fill="#ef4444"
            opacity={0.85}
            clipPath={`url(#clip-${label})`}
          />
        ))}

        {/* Hover overlay - invisible rects for interactivity */}
        {tData.map((v, i) => (
          <rect
            key={i}
            x={xPos(i) - iW / n / 2}
            y={PAD.top}
            width={iW / n}
            height={iH}
            fill="transparent"
            style={{ cursor: 'crosshair' }}
            onMouseEnter={(e) => setTooltip({ x: e.clientX, y: e.clientY, idx: i, t: v, p: pData?.[i] })}
            onMouseLeave={() => setTooltip(null)}
            onClick={() => {}} // prevents propagation to modal-close
          />
        ))}

        {/* Tooltip crosshair */}
        {tooltip && (
          <line
            x1={xPos(tooltip.idx)} y1={PAD.top}
            x2={xPos(tooltip.idx)} y2={PAD.top + iH}
            stroke="rgba(255,255,255,0.2)" strokeWidth="1" strokeDasharray="3 2"
          />
        )}

        {/* X-axis ticks */}
        {xTicks.map(i => (
          <text
            key={i}
            x={xPos(i)} y={H - 4}
            textAnchor="middle" fontSize={isModal ? 9 : 7.5}
            fill="rgba(255,255,255,0.25)"
          >
            {i}
          </text>
        ))}

        {/* Axis labels */}
        <text x={PAD.left + iW / 2} y={H - 1} textAnchor="middle" fontSize="8" fill="rgba(255,255,255,0.2)">
          Sample Index
        </text>
        <text
          x={10} y={PAD.top + iH / 2}
          textAnchor="middle" fontSize="8" fill="rgba(255,255,255,0.2)"
          transform={`rotate(-90, 10, ${PAD.top + iH / 2})`}
        >
          t-statistic
        </text>
      </svg>

      {/* Tooltip */}
      {tooltip && (
        <div style={{
          position: 'fixed', left: tooltip.x + 14, top: tooltip.y - 50,
          background: 'var(--bg-active)', border: '1px solid var(--border-hover)',
          borderRadius: 8, padding: '6px 10px', fontSize: '0.73rem',
          pointerEvents: 'none', zIndex: 500, whiteSpace: 'nowrap',
          boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
        }}>
          <div style={{ fontWeight: 600, marginBottom: 2 }}>Sample {tooltip.idx}</div>
          <div>
            t = <span style={{ color: Math.abs(tooltip.t) > 4.5 ? '#ef4444' : color, fontWeight: 700 }}>
              {tooltip.t > 0 ? '+' : ''}{tooltip.t.toFixed(4)}
            </span>
            {Math.abs(tooltip.t) > 4.5 && <span style={{ color: '#ef4444', marginLeft: 6 }}>⚠ exceeds ±4.5</span>}
          </div>
          {tooltip.p != null && (
            <div>
              p = <span style={{ color: tooltip.p < 0.05 ? '#f59e0b' : 'var(--text-secondary)', fontWeight: 600 }}>
                {tooltip.p < 0.001 ? tooltip.p.toExponential(3) : tooltip.p.toFixed(4)}
              </span>
              {tooltip.p < 0.05 && <span style={{ color: '#f59e0b', marginLeft: 6 }}>significant</span>}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
