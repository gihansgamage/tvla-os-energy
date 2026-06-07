export function downloadSvgAsPng(svgElement, filename, title = '', subtitle = '', legends = []) {
  if (!svgElement) return;

  const svgClone = svgElement.cloneNode(true);
  
  let svgString = new XMLSerializer().serializeToString(svgClone);

  const styleMap = {
    'var(--bg-active)': '#21262d',
    'var(--bg-primary)': '#090b0f',
    'var(--bg-secondary)': '#0d1117',
    'var(--bg-tertiary)': '#161b22',
    'var(--text-primary)': '#e6edf3',
    'var(--text-secondary)': '#8b949e',
    'var(--text-muted)': '#484f58',
    'var(--border-color)': 'rgba(255, 255, 255, 0.08)'
  };

  for (const [key, value] of Object.entries(styleMap)) {
    svgString = svgString.replaceAll(key, value);
  }


  const rect = svgElement.getBoundingClientRect();
  const scale = 2; // High resolution
  
  const headerHeight = (title || subtitle) ? 60 * scale : 0;

  // Create a temp canvas to measure text for legends
  const tempCanvas = document.createElement('canvas');
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.font = `${11 * scale}px sans-serif`;
  
  const padding = 15 * scale;
  const dotRadius = 4 * scale;
  const textSpacing = 6 * scale;
  
  let rows = [];
  let footerHeight = 0;
  
  if (legends && legends.length > 0) {
    let currentRow = [];
    let currentWidth = 0;
    const maxWidth = (rect.width * scale) - (40 * scale);
    
    legends.forEach(l => {
      const w = (dotRadius * 2) + textSpacing + tempCtx.measureText(l.label).width;
      if (currentRow.length > 0 && currentWidth + w + padding > maxWidth) {
        rows.push({ items: currentRow, width: currentWidth - padding });
        currentRow = [{ ...l, w }];
        currentWidth = w + padding;
      } else {
        currentRow.push({ ...l, w });
        currentWidth += w + padding;
      }
    });
    if (currentRow.length > 0) {
      rows.push({ items: currentRow, width: currentWidth - padding });
    }
    
    footerHeight = (rows.length * 20 * scale) + (15 * scale);
  }

  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  canvas.width = rect.width * scale;
  canvas.height = (rect.height * scale) + headerHeight + footerHeight;

  const img = new Image();
  img.onload = () => {
    ctx.fillStyle = '#0d1117'; // bg-secondary
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    if (title || subtitle) {
      ctx.fillStyle = '#e6edf3'; // text-primary
      ctx.font = `bold ${14 * scale}px sans-serif`;
      if (title) ctx.fillText(title, 20 * scale, 25 * scale);
      
      if (subtitle) {
        ctx.fillStyle = '#8b949e'; // text-secondary
        ctx.font = `${11 * scale}px sans-serif`;
        ctx.fillText(subtitle, 20 * scale, 45 * scale);
      }
    }

    ctx.drawImage(img, 0, headerHeight, rect.width * scale, rect.height * scale);
    
    if (legends && legends.length > 0) {
      ctx.beginPath();
      ctx.moveTo(20 * scale, headerHeight + rect.height * scale);
      ctx.lineTo(canvas.width - 20 * scale, headerHeight + rect.height * scale);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.lineWidth = 1 * scale;
      ctx.stroke();

      let yPos = headerHeight + rect.height * scale + (15 * scale);
      
      rows.forEach(row => {
        let xPos = (canvas.width - row.width) / 2;
        row.items.forEach(item => {
          ctx.beginPath();
          ctx.arc(xPos + dotRadius, yPos, dotRadius, 0, 2 * Math.PI);
          ctx.fillStyle = item.color || '#fff';
          if (!item.visible) ctx.globalAlpha = 0.3;
          ctx.fill();
          ctx.globalAlpha = 1.0;
          
          ctx.fillStyle = item.visible ? '#e6edf3' : '#8b949e';
          ctx.textBaseline = 'middle';
          ctx.font = `${11 * scale}px sans-serif`;
          ctx.fillText(item.label, xPos + (dotRadius * 2) + textSpacing, yPos);
          
          if (!item.visible) {
            const textWidth = ctx.measureText(item.label).width;
            ctx.beginPath();
            ctx.moveTo(xPos + (dotRadius * 2) + textSpacing, yPos);
            ctx.lineTo(xPos + (dotRadius * 2) + textSpacing + textWidth, yPos);
            ctx.strokeStyle = '#8b949e';
            ctx.lineWidth = 1 * scale;
            ctx.stroke();
          }

          xPos += item.w + padding;
        });
        yPos += 20 * scale;
      });
    }
    
    const pngUrl = canvas.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = pngUrl;
    a.download = filename.endsWith('.png') ? filename : `${filename}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };
  
  img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svgString);
}
