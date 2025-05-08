import React, { useState } from 'react';
import { ComposableMap, Geographies, Geography } from 'react-simple-maps';
import { Tooltip as ReactTooltip } from 'react-tooltip';
import 'react-tooltip/dist/react-tooltip.css';

const geoUrl = 'https://unpkg.com/world-atlas@2.0.2/countries-110m.json';

function getBestIdentifier(geo) {
  if (!geo || !geo.properties) return null;

  const props = geo.properties;

  let id = props.ISO_A3;
  if (typeof id === 'string' && /^[A-Za-z]{3}$/.test(id.trim())) {
    return id.trim().toUpperCase();
  }
  id = props.iso_a3;
  if (typeof id === 'string' && /^[A-Za-z]{3}$/.test(id.trim())) {
    return id.trim().toUpperCase();
  }
  id = props.ADM0_A3;
  if (typeof id === 'string' && /^[A-Za-z]{3}$/.test(id.trim())) {
    return id.trim().toUpperCase();
  }


  id = String(geo.id ?? '');
  if (typeof id === 'string' && id.length > 0 && id !== 'null' && id !== 'undefined') {
    if (/^\d+$/.test(id) || /^[A-Za-z]{3}$/.test(id.trim())) {
      return /^\d+$/.test(id) ? id : id.trim().toUpperCase();
    }
  }

  id = String(props.iso_n3 ?? props.ISO_N3 ?? props.id ?? '');
  if (typeof id === 'string' && /^\d+$/.test(id)) {
    return id;
  }

  return null;
}


function colour(s) {
  if (s == null || Number.isNaN(s)) return '#E9EAEA';
  if (s > 0.065) return '#E53E3E';
  if (s > 0.05) return '#ED8936';
  if (s > 0.025) return '#ECC94B';
  return '#48BB78';
}

export default function ERIMap({ eriData = {}, onCountrySelect }) {
  const [tip, setTip] = useState('');

  return (
    <>
      <ComposableMap
        projectionConfig={{ scale: 140 }}
        style={{ width: '100%', height: 'auto' }}
        data-tooltip-id="tip"
      >
        <Geographies geography={geoUrl}>
          {({ geographies }) =>
            geographies.map((geo) => {
              const identifier = getBestIdentifier(geo);
              const row = identifier ? eriData[identifier] : undefined;
              const fill = colour(row?.eriScore);

              const isPotentiallyClickable = !!identifier;


              return (
                <Geography
                  key={geo.rsmKey}
                  geography={geo}
                  style={{
                    default: { fill, stroke: '#666', strokeWidth: 0.3, outline: 'none' },
                    hover: {
                      fill: '#a0d8f8', stroke: '#333', strokeWidth: 0.6,
                      cursor: isPotentiallyClickable ? 'pointer' : 'default'
                    },
                    pressed: { fill: '#74c0e8', stroke: '#333', strokeWidth: 0.6 },
                  }}
                  onMouseEnter={() => {
                    const txt = row?.eriScore != null ? row.eriScore.toFixed(3) : 'N/A';
                    setTip(`${geo.properties.name} (${identifier || 'N/A'}) – Latest ERI: ${txt}`);
                  }}
                  onMouseLeave={() => setTip('')}
                  onClick={() => {
                    if (isPotentiallyClickable && onCountrySelect) {
                      console.log('[ERIMap] click -> Passing identifier, name, data:', { identifier, name: geo.properties.name, row });
                      // Pass the identifier string, name, and the map data row
                      onCountrySelect(identifier, geo.properties.name, row ?? null);
                    } else if (!isPotentiallyClickable) {
                      console.log(`[ERIMap] Click ignored for ${geo.properties.name} (No usable identifier found)`);
                    }
                  }}
                />
              );
            })
          }
        </Geographies>
      </ComposableMap>
      <ReactTooltip id="tip" content={tip} float />
    </>
  );
}