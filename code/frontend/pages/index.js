import { useState, useCallback } from 'react';
import dynamic from 'next/dynamic';
import Head from 'next/head';
import useSWR from 'swr';
import Layout from '../components/Layout';
import { fetcher } from '../utils/fetcher';

const ERIMap = dynamic(() => import('../components/ERIMap'), { ssr: false });

const NUMERIC_TO_ALPHA3 = {
  '036': 'AUS',  // India
  '124': 'CAN',  // Russia
  '276': 'DEU',  // China
  '840': 'USA',  // Libya
  '826': 'GBR',  // Sudan
  '250': 'FRA',   // Congo, Dem. Rep.
  '800': 'UGA',  // Uganda
  '834': 'TZA',  // Tanzania
  '404': 'KEN',  // Kenya
};

function useLatest() {
  const { data, error, isLoading } = useSWR('/api/eri/latest', fetcher, {
    revalidateOnFocus: false,
  });
  return { data, loading: !error && !data, error };
}

const clr = (s) => {
  if (s == null || Number.isNaN(s)) return 'text-gray-500 font-bold';
  if (s > 0.075) return 'text-red-600 font-bold';
  if (s > 0.05) return 'text-orange-500 font-bold';
  if (s > 0.025) return 'text-yellow-500 font-bold';
  return 'text-green-600 font-bold';
};


export default function Home() {
  const { data: mapData, loading: mapLoad, error: mapErr } = useLatest();

  const [selectedIsoAlpha3, setSelectedIsoAlpha3] = useState(null);
  const [selectedCountryName, setSelectedCountryName] = useState('');
  const [fallbackRow, setFallbackRow] = useState(null);



  const { data: detailPayload, error: dErr, isLoading: dLoad } = useSWR(
    selectedIsoAlpha3 ? `/api/eri/data/${selectedIsoAlpha3}` : null,
    fetcher
  );


  const handleSelect = useCallback((identifier, countryName, mapRow) => {
    console.log('[Home] handleSelect called with:', { identifier, countryName, mapRow });

    let finalAlpha3 = null;

    if (!identifier) {
      console.warn('[Home] Received null or empty identifier.');
    } else {
      const idString = String(identifier).trim().toUpperCase();

      if (/^[A-Z]{3}$/.test(idString)) {
        finalAlpha3 = idString;
        console.log(`[Home] Identifier '${identifier}' recognized as Alpha-3.`);
      }
      else if (/^\d+$/.test(idString)) {
        console.log(`[Home] Identifier '${identifier}' is numeric. Attempting conversion...`);
        const converted = NUMERIC_TO_ALPHA3[idString];
        if (converted) {
          finalAlpha3 = converted;
          console.log(`[Home] Converted numeric '${identifier}' to Alpha-3 '${finalAlpha3}'.`);
        } else {
          console.warn(`[Home] Numeric identifier '${identifier}' not found in NUMERIC_TO_ALPHA3 map.`);
        }
      } else {
        console.warn(`[Home] Identifier '${identifier}' is neither Alpha-3 nor numeric.`);
      }
    }

    // --- Update State based on the Final Alpha-3 Code ---
    if (finalAlpha3) {
      console.log(`[Home] Setting selected country: ISO=${finalAlpha3}, Name='${countryName}'`);
      setSelectedIsoAlpha3(finalAlpha3);
      setSelectedCountryName(countryName);
      setFallbackRow(mapRow);
    } else {
      console.log('[Home] No valid Alpha-3 obtained from identifier. Clearing selection.');
      setSelectedIsoAlpha3(null);
      setSelectedCountryName('');
      setFallbackRow(null);
    }

  }, [NUMERIC_TO_ALPHA3]);


  const displayData = detailPayload?.data ?? fallbackRow;
  const displayName = detailPayload?.name ?? selectedCountryName;


  // --- Sidebar Content Logic ---
  let sidebarContent = null;
  if (!selectedIsoAlpha3) {
    sidebarContent = <p className="italic text-gray-500">Click a country.</p>;
  } else if (dLoad) {
    sidebarContent = <p className="italic text-gray-500">Loading details for {displayName || selectedIsoAlpha3}…</p>;
  } else if (dErr) {
    sidebarContent = (
      <p className="text-red-600 italic">
        {dErr.status === 404
          ? `No detailed explanation data found for ${displayName}. Displaying latest map data:`
          : `Error loading details for ${displayName}. Displaying latest map data:`
        }
      </p>
    );
    if (fallbackRow) {
      sidebarContent = (
        <>
          {sidebarContent}
          {/* Fallback data display */}
          <div className="mt-4 pt-4 border-t border-dashed border-red-300"> {/* top margin/padding */}
            <div className="flex flex-col gap-1"> {/* flexbox layout, gap */}
              <p>
                <strong className="text-gray-800">ISO:</strong> {/*  label bolder */}
                <span className="ml-1 text-gray-700">{selectedIsoAlpha3}</span> {/* spacing, slightly darker text */}
              </p>
              <p>
                <strong className="text-gray-800">Year:</strong> {/*  label bolder */}
                <span className="ml-1 text-gray-700">{fallbackRow.year ?? 'N/A'}</span> {/* spacing, slightly darker text */}
              </p>
              <p>
                <strong className="text-gray-800">Latest ERI (Map):</strong>{' '} {/* label bolder */}
                {/* Score color clr helper */}
                {fallbackRow.eriScore != null
                  ? <span className={clr(fallbackRow.eriScore)}>{fallbackRow.eriScore.toFixed(3)}</span>
                  : <span className="text-gray-700">N/A</span> // Fallback text color
                }
              </p>
            </div>
          </div>
        </>
      );
    }
  } else if (displayData) { // Data loaded successfully 
    const explanationText = (displayData.explanation && typeof displayData.explanation === 'string' && !displayData.explanation.toLowerCase().startsWith('error:'))
      ? displayData.explanation
      : 'No explanation available.';

    sidebarContent = (
      <>
        {/* Main data display */}
        <div className="flex flex-col gap-1">
          <p>
            <strong className="text-gray-800">ISO:</strong> {/* label */}
            <span className="ml-1 text-gray-700">{selectedIsoAlpha3}</span> {/* spacing */}
          </p>
          <p>
            <strong className="text-gray-800">Year:</strong> {/* Make label bolder */}
            <span className="ml-1 text-gray-700">{displayData.year ?? 'N/A'}</span> {/* Add spacing, slightly darker text */}
            {/* Indicate if this is fallback data */}
            {displayData === fallbackRow && <span className="text-gray-500 text-xs ml-1">(Latest Map Data)</span>}
          </p>
          <p>
            <strong className="text-gray-800">ERI:</strong>{' '} {/* Make label bolder */}
            {displayData.eriScore != null && !Number.isNaN(displayData.eriScore)
              ? <span className={clr(displayData.eriScore)}>{displayData.eriScore.toFixed(3)}</span>
              : <span className="text-gray-700">N/A</span>} {/* Fallback text color */}
          </p>
        </div>

        <div className="mt-4 pt-3 border-t">
          <h4 className="font-semibold mb-1 text-gray-800">Explanation</h4> {/*  heading color  */}
          <p className={`text-sm italic break-words ${explanationText.includes('No explanation') ? 'text-gray-800' : 'text-gray-700'}`}>
            {explanationText}
          </p>
        </div>

        {/* SHAP details display */}
        {/* opt chaining defensively */}
        {detailPayload?.data?.top_features_shap && detailPayload.data.top_features_shap.length > 0 && (
          <div className="mt-3 text-xs">
            <h5 className="font-medium mb-1 text-gray-800">Key Factors (Model Analysis):</h5> {/*  heading color  */}
            <ul>
              {detailPayload.data.top_features_shap.slice(0, 5).map(f => (
                // Add fallback key for list items
                <li key={f.feature ?? Math.random()} className={f.shap_value > 0 ? 'text-red-700' : 'text-green-700'}>
                  • {f.feature ?? 'Unknown'} ({f.shap_value != null ? f.shap_value > 0 ? 'Inc' : 'Dec' : ''} Risk)
                </li>
              ))}
            </ul>
          </div>
        )}
        {/* No Shap */}
        {detailPayload?.data && (!detailPayload.data.top_features_shap || detailPayload.data.top_features_shap.length === 0) && (
          <div className="mt-3 text-xs italic text-gray-800">
            No key factor details available for this entry.
          </div>
        )}
      </>
    );
  }
  // --- End of Sidebar Content Logic ---


  return (
    <Layout>
      <Head><title>ERI Dashboard – World Overview</title></Head>

      <div className="flex flex-col lg:flex-row gap-4 p-4">
        {/* map */}
        <div className="w-full lg:w-2/3">
          <h2 className="text-2xl font-semibold mb-2 text-gray-800">World Economic Risk Overview</h2>
          <p className="text-gray-800 mb-4">Hover or click a country to view its ERI details.</p>

          {mapErr && <p className="p-3 bg-red-100 text-red-700 rounded mb-4">Error loading map.</p>}

          <div className="border rounded-lg shadow bg-white min-h-[400px] flex items-center justify-center">
            {mapLoad || !mapData
              ? <div className="h-96 flex items-center justify-center">
                <p className="text-gray-800">Loading map data…</p>
              </div>
              : <ERIMap eriData={mapData} onCountrySelect={handleSelect} />
            }
          </div>
        </div>

        {/* sidebar */}
        <div className="w-full lg:w-1/3 bg-white p-4 rounded-lg shadow-md border">
          <h3 className="text-xl font-semibold mb-3 border-b pb-2 text-gray-800">
            {displayName ? `Details for ${displayName}` : 'Country Details'}
          </h3>
          <div className="text-sm text-gray-800">
            {sidebarContent}
          </div>
        </div>
      </div>
    </Layout>
  );
}