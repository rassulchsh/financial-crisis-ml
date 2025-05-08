import { MongoClient } from 'mongodb';

const uri = process.env.MONGO_URI;
const dbName = process.env.MONGO_DB_NAME || 'financial_crisis';
const collectionName = process.env.MONGO_ERI_COLLECTION || 'eri_explanations';

const NUM_TO_ALPHA_API = {
  '036': 'AUS', '124': 'CAN', '156': 'CHN', '170': 'COL', '250': 'FRA',
  '276': 'DEU', '356': 'IND', '364': 'IRN', '398': 'KAZ', '404': 'KEN',
  '643': 'RUS', '710': 'ZAF', '729': 'SDN', '800': 'UGA', '826': 'GBR',
  '840': 'USA', '180': 'COD',
};


/* ── cached connection helper ────────────────────────────────────────── */
let cachedClient, cachedDb;
async function connect() {
  if (cachedClient && cachedDb) {
    console.log('[api/eri/data] Using cached MongoDB connection.');
    return { client: cachedClient, db: cachedDb };
  }

  if (!uri) {
    console.error('[api/eri/data] MONGO_URI missing in env');
    throw new Error('MONGO_URI missing in env');
  }

  try {
    console.log('[api/eri/data] Attempting new MongoDB Connection...');
    const client = new MongoClient(uri);
    await client.connect();
    console.log('[api/eri/data] MongoDB Connected.');

    cachedClient = client;
    cachedDb = client.db(dbName);
    return { client, db: cachedDb };
  } catch (err) {
    console.error('[api/eri/data] MongoDB Connection Failed:', err);
    throw err;
  }
}

/* ── API handler ─────────────────────────────────────────────────────── */
export default async function handler(req, res) {
  if (req.method !== 'GET') {
    res.setHeader('Allow', ['GET']);
    return res.status(405).end(`Method ${req.method} Not Allowed`);
  }

  const raw = req.query.isoCode;
  if (!raw || typeof raw !== 'string') {
    console.warn('[api/eri/data] Received invalid isoCode parameter:', raw);
    return res.status(400).json({ message: 'Invalid or missing isoCode parameter' });
  }

  let isoQuery = raw.trim().toUpperCase();
  console.log(`[api/eri/data] Received raw isoCode: ${raw}, normalized: ${isoQuery}`);

  if (/^\d+$/.test(isoQuery)) {
    const numericIdPadded = isoQuery.padStart(3, '0');
    console.log(`[api/eri/data] Input is numeric: ${numericIdPadded}. Attempting conversion...`);
    const convertedAlpha3 = NUM_TO_ALPHA_API[numericIdPadded];

    if (convertedAlpha3) {
      isoQuery = convertedAlpha3;
      console.log(`[api/eri/data] Converted numeric ID to Alpha-3: ${isoQuery}`);
    } else {
      console.warn(`[api/eri/data] Numeric ID '${numericIdPadded}' not found in API conversion map.`);
      return res.status(404).json({ message: `No data found for numeric country code ${raw}` });
    }
  } else if (!/^[A-Z]{3}$/.test(isoQuery)) {
    console.warn(`[api/eri/data] Input is not numeric and not valid Alpha-3: ${isoQuery}`);
    return res.status(400).json({ message: `Invalid country code format: ${raw}` });
  }


  try {
    const { db } = await connect();
    const col = db.collection(collectionName);
    console.log(`[api/eri/data] Querying collection '${collectionName}' for ISO: ${isoQuery}`);


    const [row] = await col.aggregate([
      {
        $match: {
          iso: isoQuery
        }
      },
      {
        $project: {
          isoNorm: { $toUpper: { $trim: { input: '$iso' } } },
          year: 1, explanation: 1, top_features_shap: 1,
          eriScoreRaw: '$eriScore'
        }
      },
      { $match: { isoNorm: isoQuery } },

      {
        $project: {
          isoNorm: 1, year: 1, explanation: 1, top_features_shap: 1,
          eriScore: {
            $cond: [
              { $eq: [{ $type: '$eriScoreRaw' }, 'decimal'] },
              { $toDouble: '$eriScoreRaw' },
              '$eriScoreRaw'
            ]
          }
        }
      },
      { $sort: { year: -1 } },
      { $limit: 1 },
      {
        $project: {
          _id: 0,
          iso: '$isoNorm',
          year: 1,
          eriScore: 1, explanation: 1, top_features_shap: 1
        }
      }
    ]).toArray();

    console.log('[api/eri/data] Aggregation result row:', row);


    if (!row) {
      console.log(`[api/eri/data] No data found for ISO: ${isoQuery} after aggregation.`);
      return res.status(404).json({ message: `No data found for country code ${isoQuery}` });
    }

    if (row.eriScore && typeof row.eriScore !== 'number') {
      try {
        row.eriScore = parseFloat(row.eriScore.toString());
        if (Number.isNaN(row.eriScore)) row.eriScore = null;
      } catch (e) {
        console.warn(`[api/eri/data] Failed to convert eriScore to float for ${isoQuery}:`, e);
        row.eriScore = null;
      }
    }


    console.log(`[api/eri/data] Sending data for ${isoQuery}, year ${row.year}.`);

    return res.status(200).json({
      name: row.iso,
      iso: row.iso,
      data: row
    });

  } catch (err) {
    console.error('[api/eri/data] Fatal error during API processing:', err);
    return res.status(500).json({ message: 'Internal Server Error while fetching country data' });
  }
}