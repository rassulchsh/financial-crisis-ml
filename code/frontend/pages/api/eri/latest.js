// frontend/pages/api/eri/latest.js
import { MongoClient } from 'mongodb';

const uri = process.env.MONGO_URI;
const dbName = process.env.MONGO_DB_NAME || 'financial_crisis';
const collectionName = 'eri_explanations_nosha';

// ────────────────────────────────────────────────────────────────
// Cached connection (AWS-Lambda / Vercel cold-start friendly)
let cachedClient = null;
let cachedDb = null;
async function connect() {
  if (cachedClient && cachedDb) return { client: cachedClient, db: cachedDb };

  if (!uri) throw new Error('MONGO_URI missing in env');
  const client = new MongoClient(uri);
  await client.connect();

  cachedClient = client;
  cachedDb = client.db(dbName);
  console.log('[api/eri/latest] Connected to MongoDB');

  return { client, db: cachedDb };
}
// ────────────────────────────────────────────────────────────────

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    res.setHeader('Allow', ['GET']);
    return res.status(405).end(`Method ${req.method} Not Allowed`);
  }

  try {
    const { db } = await connect();
    const col = db.collection(collectionName);

    const latest = await col.aggregate([
      // (your existing pipeline up through the final $project)
    ]).toArray();

    // ── Build a lookup object AND normalize the ISO codes ─────────
    const eriDataMap = latest.reduce((obj, row) => {
      // ensure there is no trailing/leading whitespace, always uppercase
      const iso3 = String(row.iso ?? '')
        .trim()
        .toUpperCase();
      obj[iso3] = {
        // copy everything, but overwrite `iso` with our clean string
        ...row,
        iso: iso3,
      };
      return obj;
    }, {});

    console.log(
      `[api/eri/latest] Sent latest ERI for ${Object.keys(eriDataMap).length} countries`
    );
    return res.status(200).json(eriDataMap);
  } catch (err) {
    console.error('[api/eri/latest] Fatal', err);
    return res.status(500).json({ message: 'Internal Server Error' });
  }
}
