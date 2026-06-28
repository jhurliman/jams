import type { Peaks } from './hooks/usePeaks.ts';

// Bump whenever the analysis math (sample rate, bucket size, bands, peak/RMS) changes so old
// cached entries are ignored.
const CACHE_VERSION = 2;
const DB_NAME = 'jams-annotator';
const STORE = 'peaks';

interface CacheRecord extends Peaks {
  version: number;
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/** Cached band peaks for a track, or null on miss / version mismatch / unavailable IndexedDB. */
export async function loadCachedPeaks(trackId: string): Promise<Peaks | null> {
  try {
    const db = await openDb();
    const rec = await new Promise<CacheRecord | undefined>((resolve, reject) => {
      const tx = db.transaction(STORE, 'readonly');
      const r = tx.objectStore(STORE).get(trackId);
      r.onsuccess = () => resolve(r.result as CacheRecord | undefined);
      r.onerror = () => reject(r.error);
    });
    db.close();
    if (!rec || rec.version !== CACHE_VERSION) return null;
    return { peak: rec.peak, rms: rec.rms, bucketDur: rec.bucketDur, duration: rec.duration };
  } catch {
    return null;
  }
}

/** Store band peaks (best-effort; failures are ignored). Float32Arrays are structured-cloneable. */
export async function storeCachedPeaks(trackId: string, peaks: Peaks): Promise<void> {
  try {
    const db = await openDb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, 'readwrite');
      const rec: CacheRecord = { version: CACHE_VERSION, ...peaks };
      tx.objectStore(STORE).put(rec, trackId);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
    db.close();
  } catch {
    /* cache is best-effort */
  }
}
