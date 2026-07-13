/** Persisted import preferences (localStorage). Stems transcription defaults ON — the
 *  point of importing is seeing the analysis — but it's by far the slowest stage, so the
 *  TrackList footer exposes a toggle that sticks across sessions. */
const STEMS_KEY = 'jams.import.stems';

export const importStemsEnabled = (): boolean => localStorage.getItem(STEMS_KEY) !== 'false';

export const setImportStems = (on: boolean): void => {
  localStorage.setItem(STEMS_KEY, String(on));
};
