import { api } from '../api.ts';
import { useEditor } from '../store.ts';
import { PianoRoll } from './PianoRoll.tsx';

/** Read-only stack of per-stem piano-rolls under the main waveform, with MIDI download links.
 *  Renders nothing unless a stems result is present for the current track. */
export function StemLanes() {
  const trackId = useEditor((s) => s.trackId);
  const stems = useEditor((s) => s.stems);
  if (!trackId || !stems || stems.transcriptions.length === 0) return null;

  const hasCombined = stems.midiPaths.combined !== undefined;

  return (
    <div className="stemlanes">
      <div className="stemlanes-head">
        <span className="dim">Stems (read-only)</span>
        {hasCombined && (
          <a className="midi-dl" href={api.midiUrl(trackId, 'combined')} download>
            combined.mid
          </a>
        )}
      </div>
      {stems.transcriptions.map((t) => {
        const hasMidi = stems.midiPaths[t.stemType] !== undefined;
        return (
          <div className="stemlane" key={t.stemType}>
            <div className="stemlane-label">
              <span className="stem-name">{t.stemType}</span>
              <span className="dim stem-count">{t.notes.length} notes</span>
              {hasMidi && (
                <a className="midi-dl" href={api.midiUrl(trackId, t.stemType)} download>
                  .mid
                </a>
              )}
            </div>
            <PianoRoll transcription={t} />
          </div>
        );
      })}
    </div>
  );
}
