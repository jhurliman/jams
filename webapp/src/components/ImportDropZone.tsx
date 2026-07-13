import { useCallback, useEffect, useRef, useState } from 'react';

import { api } from '../api.ts';
import { useEditor } from '../store.ts';

type Phase =
  | { kind: 'idle' }
  | { kind: 'hover' }
  | { kind: 'analyzing'; name: string }
  | { kind: 'error'; message: string };

const AUDIO_EXT = /\.(wav|mp3|flac|aiff|ogg|m4a|aac)$/i;

/** Full-window drag-and-drop import: drop an audio file anywhere to run it through the
 *  jams analysis backend and open it as a new track. Renders nothing while idle. */
export function ImportDropZone() {
  const [phase, setPhase] = useState<Phase>({ kind: 'idle' });
  // dragenter/dragleave fire for every child element; track depth to know when the
  // pointer actually left the window.
  const depth = useRef(0);

  const doImport = useCallback(async (file: File) => {
    if (!AUDIO_EXT.test(file.name)) {
      setPhase({ kind: 'error', message: `'${file.name}' is not a supported audio file` });
      return;
    }
    setPhase({ kind: 'analyzing', name: file.name });
    try {
      const { id } = await api.importTrack(file);
      const ed = useEditor.getState();
      ed.refreshTracks();
      await ed.loadTrack(id);
      setPhase({ kind: 'idle' });
    } catch (err) {
      setPhase({ kind: 'error', message: err instanceof Error ? err.message : String(err) });
    }
  }, []);

  useEffect(() => {
    const hasFiles = (e: DragEvent) => [...(e.dataTransfer?.types ?? [])].includes('Files');
    const onDragEnter = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth.current += 1;
      setPhase((p) => (p.kind === 'idle' || p.kind === 'error' ? { kind: 'hover' } : p));
    };
    const onDragOver = (e: DragEvent) => {
      if (hasFiles(e)) e.preventDefault();
    };
    const onDragLeave = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth.current = Math.max(0, depth.current - 1);
      if (depth.current === 0) setPhase((p) => (p.kind === 'hover' ? { kind: 'idle' } : p));
    };
    const onDrop = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth.current = 0;
      const file = e.dataTransfer?.files[0];
      if (file) void doImport(file);
      else setPhase({ kind: 'idle' });
    };
    window.addEventListener('dragenter', onDragEnter);
    window.addEventListener('dragover', onDragOver);
    window.addEventListener('dragleave', onDragLeave);
    window.addEventListener('drop', onDrop);
    return () => {
      window.removeEventListener('dragenter', onDragEnter);
      window.removeEventListener('dragover', onDragOver);
      window.removeEventListener('dragleave', onDragLeave);
      window.removeEventListener('drop', onDrop);
    };
  }, [doImport]);

  if (phase.kind === 'idle') return null;
  return (
    <div className={`import-overlay ${phase.kind}`}>
      {phase.kind === 'hover' && <div className="import-card">Drop to analyze &amp; import</div>}
      {phase.kind === 'analyzing' && (
        <div className="import-card">
          <div className="spinner" />
          <div>
            Analyzing <strong>{phase.name}</strong>…
          </div>
          <div className="dim">key · tempo · beats · structure — this can take a minute</div>
        </div>
      )}
      {phase.kind === 'error' && (
        <div className="import-card error" onClick={() => setPhase({ kind: 'idle' })}>
          <div>Import failed</div>
          <div className="dim">{phase.message}</div>
          <div className="dim">(click to dismiss)</div>
        </div>
      )}
    </div>
  );
}
