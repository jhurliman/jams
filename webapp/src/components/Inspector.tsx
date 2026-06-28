import { SECTION_LABELS, type SectionLabel } from '../../shared/types.ts';
import type { AudioControls } from '../hooks/useAudio.ts';
import { labelColor } from '../labels.ts';
import { useEditor } from '../store.ts';

export function Inspector({ audio }: { audio: AudioControls }) {
  const {
    annotation,
    selectedSegment,
    selectedBeat,
    relabelSegment,
    updateSegment,
    deleteSegment,
    splitSegmentAt,
    deleteBeat,
    cycleBeatBar,
  } = useEditor();

  if (!annotation) return <aside className="inspector" />;

  const seg = selectedSegment !== null ? annotation.segments[selectedSegment] : null;
  const beat = selectedBeat !== null ? annotation.beats[selectedBeat] : null;

  return (
    <aside className="inspector">
      <div className="counts">
        {annotation.segments.length} segments · {annotation.beats.length} beats
      </div>

      {seg && selectedSegment !== null && (
        <div className="panel">
          <h3 style={{ color: labelColor(seg.label) }}>Segment {selectedSegment + 1}</h3>
          <label>
            Label
            <select
              value={seg.label}
              onChange={(e) => relabelSegment(selectedSegment, e.target.value as SectionLabel)}
            >
              {SECTION_LABELS.map((l) => (
                <option key={l} value={l}>
                  {l}
                </option>
              ))}
            </select>
          </label>
          <label>
            Start (s)
            <input
              type="number"
              step="0.01"
              value={seg.start.toFixed(3)}
              onChange={(e) => updateSegment(selectedSegment, { start: Number(e.target.value) })}
            />
          </label>
          <label>
            End (s)
            <input
              type="number"
              step="0.01"
              value={seg.end.toFixed(3)}
              onChange={(e) => updateSegment(selectedSegment, { end: Number(e.target.value) })}
            />
          </label>
          <div className="dim">duration {(seg.end - seg.start).toFixed(2)}s</div>
          <div className="row">
            <button className="btn" onClick={() => splitSegmentAt(audio.audioRef.current?.currentTime ?? 0)}>
              Split at playhead
            </button>
            <button className="btn danger" onClick={() => deleteSegment(selectedSegment)}>
              Delete
            </button>
          </div>
        </div>
      )}

      {beat && selectedBeat !== null && (
        <div className="panel">
          <h3>Beat {selectedBeat + 1}</h3>
          <div className="dim">time {beat.time.toFixed(3)}s</div>
          <label>
            Bar position
            <button className="btn" onClick={() => cycleBeatBar(selectedBeat)}>
              {beat.bar} {beat.bar === 1 ? '(downbeat)' : ''} ↻
            </button>
          </label>
          <button className="btn danger" onClick={() => deleteBeat(selectedBeat)}>
            Delete beat
          </button>
        </div>
      )}

      {!seg && !beat && (
        <div className="hint">
          <p>Click a segment band or beat to edit it.</p>
          <ul>
            <li>Drag a segment boundary to move it</li>
            <li>Double-click empty space to add a beat; double-click a beat to delete</li>
            <li>Alt-click a beat to cycle its bar position</li>
            <li>⌘/Ctrl + wheel to zoom · wheel to pan</li>
            <li>
              <strong>E</strong> toggles the read-only model prediction lane (eval) under the ruler
            </li>
          </ul>
          <p className="legend">
            <span style={{ color: '#46cad8' }}>▮</span> ground-truth downbeats ·{' '}
            <span style={{ color: '#f5b84a' }}>▮</span> eval (model) beats in the lane
          </p>
        </div>
      )}
    </aside>
  );
}
