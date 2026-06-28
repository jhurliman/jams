import { create } from 'zustand';

import type { Annotation, SectionLabel, Segment, TrackMeta } from '../shared/types.ts';
import { api } from './api.ts';

interface ViewState {
  pxPerSec: number;
  scrollLeft: number;
  viewportWidth: number;
}

interface EditorState {
  trackId: string | null;
  meta: TrackMeta | null;
  annotation: Annotation | null;
  /** Read-only model prediction (eval layer). */
  prediction: Annotation | null;
  showEval: boolean;
  loading: boolean;
  /** Bumped on every annotation change; lets the canvas invalidate its cached render. */
  rev: number;
  dirty: boolean;
  saving: boolean;
  selectedSegment: number | null;
  selectedBeat: number | null;
  view: ViewState;
  past: Annotation[];
  future: Annotation[];

  loadTrack: (id: string) => Promise<void>;
  save: () => Promise<void>;
  toggleEval: () => void;
  setView: (patch: Partial<ViewState>) => void;
  zoomAround: (factor: number, anchorClientX: number) => void;
  selectSegment: (i: number | null) => void;
  selectBeat: (i: number | null) => void;
  updateSegment: (i: number, patch: Partial<Segment>) => void;
  relabelSegment: (i: number, label: SectionLabel) => void;
  splitSegmentAt: (time: number) => void;
  deleteSegment: (i: number) => void;
  addBeat: (time: number) => void;
  moveBeat: (i: number, time: number) => void;
  deleteBeat: (i: number) => void;
  cycleBeatBar: (i: number) => void;
  undo: () => void;
  redo: () => void;
}

const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));

export const useEditor = create<EditorState>((set, get) => {
  /** Apply a mutation to the annotation, pushing the prior state onto the undo stack. */
  const commit = (mutator: (ann: Annotation) => void): void => {
    const cur = get().annotation;
    if (!cur) return;
    const next = structuredClone(cur);
    mutator(next);
    set({
      annotation: next,
      dirty: true,
      past: [...get().past, cur].slice(-200),
      future: [],
      rev: get().rev + 1,
    });
  };

  return {
    trackId: null,
    meta: null,
    annotation: null,
    prediction: null,
    showEval: true,
    loading: false,
    rev: 0,
    dirty: false,
    saving: false,
    selectedSegment: null,
    selectedBeat: null,
    view: { pxPerSec: 80, scrollLeft: 0, viewportWidth: 1000 },
    past: [],
    future: [],

    loadTrack: async (id) => {
      set({
        loading: true,
        trackId: id,
        selectedSegment: null,
        selectedBeat: null,
        prediction: null,
      });
      const [meta, annotation, prediction] = await Promise.all([
        api.getTrack(id),
        api.getAnnotation(id),
        api.getPrediction(id),
      ]);
      // Ignore a stale response: if another track was selected while these fetches were in
      // flight, don't clobber its state (which would also make save() write to the wrong track).
      if (get().trackId !== id) return;
      // Load showing the whole track (fit). The fit level is also the zoom-out floor (see
      // zoomAround) so long tracks aren't stuck at a too-high minimum px/s.
      const fit = get().view.viewportWidth / Math.max(meta.durationSec, 1);
      const pxPerSec = Math.min(fit, 600);
      set({
        meta,
        annotation,
        prediction,
        loading: false,
        dirty: false,
        past: [],
        future: [],
        rev: get().rev + 1,
        view: { ...get().view, pxPerSec, scrollLeft: 0 },
      });
    },

    toggleEval: () => set({ showEval: !get().showEval }),

    save: async () => {
      const { trackId, annotation } = get();
      if (!trackId || !annotation) return;
      set({ saving: true });
      try {
        await api.saveAnnotation(trackId, annotation);
        set({ dirty: false, meta: get().meta ? { ...get().meta!, edited: true } : null });
      } finally {
        set({ saving: false });
      }
    },

    setView: (patch) => set({ view: { ...get().view, ...patch } }),

    zoomAround: (factor, anchorClientX) => {
      const { view, meta } = get();
      if (!meta) return;
      // Floor = fit (whole track in view); no point zooming out further. Fixes long tracks where
      // fit is below the old hard floor of 4 px/s (which made zoom-out impossible and zoom-in jump),
      // and short tracks where the floor should be above 4 (so you can't zoom out into blank space).
      const minPx = view.viewportWidth / Math.max(meta.durationSec, 1);
      const anchorTime = (view.scrollLeft + anchorClientX) / view.pxPerSec;
      const pxPerSec = clamp(view.pxPerSec * factor, minPx, 600);
      const maxScroll = Math.max(0, meta.durationSec * pxPerSec - view.viewportWidth);
      const scrollLeft = clamp(anchorTime * pxPerSec - anchorClientX, 0, maxScroll);
      set({ view: { ...view, pxPerSec, scrollLeft } });
    },

    selectSegment: (i) => set({ selectedSegment: i, selectedBeat: null }),
    selectBeat: (i) => set({ selectedBeat: i, selectedSegment: null }),

    updateSegment: (i, patch) =>
      commit((ann) => {
        const seg = ann.segments[i];
        if (!seg) return;
        Object.assign(seg, patch);
        if (seg.end < seg.start) [seg.start, seg.end] = [seg.end, seg.start];
      }),

    relabelSegment: (i, label) =>
      commit((ann) => {
        const seg = ann.segments[i];
        if (seg) seg.label = label;
      }),

    splitSegmentAt: (time) =>
      commit((ann) => {
        const idx = ann.segments.findIndex((s) => time > s.start && time < s.end);
        if (idx < 0) return;
        const seg = ann.segments[idx]!;
        const right: Segment = { start: time, end: seg.end, label: seg.label };
        seg.end = time;
        ann.segments.splice(idx + 1, 0, right);
      }),

    deleteSegment: (i) =>
      commit((ann) => {
        ann.segments.splice(i, 1);
      }),

    addBeat: (time) =>
      commit((ann) => {
        ann.beats.push({ time, bar: 1 });
        ann.beats.sort((a, b) => a.time - b.time);
      }),

    moveBeat: (i, time) =>
      commit((ann) => {
        const beat = ann.beats[i];
        if (beat) beat.time = time;
      }),

    deleteBeat: (i) =>
      commit((ann) => {
        ann.beats.splice(i, 1);
      }),

    cycleBeatBar: (i) =>
      commit((ann) => {
        const beat = ann.beats[i];
        if (beat) beat.bar = (beat.bar % 4) + 1;
      }),

    undo: () => {
      const { past, annotation } = get();
      if (!past.length || !annotation) return;
      const prev = past[past.length - 1]!;
      set({
        annotation: prev,
        past: past.slice(0, -1),
        future: [annotation, ...get().future].slice(0, 200),
        dirty: true,
        rev: get().rev + 1,
      });
    },

    redo: () => {
      const { future, annotation } = get();
      if (!future.length || !annotation) return;
      const next = future[0]!;
      set({
        annotation: next,
        future: future.slice(1),
        past: [...get().past, annotation],
        dirty: true,
        rev: get().rev + 1,
      });
    },
  };
});
