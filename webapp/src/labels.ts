import type { SectionLabel } from '../shared/types.ts';

/** A distinct, readable color per functional section label. */
export const LABEL_COLOR: Record<SectionLabel, string> = {
  intro: '#4f8cff',
  altintro: '#6aa0ff',
  buildup: '#f5a623',
  drop: '#e8467c',
  breakdown: '#9b59ff',
  bridge: '#1ab6b6',
  cooldown: '#3fb950',
  outro: '#8a93a6',
  altoutro: '#a4adbf',
  start: '#566074',
  end: '#566074',
};

export const labelColor = (label: SectionLabel): string => LABEL_COLOR[label] ?? '#888';
