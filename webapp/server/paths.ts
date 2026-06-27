import { existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));

/** Root of the Raveform data tree. Override with RAVEFORM_DIR; defaults to the repo's eval data. */
export const DATA_DIR = process.env.RAVEFORM_DIR
  ? resolve(process.env.RAVEFORM_DIR)
  : resolve(here, '../../eval/data/raveform');

export const SEGMENTS_JSON = resolve(DATA_DIR, 'segments.json');
export const BEATS_DIR = resolve(DATA_DIR, 'raveform/structures/beats');
export const AUDIO_DIR = resolve(DATA_DIR, 'audio');
export const ANNOTATIONS_DIR = resolve(DATA_DIR, 'annotations');

export const audioPath = (id: string): string => resolve(AUDIO_DIR, `${id}.m4a`);
export const beatCsvPath = (id: string): string => resolve(BEATS_DIR, `${id}.beat.csv`);
export const editedPath = (id: string): string => resolve(ANNOTATIONS_DIR, `${id}.json`);

export const hasAudio = (id: string): boolean => existsSync(audioPath(id));
