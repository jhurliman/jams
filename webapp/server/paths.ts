import { execSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));

/**
 * Root of the Raveform data tree, resolved from the first of:
 *   1. RAVEFORM_DIR env override
 *   2. this repo's eval/data/raveform
 *   3. the *main* checkout's eval/data/raveform — eval/data is gitignored, so
 *      when running from a git worktree the data only exists in the main repo
 */
function resolveDataDir(): string {
  if (process.env.RAVEFORM_DIR) return resolve(process.env.RAVEFORM_DIR);

  const local = resolve(here, '../../eval/data/raveform');
  if (existsSync(resolve(local, 'segments.json'))) return local;

  try {
    // In a worktree, --git-common-dir points at <main-checkout>/.git.
    const commonDir = execSync('git rev-parse --git-common-dir', {
      cwd: here,
      encoding: 'utf8',
    }).trim();
    const mainRoot = dirname(resolve(here, commonDir));
    const main = resolve(mainRoot, 'eval/data/raveform');
    if (existsSync(resolve(main, 'segments.json'))) {
      console.log(`[annotator] worktree detected — using main checkout data at ${main}`);
      return main;
    }
  } catch {
    // not a git checkout; fall through to the local default and its error below
  }
  return local;
}

export const DATA_DIR = resolveDataDir();

if (!existsSync(resolve(DATA_DIR, 'segments.json'))) {
  console.error(
    `[annotator] no Raveform data at ${DATA_DIR} (segments.json missing).\n` +
      `            Set RAVEFORM_DIR=/path/to/eval/data/raveform or run the acquire scripts first.`,
  );
}

export const SEGMENTS_JSON = resolve(DATA_DIR, 'segments.json');
export const IMPORTS_JSON = resolve(DATA_DIR, 'imports.json');
export const BEATS_DIR = resolve(DATA_DIR, 'raveform/structures/beats');
export const AUDIO_DIR = resolve(DATA_DIR, 'audio');
export const ANNOTATIONS_DIR = resolve(DATA_DIR, 'annotations');
export const PREDICTIONS_DIR = resolve(DATA_DIR, 'predictions');
export const STEMS_DIR = resolve(DATA_DIR, 'stems');

/** Dataset audio is .m4a; imported tracks keep their uploaded extension. */
export const AUDIO_EXTS = ['.m4a', '.mp3', '.wav', '.flac', '.ogg', '.aac', '.aiff'] as const;

export const audioPath = (id: string): string => {
  for (const ext of AUDIO_EXTS) {
    const p = resolve(AUDIO_DIR, `${id}${ext}`);
    if (existsSync(p)) return p;
  }
  return resolve(AUDIO_DIR, `${id}.m4a`);
};
export const beatCsvPath = (id: string): string => resolve(BEATS_DIR, `${id}.beat.csv`);
export const editedPath = (id: string): string => resolve(ANNOTATIONS_DIR, `${id}.json`);
export const predictionPath = (id: string): string => resolve(PREDICTIONS_DIR, `${id}.json`);
export const stemsPath = (id: string): string => resolve(STEMS_DIR, `${id}.json`);

export const hasAudio = (id: string): boolean => existsSync(audioPath(id));
export const hasPrediction = (id: string): boolean => existsSync(predictionPath(id));
export const hasStems = (id: string): boolean => existsSync(stemsPath(id));
