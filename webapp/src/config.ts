/** Client-side tuning knobs. */

/** Structure analysis runs at roughly this fraction of a track's realtime duration on
 *  an Apple-Silicon MBP with warm caches (measured ~0.14x: 60 s track -> 8.6 s demix +
 *  All-In-One on MPS; padded to 0.2 so the bar finishes early rather than stalling).
 *  Used only to animate the progress bar within the structure stage — stage
 *  transitions themselves come from the server and are always real. */
export const STRUCTURE_ANALYSIS_REALTIME_FACTOR = 0.2;
