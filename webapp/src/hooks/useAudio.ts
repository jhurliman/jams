import { useEffect, useRef, useState } from 'react';

export interface AudioControls {
  audioRef: React.RefObject<HTMLAudioElement | null>;
  isPlaying: boolean;
  toggle: () => void;
  play: () => void;
  pause: () => void;
  seek: (time: number) => void;
}

/** Wraps an HTMLAudioElement for a track URL. The playhead is read imperatively from
 *  `audioRef.current.currentTime` (so the canvas can rAF-draw it smoothly). */
export function useAudio(url: string | null): AudioControls {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    if (!url) return;
    const audio = new Audio(url);
    audio.preload = 'auto';
    audioRef.current = audio;
    const onPlay = () => setIsPlaying(true);
    const onPause = () => setIsPlaying(false);
    audio.addEventListener('play', onPlay);
    audio.addEventListener('pause', onPause);
    audio.addEventListener('ended', onPause);
    return () => {
      audio.pause();
      audio.removeEventListener('play', onPlay);
      audio.removeEventListener('pause', onPause);
      audio.removeEventListener('ended', onPause);
      audioRef.current = null;
    };
  }, [url]);

  const play = () => void audioRef.current?.play();
  const pause = () => audioRef.current?.pause();
  const toggle = () => (audioRef.current?.paused ? play() : pause());
  const seek = (time: number) => {
    if (audioRef.current) audioRef.current.currentTime = Math.max(0, time);
  };

  return { audioRef, isPlaying, toggle, play, pause, seek };
}
