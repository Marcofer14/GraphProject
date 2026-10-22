import { useEffect, useRef, useState } from "react";

const PLAY_SECONDS = 14; // a whole shift replays in this many seconds

/** Replay clock for a shift that ends at `end` seconds. Infinity means "show everything". */
export function usePlayback(end: number, reset: unknown) {
  const [time, setTime] = useState(Infinity);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    setTime(Infinity);
    setPlaying(false);
  }, [reset]);

  const frame = useRef(0);
  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      let done = false;
      setTime((t) => {
        const next = (Number.isFinite(t) ? t : 0) + (dt * end) / PLAY_SECONDS;
        done = next >= end;
        return done ? Infinity : next;
      });
      if (done) setPlaying(false);
      else frame.current = requestAnimationFrame(tick);
    };
    frame.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame.current);
  }, [playing, end]);

  const toggle = () => {
    if (!Number.isFinite(time)) setTime(0);
    setPlaying((p) => !p);
  };
  const seek = (t: number) => {
    setPlaying(false);
    setTime(t >= end ? Infinity : t);
  };
  return { time, playing, toggle, seek };
}
