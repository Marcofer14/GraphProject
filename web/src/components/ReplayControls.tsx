import { clock } from "../format";

interface Props {
  end: number;
  time: number;
  playing: boolean;
  disabled: boolean;
  onToggle: () => void;
  onSeek: (t: number) => void;
}

/** Replay button, time slider and clock, shared by the route and station-day sections. */
export function ReplayControls({ end, time, playing, disabled, onToggle, onSeek }: Props) {
  return (
    <>
      <button type="button" onClick={onToggle} disabled={disabled}>
        {playing ? "Pause" : Number.isFinite(time) ? "Resume" : "Replay the shift"}
      </button>
      <input
        type="range"
        aria-label="Time into the shift"
        min={0}
        max={Math.ceil(end)}
        step={60}
        value={Number.isFinite(time) ? Math.round(time) : Math.ceil(end)}
        onChange={(e) => onSeek(Number(e.target.value))}
      />
      <span className="clock" aria-live="off">
        {clock(Number.isFinite(time) ? time : end)} into the shift
      </span>
    </>
  );
}
