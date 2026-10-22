export function hours(h: number): string {
  const total = Math.round(h * 60);
  return `${Math.floor(total / 60)} h ${String(total % 60).padStart(2, "0")} min`;
}

export function seconds(s: number): string {
  if (s === 0) return "—";
  return s < 1 ? `${Math.round(s * 1000)} ms` : `${s.toFixed(1)} s`;
}

export function clock(s: number): string {
  const m = Math.floor(s / 60);
  return `${Math.floor(m / 60)}:${String(m % 60).padStart(2, "0")}`;
}

export const pct = (x: number, digits = 1) => `${x.toFixed(digits)} %`;
