import { useMemo } from "react";

import type { Listing } from "../types";

interface Props {
  routes: Listing[];
  value: string | undefined;
  onChange: (id: string) => void;
}

export function RoutePicker({ routes, value, onChange }: Props) {
  const byCity = useMemo(() => {
    const groups = new Map<string, Listing[]>();
    routes.forEach((r) => groups.set(r.city, [...(groups.get(r.city) ?? []), r]));
    return [...groups.entries()];
  }, [routes]);
  return (
    <label>
      <span className="sr-only">Route</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {byCity.map(([city, rs]) => (
          <optgroup key={city} label={city}>
            {rs.map((r, i) => (
              <option key={r.id} value={r.id}>
                {city} {i + 1} · {r.stops} stops
              </option>
            ))}
          </optgroup>
        ))}
      </select>
    </label>
  );
}
