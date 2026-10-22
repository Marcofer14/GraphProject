import { useEffect, useState } from "react";

import { Hero, ResultsTable } from "./components/Hero";
import { RouteCompare } from "./components/RouteCompare";
import { loadIndex } from "./data";
import type { DemoIndex } from "./types";

function useDarkMode(): boolean {
  const query = "(prefers-color-scheme: dark)";
  const [dark, setDark] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const onChange = () => setDark(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return dark;
}

export function App() {
  const [index, setIndex] = useState<DemoIndex | null>(null);
  const [error, setError] = useState<string | null>(null);
  const dark = useDarkMode();

  useEffect(() => {
    loadIndex().then(setIndex, (e: Error) => setError(e.message));
  }, []);

  if (error) return <main className="page"><p className="error">Couldn't load the demo data ({error}).</p></main>;
  if (!index) return <main className="page"><p className="muted">Loading…</p></main>;
  return (
    <main className="page">
      <Hero index={index} />
      <RouteCompare routes={index.routes} dark={dark} />
      <ResultsTable index={index} />
      <footer className="footer">
        <p>
          Built by Marco Fernández ·{" "}
          <a href="https://github.com/Marcofer14/GraphProject">source code and design decisions</a>
        </p>
        <p className="muted">
          Data: Amazon Last Mile Routing Research Challenge (CC BY-NC 4.0). Map tiles © OpenFreeMap, data ©
          OpenStreetMap contributors.
        </p>
      </footer>
    </main>
  );
}
