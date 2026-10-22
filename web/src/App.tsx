import { useEffect, useState } from "react";

import { Hero, ResultsTable } from "./components/Hero";
import { Race } from "./components/Race";
import { RouteCompare } from "./components/RouteCompare";
import { loadIndex, loadRoute } from "./data";
import type { DemoIndex, RouteDoc } from "./types";

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

/** The route shown in every section: picking one anywhere switches them all. */
function useRoute(index: DemoIndex | null) {
  const [routeId, setRouteId] = useState<string>();
  const [doc, setDoc] = useState<RouteDoc | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (index && !routeId) setRouteId(index.routes[0]?.id);
  }, [index, routeId]);
  useEffect(() => {
    if (!routeId) return;
    let live = true;
    loadRoute(routeId).then(
      (d) => live && setDoc(d),
      (e: Error) => live && setError(e.message),
    );
    return () => {
      live = false;
    };
  }, [routeId]);
  return { routeId, setRouteId, doc, error };
}

export function App() {
  const [index, setIndex] = useState<DemoIndex | null>(null);
  const [error, setError] = useState<string | null>(null);
  const dark = useDarkMode();
  const route = useRoute(index);

  useEffect(() => {
    loadIndex().then(setIndex, (e: Error) => setError(e.message));
  }, []);

  if (error) return <main className="page"><p className="error">Couldn't load the demo data ({error}).</p></main>;
  if (!index) return <main className="page"><p className="muted">Loading…</p></main>;
  const shared = {
    routes: index.routes,
    routeId: route.routeId,
    onRoute: route.setRouteId,
    doc: route.doc,
    dark,
  };
  return (
    <main className="page">
      <Hero index={index} />
      {route.error && <p className="error">Couldn't load this route ({route.error}).</p>}
      <RouteCompare {...shared} />
      <Race {...shared} />
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
