import { useEffect, useState } from "react";

import { Hero, ResultsTable } from "./components/Hero";
import { Architecture } from "./components/Architecture";
import { HowToUse } from "./components/HowToUse";
import { Limits } from "./components/Limits";
import { Nav } from "./components/Nav";
import { UseCases } from "./components/UseCases";
import { Walkthrough } from "./components/Walkthrough";
import { Race } from "./components/Race";
import { RouteCompare } from "./components/RouteCompare";
import { StationDay } from "./components/StationDay";
import { loadDay, loadIndex, loadRoute } from "./data";
import type { DemoIndex, Listing } from "./types";

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

/** The item picked from a listing (first by default) and its loaded document. */
function useDoc<T>(items: Listing[] | undefined, load: (id: string) => Promise<T>) {
  const [id, setId] = useState<string>();
  const [doc, setDoc] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (items && !id) setId(items[0]?.id);
  }, [items, id]);
  useEffect(() => {
    if (!id) return;
    let live = true;
    load(id).then(
      (d) => live && setDoc(d),
      (e: Error) => live && setError(e.message),
    );
    return () => {
      live = false;
    };
  }, [id, load]);
  return { id, setId, doc, error };
}

export function App() {
  const [index, setIndex] = useState<DemoIndex | null>(null);
  const [error, setError] = useState<string | null>(null);
  const dark = useDarkMode();
  const route = useDoc(index?.routes, loadRoute); // shared by the compare and race sections
  const day = useDoc(index?.days, loadDay);
  const [budget, setBudget] = useState(1); // the race's computing time, also set by the guided example

  useEffect(() => {
    loadIndex().then(setIndex, (e: Error) => setError(e.message));
  }, []);

  if (error) return <main className="page"><p className="error">Couldn't load the demo data ({error}).</p></main>;
  if (!index) return <main className="page"><p className="muted">Loading…</p></main>;
  const shared = {
    routes: index.routes,
    routeId: route.id,
    onRoute: route.setId,
    doc: route.doc,
    dark,
  };
  return (
    <>
    <Nav />
    <main className="page">
      <Hero index={index} />
      <HowToUse />
      <Walkthrough index={index} onRoute={route.setId} onBudget={setBudget} onDay={day.setId} />
      {route.error && <p className="error">Couldn't load this route ({route.error}).</p>}
      <RouteCompare {...shared} />
      <Race {...shared} budget={budget} onBudget={setBudget} />
      {day.error && <p className="error">Couldn't load this day ({day.error}).</p>}
      <StationDay days={index.days} dayId={day.id} onDay={day.setId} doc={day.doc} dark={dark} />
      <UseCases />
      <Architecture index={index} />
      <ResultsTable index={index} />
      <Limits index={index} />
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
    </>
  );
}
