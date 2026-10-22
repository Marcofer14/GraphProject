import { useEffect, useState } from "react";

import { Architecture } from "./components/Architecture";
import { Hero } from "./components/Hero";
import { Limits } from "./components/Limits";
import { Nav } from "./components/Nav";
import { UseCases } from "./components/UseCases";
import { Why } from "./components/Why";
import { Workspace } from "./components/Workspace";
import { loadDay, loadIndex, loadRoute } from "./data";
import type { DemoIndex, Listing } from "./types";

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
  const route = useDoc(index?.routes, loadRoute);
  const day = useDoc(index?.days, loadDay);

  useEffect(() => {
    loadIndex().then(setIndex, (e: Error) => setError(e.message));
  }, []);

  if (error) return <main className="page"><p className="error">Couldn't load the demo data ({error}).</p></main>;
  if (!index) return <main className="page"><p className="muted">Loading…</p></main>;
  return (
    <>
      <Nav />
      <main className="page">
        <Hero index={index} />
        {(route.error || day.error) && <p className="error">Couldn't load part of the data ({route.error ?? day.error}).</p>}
        <Workspace
          index={index}
          routeId={route.id}
          onRoute={route.setId}
          route={route.doc}
          dayId={day.id}
          onDay={day.setId}
          day={day.doc}
        />
        <Why index={index} />
        <UseCases />
        <Architecture index={index} />
        <Limits index={index} />
        <footer className="footer">
          <p>
            Built by <a href="https://marcofer14.github.io/">Marco Fernández</a> ·{" "}
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
