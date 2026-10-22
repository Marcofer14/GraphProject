import type { Feature, FeatureCollection } from "geojson";
import {
  type GeoJSONSource,
  LngLatBounds,
  type Map as MLMap,
  Map as MapLibre,
  NavigationControl,
  setWorkerUrl,
} from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?url";
import { useEffect, useMemo, useRef, useState } from "react";

import { arrivals, drivenBy, toLngLat, tracks } from "../playback";
import type { LatLng, SolutionData } from "../types";

// MapLibre looks for its worker next to its own module, which the bundler moves
setWorkerUrl(workerUrl);

const STYLES = {
  light: "https://tiles.openfreemap.org/styles/positron",
  dark: "https://tiles.openfreemap.org/styles/dark",
};
const UNVISITED = "#B4B2A9";
const EMPTY: FeatureCollection = { type: "FeatureCollection", features: [] };

interface Props {
  depot: LatLng;
  stops: LatLng[];
  solution: SolutionData;
  color: string;
  time: number; // seconds after departure; past the end shows the whole route
  dark: boolean;
  onMap?: (map: MLMap | null) => void;
}

function addLayers(map: MLMap) {
  for (const id of ["driven", "stops", "van", "depot"]) map.addSource(id, { type: "geojson", data: EMPTY });
  map.addLayer({
    id: "driven",
    type: "line",
    source: "driven",
    layout: { "line-join": "round", "line-cap": "round" },
    paint: { "line-width": 2.5, "line-opacity": 0.9 },
  });
  map.addLayer({
    id: "stops",
    type: "circle",
    source: "stops",
    paint: { "circle-radius": 3.5, "circle-stroke-width": 1, "circle-stroke-color": "#ffffff" },
  });
  map.addLayer({
    id: "depot",
    type: "circle",
    source: "depot",
    paint: { "circle-radius": 7, "circle-color": "#2C2C2A", "circle-stroke-width": 2, "circle-stroke-color": "#ffffff" },
  });
  map.addLayer({
    id: "van",
    type: "circle",
    source: "van",
    paint: { "circle-radius": 6, "circle-stroke-width": 2, "circle-stroke-color": "#ffffff" },
  });
}

const point = (coordinates: [number, number], properties = {}): Feature => ({
  type: "Feature",
  geometry: { type: "Point", coordinates },
  properties,
});

export function RouteMap({ depot, stops, solution, color, time, dark, onMap }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MLMap | null>(null);
  const [styleVersion, setStyleVersion] = useState(0); // bumps when our layers are (re)added
  const onMapRef = useRef(onMap);
  onMapRef.current = onMap;

  useEffect(() => {
    const map = new MapLibre({
      container: container.current!,
      style: STYLES[dark ? "dark" : "light"],
      attributionControl: { compact: true },
      cooperativeGestures: true, // the page scrolls past the map; zoom needs ctrl / two fingers
    });
    map.addControl(new NavigationControl({ showCompass: false }), "top-right");
    map.on("style.load", () => {
      addLayers(map);
      setStyleVersion((v) => v + 1);
    });
    mapRef.current = map;
    onMapRef.current?.(map);
    return () => {
      onMapRef.current?.(null);
      map.remove();
      mapRef.current = null;
    };
  }, []); // the map is created once; theme changes go through setStyle below

  const firstStyle = useRef(true);
  useEffect(() => {
    if (firstStyle.current) {
      firstStyle.current = false;
      return;
    }
    mapRef.current?.setStyle(STYLES[dark ? "dark" : "light"]);
  }, [dark]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || stops.length === 0) return;
    const bounds = new LngLatBounds();
    stops.forEach((s) => bounds.extend(toLngLat(s)));
    map.fitBounds(bounds, { padding: 32, duration: 0 }); // the station may be off-screen
  }, [stops]);

  const vans = useMemo(() => tracks(depot, stops, solution), [depot, stops, solution]);
  const arrive = useMemo(() => arrivals(stops.length, solution), [stops, solution]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || styleVersion === 0) return;
    (map.getSource("stops") as GeoJSONSource).setData({
      type: "FeatureCollection",
      features: stops.map((s, i) => point(toLngLat(s), { arrive: arrive[i] })),
    });
    (map.getSource("depot") as GeoJSONSource).setData(point(toLngLat(depot)));
  }, [depot, stops, arrive, styleVersion]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || styleVersion === 0) return;
    map.setPaintProperty("driven", "line-color", color);
    map.setPaintProperty("van", "circle-color", color);
  }, [color, styleVersion]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || styleVersion === 0) return;
    const driven = vans.map((v) => drivenBy(v, time));
    (map.getSource("driven") as GeoJSONSource).setData({
      type: "FeatureCollection",
      features: driven.map((coordinates) => ({
        type: "Feature",
        geometry: { type: "LineString", coordinates },
        properties: {},
      })),
    });
    (map.getSource("van") as GeoJSONSource).setData({
      type: "FeatureCollection",
      features: driven.filter((_, v) => time < vans[v].times.at(-1)!).map((d) => point(d.at(-1)!)),
    });
    map.setPaintProperty("stops", "circle-color", [
      "case",
      ["<=", ["get", "arrive"], time],
      color,
      UNVISITED,
    ]);
  }, [vans, time, color, styleVersion]);

  return <div ref={container} className="map" />;
}
