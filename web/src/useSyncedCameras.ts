import type { Map as MLMap } from "maplibre-gl";
import { useEffect } from "react";

/** Keep every given map on the same camera: moving one moves the others. */
export function useSyncedCameras(maps: (MLMap | null)[]) {
  useEffect(() => {
    const ready = maps.filter((m): m is MLMap => m !== null);
    if (ready.length < 2) return;
    let syncing = false;
    const handlers = ready.map((from) => () => {
      if (syncing) return;
      syncing = true;
      const camera = {
        center: from.getCenter(),
        zoom: from.getZoom(),
        bearing: from.getBearing(),
        pitch: from.getPitch(),
      };
      ready.filter((to) => to !== from).forEach((to) => to.jumpTo(camera));
      syncing = false;
    });
    ready.forEach((m, i) => m.on("move", handlers[i]));
    return () => ready.forEach((m, i) => m.off("move", handlers[i]));
  }, maps); // the maps themselves are the deps (a fixed number per caller)
}
