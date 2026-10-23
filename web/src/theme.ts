import { useSyncExternalStore } from "react";

// Light / dark theme. index.html sets data-theme on <html> before the first paint (saved choice,
// else the system's); the choice is saved under the same key as the portfolio, which shares this
// origin, so both pages open in the theme last picked.
export type Theme = "light" | "dark";

const listeners = new Set<() => void>();
const current = (): Theme => (document.documentElement.dataset.theme === "dark" ? "dark" : "light");

export function setTheme(theme: Theme) {
  document.documentElement.dataset.theme = theme;
  document.querySelector('meta[name="theme-color"]')?.setAttribute("content", theme === "dark" ? "#121419" : "#f4f1ea");
  try {
    localStorage.setItem("theme", theme);
  } catch {
    // private mode or blocked storage: the theme still applies to this visit
  }
  listeners.forEach((l) => l());
}

export function useTheme(): Theme {
  return useSyncExternalStore(
    (l) => {
      listeners.add(l);
      return () => listeners.delete(l);
    },
    current,
    () => "light",
  );
}
