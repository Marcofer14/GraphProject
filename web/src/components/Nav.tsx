import { openChat } from "../chat/assistant";
import { setTheme, useTheme } from "../theme";

const LINKS = [
  ["#intro", "The problem"],
  ["#demo", "Demo"],
  ["#why", "Why it works"],
  ["#use-cases", "Use cases"],
  ["#architecture", "Architecture"],
  ["#limits", "Limits"],
];

/** Sticky menu in the portfolio's style; the links scroll sideways on narrow screens. */
export function Nav() {
  const theme = useTheme();
  const next = theme === "dark" ? "light" : "dark";
  return (
    <nav className="topnav" aria-label="Sections">
      <a className="brand" href="#intro">
        Neural <i>routing</i>
      </a>
      <div className="topnav-links">
        {LINKS.map(([href, label]) => (
          <a key={href} href={href}>
            {label}
          </a>
        ))}
      </div>
      <a className="back" href="https://marcofer14.github.io/">
        ← Portfolio
      </a>
      <button
        type="button"
        className="theme-toggle"
        aria-label={`Switch to ${next} mode`}
        title="Light / dark"
        onClick={() => setTheme(next)}
      >
        {theme === "dark" ? (
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <circle cx="12" cy="12" r="4.5" />
            <path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8" />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
          </svg>
        )}
      </button>
      <button type="button" className="primary" onClick={openChat}>
        Ask
      </button>
    </nav>
  );
}
