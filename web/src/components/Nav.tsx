import { openChat } from "../chat/assistant";

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
      <button type="button" className="primary" onClick={openChat}>
        Ask
      </button>
    </nav>
  );
}
