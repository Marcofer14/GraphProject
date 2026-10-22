const LINKS = [
  ["#intro", "Intro"],
  ["#how-to", "How to use"],
  ["#example", "Example"],
  ["#compare", "Compare"],
  ["#race", "Race"],
  ["#day", "Whole day"],
  ["#use-cases", "Use cases"],
  ["#architecture", "Architecture"],
  ["#results", "Results"],
];

/** Sticky section menu; scrolls sideways on narrow screens. */
export function Nav() {
  return (
    <nav className="topnav" aria-label="Sections">
      <a className="brand" href="#intro">
        Neural routing
      </a>
      <div className="topnav-links">
        {LINKS.map(([href, label]) => (
          <a key={href} href={href}>
            {label}
          </a>
        ))}
      </div>
    </nav>
  );
}
