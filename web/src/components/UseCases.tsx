import type { ReactNode } from "react";

// simple line icons, drawn with currentColor
const ICONS: Record<string, ReactNode> = {
  parcel: (
    <>
      <path d="M3 7.5 12 3l9 4.5v9L12 21l-9-4.5z" />
      <path d="M3 7.5 12 12l9-4.5M12 12v9M7.5 5.2l9 4.6" />
    </>
  ),
  replan: (
    <>
      <path d="M20 11a8 8 0 1 0-2.3 5.7" />
      <path d="M20 4v7h-7" />
    </>
  ),
  station: (
    <>
      <path d="M3 21V9l9-5 9 5v12" />
      <path d="M8 21v-6h8v6M3 21h18" />
    </>
  ),
  windows: (
    <>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M12 7v5l3 2" />
    </>
  ),
  chip: (
    <>
      <rect x="6" y="6" width="12" height="12" rx="2" />
      <path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4" />
    </>
  ),
  whatif: (
    <>
      <path d="M6 3v6a3 3 0 0 0 3 3h6a3 3 0 0 1 3 3v6" />
      <circle cx="6" cy="3" r="0.5" />
      <path d="M18 3v4M16 5h4" />
    </>
  ),
};

function Icon({ name }: { name: string }) {
  return (
    <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
      {ICONS[name]}
    </svg>
  );
}

const CASES = [
  {
    icon: "parcel",
    hue: "var(--route)",
    tag: "Built and tested here",
    title: "Parcel and e-commerce delivery",
    text: "Vans with 100-200 stops a day, volume limits and delivery windows. Shorter routes mean fewer hours, less fuel and fewer late parcels.",
  },
  {
    icon: "replan",
    hue: "var(--poke)",
    tag: "Half a second",
    title: "Re-planning during the day",
    text: "An urgent order, a van that breaks down, a cancelled stop: the route is recomputed on the spot instead of waiting for an overnight batch.",
  },
  {
    icon: "station",
    hue: "var(--truco)",
    tag: "About 2 s",
    title: "Planning a whole station",
    text: "Splitting a day's stops between the vans and routing each one, for up to a thousand stops, where a general solver needs much longer.",
  },
  {
    icon: "windows",
    hue: "var(--uni)",
    tag: "Retrain on your trips",
    title: "Other routing with stops and time windows",
    text: "Grocery and meal-kit delivery, pharmacy and B2B distribution, field technicians, waste collection: the same problem shape.",
  },
  {
    icon: "chip",
    hue: "var(--iberia)",
    tag: "No licences",
    title: "Modest or on-site hardware",
    text: "Half a second on a 2019 mid-range gaming GPU, about two seconds on an ordinary desktop CPU: no data-centre hardware.",
  },
  {
    icon: "whatif",
    hue: "var(--indigo)",
    tag: "Cheap to ask",
    title: "Quick what-if analysis",
    text: "Because each plan is cheap, many scenarios can be tried: one van fewer, a later start, a new station.",
  },
];

const NOT_FOR = [
  {
    title: "There is time to spare",
    text: "Given 10 seconds or more on a CPU, a mature solver like PyVRP is still slightly better on average.",
  },
  {
    title: "Hard rules come first",
    text: "Driver breaks, legal limits, vehicle access: the model treats time windows as soft and does not know these rules.",
  },
  {
    title: "Local knowledge is the point",
    text: "Parking, building access and customer habits are not in the data; it does not replace what drivers know.",
  },
  {
    title: "The problem is very different",
    text: "It learned from Amazon routes in five US cities in 2018; another kind of problem needs retraining.",
  },
];

export function UseCases() {
  return (
    <section className="block" id="use-cases">
      <div className="sec-head">
        <h2>
          Where it is <i>useful</i>
        </h2>
        <span className="num">04</span>
      </div>
      <p className="lead">
        The model plans good routes fast, on cheap hardware. That matters most where plans have to be made often, or
        changed on short notice.
      </p>
      <div className="bento">
        {CASES.map((c) => (
          <article key={c.title} className="use" style={{ ["--hue" as string]: c.hue }}>
            <div className="use-top">
              <span className="use-icon">
                <Icon name={c.icon} />
              </span>
              <span className="use-tag">{c.tag}</span>
            </div>
            <h3>{c.title}</h3>
            <p>{c.text}</p>
          </article>
        ))}
      </div>
      <div className="notfor">
        <div className="notfor-head">
          <p className="eyebrow">Fit check</p>
          <h3>Not the right tool when…</h3>
        </div>
        <ul>
          {NOT_FOR.map((n) => (
            <li key={n.title}>
              <span className="x" aria-hidden="true">
                ✕
              </span>
              <div>
                <b>{n.title}</b>
                <p>{n.text}</p>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
