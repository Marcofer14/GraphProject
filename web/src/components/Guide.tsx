import type { Mode } from "./Workspace";

const GUIDES: Record<Mode, { title: string; intro: string; steps: string[] }> = {
  compare: {
    title: "Compare two plans",
    intro: "The same stops planned two ways, side by side. 25 real routes, five per city, drawn at random.",
    steps: [
      "Pick a route in the list.",
      "Choose what each map shows: the real driver, our model, the network alone, OR-Tools or PyVRP.",
      "Press “Replay the shift”: both vans leave the station at once. Drag the slider to jump in time.",
      "Below each map: total driving, the difference with the other map, late stops and computing time.",
    ],
  },
  race: {
    title: "The race",
    intro: "Every solver gets the same computing time. Who has the best route after 1, 2 or 10 seconds?",
    steps: [
      "Move the time slider, click on the chart, or use the 1 s, 2 s and 10 s buttons.",
      "The chart shows how far each solver is from the best route as it gets more time.",
      "The three maps show the route each solver has at that moment, or “still computing”.",
      "Switch our model between GPU and CPU to see how much the graphics card matters.",
    ],
  },
  day: {
    title: "A whole day",
    intro: "Every stop a station delivered in one day, re-planned from scratch with the same vans.",
    steps: [
      "Pick a day: 20 real days the model never saw, 3 to 6 vans each.",
      "Each van has its own colour; the same area keeps its colour on both maps.",
      "Choose the plans to compare and replay the day: all vans leave at once.",
      "Compare fleet driving, vans, late stops and computing time below the maps.",
    ],
  },
};

const TERMS = [
  ["Driving", "Hours behind the wheel, from Amazon's real driving times between stops."],
  [
    "Above the best route",
    "Extra driving compared with the best route any solver found for the same stops, leaving out the trip to and from the station. 0 % is the best.",
  ],
  ["Late stops", "Deliveries reached after their window closed."],
  ["Compute", "Time to produce the plan: GPU for our network, one CPU core otherwise."],
];

export function Guide({ mode }: { mode: Mode }) {
  const g = GUIDES[mode];
  return (
    <aside className="side-card" aria-label="How to use this demo">
      <p className="eyebrow">How to use it</p>
      <h3>{g.title}</h3>
      <p className="small muted">{g.intro}</p>
      <ol className="guide-steps">
        {g.steps.map((s) => (
          <li key={s}>{s}</li>
        ))}
      </ol>
      <dl className="terms">
        {TERMS.map(([t, d]) => (
          <div key={t}>
            <dt>{t}</dt>
            <dd>{d}</dd>
          </div>
        ))}
      </dl>
    </aside>
  );
}
