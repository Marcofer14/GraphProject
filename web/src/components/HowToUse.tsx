const TOOLS = [
  {
    href: "#compare",
    title: "Compare two plans on one route",
    steps: [
      "Pick a route from the list: 25 real routes, five per city, drawn at random.",
      "Choose the plan for each map: the real driver, our model, the network alone, OR-Tools or PyVRP.",
      "Press “Replay the shift” to watch both vans leave the station at the same time, or drag the time slider.",
    ],
  },
  {
    href: "#race",
    title: "Race the solvers against the clock",
    steps: [
      "Move the computing-time slider, click on the chart, or use the 1 s, 2 s and 10 s buttons.",
      "Each map shows the route that solver returns with that much time, or “still computing”.",
      "Switch our model between GPU and CPU to see how much the graphics card matters.",
    ],
  },
  {
    href: "#day",
    title: "Re-plan a whole station-day",
    steps: [
      "Pick a day: all the stops one station delivered with 3 to 6 vans.",
      "Each van has its own colour; the same area keeps its colour on both maps.",
      "Replay the day or compare fleet hours, late stops and computing time.",
    ],
  },
];

const TERMS = [
  ["Driving time", "Hours behind the wheel, from Amazon's real travel times between stops (not straight lines)."],
  [
    "Above the best route",
    "Extra driving compared with the best route any solver found for the same stops, leaving out the trip to " +
      "and from the station that every plan must make. 0 % is the best; the real drivers are at about 22 %.",
  ],
  ["Late stops", "Deliveries reached after their delivery window closed. Every plan is checked the same way."],
  ["Compute", "How long the solver took to produce the plan, on a GPU for our network and one CPU core otherwise."],
];

export function HowToUse() {
  return (
    <section className="block" id="how-to">
      <h2>How to use this page</h2>
      <p className="lead">
        Three interactive tools, all on real routes the model never saw during training. Nothing is computed while you
        browse: every plan was solved once and saved, so what you see is exactly what each solver produced.
      </p>
      <div className="cards">
        {TOOLS.map((tool, i) => (
          <article key={tool.href} className="card tool">
            <span className="step-n" aria-hidden="true">
              {i + 1}
            </span>
            <p className="card-title">
              <a href={tool.href}>{tool.title}</a>
            </p>
            <ol className="tool-steps">
              {tool.steps.map((s) => (
                <li key={s}>{s}</li>
              ))}
            </ol>
          </article>
        ))}
      </div>
      <h3>Reading the numbers</h3>
      <dl className="terms">
        {TERMS.map(([term, text]) => (
          <div key={term}>
            <dt>{term}</dt>
            <dd>{text}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}
