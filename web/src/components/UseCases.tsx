const CASES = [
  {
    title: "Parcel and e-commerce delivery",
    text: "The case it was built and tested on: vans with 100-200 stops a day, volume limits and delivery windows. Shorter routes mean fewer hours, less fuel and fewer late parcels.",
  },
  {
    title: "Re-planning during the day",
    text: "An urgent order arrives, a van breaks down, a customer cancels. A plan in half a second means the route can be recomputed on the spot instead of waiting for an overnight batch.",
  },
  {
    title: "Planning a whole station",
    text: "Splitting all of a day's stops between the vans and routing each one, in about two seconds for up to a thousand stops, where a general solver needs much longer.",
  },
  {
    title: "Other routing with stops and time windows",
    text: "Grocery and meal-kit delivery, pharmacy and B2B distribution, field technicians, waste collection: the same problem shape, retrained on that business's own trips.",
  },
  {
    title: "Modest or on-site hardware",
    text: "It plans in half a second on a 2019 mid-range gaming GPU and in about two seconds on an ordinary desktop CPU: no data-centre hardware or paid solver licences needed.",
  },
  {
    title: "Quick what-if analysis",
    text: "Because each plan is cheap, many scenarios can be tried: one van fewer, a later start, a new station. Useful for planners, not only for daily dispatch.",
  },
];

const NOT_FOR = [
  "When there is time to spare: given 10 seconds or more on a CPU, a mature solver like PyVRP is still slightly better on average.",
  "When hard rules matter more than driving time (driver breaks, legal limits, vehicle access): the model treats time windows as soft and does not know these rules.",
  "As a drop-in for drivers' local knowledge: parking, building access and customer habits are not in the data.",
  "On a very different kind of problem without retraining: it learned from Amazon routes in five US cities in 2018.",
];

export function UseCases() {
  return (
    <section className="block" id="use-cases">
      <h2>Where it is useful</h2>
      <p className="lead">
        The model plans good routes fast, on cheap hardware. That matters most where plans have to be made often, or
        changed on short notice.
      </p>
      <div className="cards">
        {CASES.map((c) => (
          <article key={c.title} className="card">
            <p className="card-title">{c.title}</p>
            <p className="card-text">{c.text}</p>
          </article>
        ))}
      </div>
      <h3>Where it is not the right tool</h3>
      <ul className="limits">
        {NOT_FOR.map((t) => (
          <li key={t}>{t}</li>
        ))}
      </ul>
    </section>
  );
}
