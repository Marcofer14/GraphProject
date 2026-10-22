# Project assistant (Cloudflare Worker)

The demo's chat sends questions here. The Worker keeps the Groq API key off the page, picks the
relevant sections of [`knowledge.md`](knowledge.md) for each question (so each request stays small
enough for free tiers), and answers with Groq's free tier, falling back to Cloudflare Workers AI's
free daily allowance. If both are unavailable the page answers from its prepared FAQ.

Limits: 20 questions per visitor per 10 minutes (per Worker instance), questions under 600
characters, conversations under 12 messages, answers under 350 tokens, only questions about the
project (a word-overlap guard refuses the rest without calling a model, and the prompt refuses
what slips through), and only requests from the
demo's origin.

## Deploy (once)

You need a free Cloudflare account and a free Groq API key (console.groq.com → API Keys). Never
paste the key in a chat, a file or a commit; it only goes into the `wrangler secret put` prompt.

```bash
cd chat
npm install
npx wrangler login                      # opens the browser to authorise Wrangler
npx wrangler secret put GROQ_API_KEY    # paste the key when asked; stored encrypted
npx wrangler deploy                     # prints https://neural-routing-chat.<you>.workers.dev
```

Then put that URL in `web/.env.production` as `VITE_CHAT_URL=https://...workers.dev` and push: the
demo is rebuilt with the chat connected. Check it with `curl https://...workers.dev/health`.

To rotate the key: create a new one in Groq, run `npx wrangler secret put GROQ_API_KEY` again, and
delete the old one in Groq.

## Change it

* Facts: edit `knowledge.md` (one `## ` section per topic) and `npx wrangler deploy`.
* Models: `GROQ_MODELS` (tried in order) and `WORKERS_AI_MODEL` in `wrangler.toml`. Providers retire
  models often: if every answer fails, check that these IDs still exist on the free plans.
* Tests: `npm test`. Local run without a Cloudflare login: `npx wrangler dev --env local`
  (put `GROQ_API_KEY=...` in `.dev.vars`, which is git-ignored, to use Groq locally).
