import { Fragment, type ReactNode } from "react";

import { faqAnswer } from "./faq";

// URL of the Cloudflare Worker (chat/); without it, typed questions get the prepared FAQ answers
export const CHAT_URL = import.meta.env.VITE_CHAT_URL as string | undefined;
export const MAX_CHARS = 600;
export const MAX_TURNS = 12; // the Worker refuses longer conversations

export interface Turn {
  role: "user" | "assistant";
  content: string;
}

/** Answer a typed question: the AI assistant when connected, the closest prepared answer otherwise. */
export async function askAssistant(history: Turn[]): Promise<{ answer: string; note?: string }> {
  const question = history.at(-1)!.content;
  const prepared = () => {
    const faq = faqAnswer(question);
    return faq
      ? { answer: faq.a, note: "Prepared answer: the AI assistant is not available right now." }
      : {
          answer:
            "I don't have a prepared answer for that, and the AI assistant is not available right now. Try one of the questions above, or see the GitHub repository.",
        };
  };
  if (!CHAT_URL) return prepared();
  try {
    const res = await fetch(`${CHAT_URL.replace(/\/$/, "")}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history.slice(-MAX_TURNS).map(({ role, content }) => ({ role, content })) }),
    });
    const data = (await res.json()) as { answer?: string; error?: string };
    if (res.ok && data.answer) return { answer: data.answer };
    if (res.status === 400 && data.error) return { answer: data.error };
    return prepared();
  } catch {
    return prepared();
  }
}

/** **bold**, *italics* and "- " bullets from the model's markdown, without injecting HTML. */
export function Rich({ text }: { text: string }) {
  const inline = (line: string): ReactNode[] =>
    line.split(/(\*\*[^*]+\*\*|\*[^*\s][^*]*\*)/g).map((part, i) => {
      if (part.startsWith("**") && part.endsWith("**")) return <b key={i}>{part.slice(2, -2)}</b>;
      if (part.length > 2 && part.startsWith("*") && part.endsWith("*")) return <i key={i}>{part.slice(1, -1)}</i>;
      return <Fragment key={i}>{part}</Fragment>;
    });
  const blocks: ReactNode[] = [];
  let bullets: string[] = [];
  const flush = () => {
    if (bullets.length)
      blocks.push(
        <ul key={blocks.length}>
          {bullets.map((b, i) => (
            <li key={i}>{inline(b)}</li>
          ))}
        </ul>,
      );
    bullets = [];
  };
  for (const line of text.split("\n")) {
    const m = line.match(/^\s*(?:[-*•]|\d+\.)\s+(.*)$/);
    if (m) bullets.push(m[1]);
    else {
      flush();
      if (line.trim()) blocks.push(<p key={blocks.length}>{inline(line)}</p>);
    }
  }
  flush();
  return <>{blocks}</>;
}

/** Scroll to the help panel and focus the question box (menu and intro buttons use it). */
export const openChat = () => window.dispatchEvent(new Event("open-chat"));
