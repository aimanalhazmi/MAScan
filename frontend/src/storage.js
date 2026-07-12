// Conversation history in localStorage. A conversation is a chat with its
// messages and the captured run state (node status + reports) so reopening it
// restores the graph too.

const KEY = "mascan.conversations";
const GRAPH_KEY = "mascan.graphs";

export function loadConversations() {
  try {
    return JSON.parse(localStorage.getItem(KEY)) || [];
  } catch {
    return [];
  }
}

// The graph only needs node states, the plan, and the clarifications. Keeping
// that slice tiny lets it persist on its own, so the graph survives a refresh
// even when the full chat (with heavy report bodies) is too big to store.
function graphSlice(run) {
  if (!run) return null;
  const { status, nodeStatus, plan, failures, clarifications, summary } = run;
  return { status, nodeStatus, plan, failures, clarifications, summary };
}

export function saveConversations(conversations) {
  try {
    const graphs = Object.fromEntries(conversations.map((c) => [c.id, graphSlice(c.run)]));
    localStorage.setItem(GRAPH_KEY, JSON.stringify(graphs));
  } catch {
    // Best-effort backup; never fatal.
  }
  try {
    localStorage.setItem(KEY, JSON.stringify(conversations));
  } catch {
    // Over quota: drop the heavy report bodies but keep the rest of the chat.
    const light = conversations.map((c) =>
      c.run ? { ...c, run: { ...c.run, reports: {}, finalMarkdown: "" } } : c
    );
    try {
      localStorage.setItem(KEY, JSON.stringify(light));
    } catch {
      // Give up silently rather than crash the app.
    }
  }
}

// Restore a conversation's graph state: the full run if it was saved, otherwise
// the lightweight snapshot that always persists.
export function loadRun(conversation) {
  if (conversation?.run) return conversation.run;
  try {
    const map = JSON.parse(localStorage.getItem(GRAPH_KEY)) || {};
    return map[conversation?.id] || null;
  } catch {
    return null;
  }
}

export function newConversation() {
  return {
    id: crypto.randomUUID(),
    title: "New chat",
    messages: [],
    run: null,
    createdAt: Date.now(),
  };
}

// First user message becomes the title.
export function titleFrom(text) {
  const t = text.trim().replace(/\s+/g, " ");
  return t.length > 48 ? t.slice(0, 48) + "…" : t || "New chat";
}
