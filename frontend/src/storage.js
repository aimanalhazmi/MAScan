// Conversation history in localStorage. A conversation is a chat with its
// messages and the captured run state (node status + reports) so reopening it
// restores the graph too.

const KEY = "mascan.conversations";

export function loadConversations() {
  try {
    return JSON.parse(localStorage.getItem(KEY)) || [];
  } catch {
    return [];
  }
}

export function saveConversations(conversations) {
  localStorage.setItem(KEY, JSON.stringify(conversations));
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
