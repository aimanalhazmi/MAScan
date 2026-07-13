// Conversation history list, ChatGPT/OpenWebUI style.
export default function ChatSidebar({ conversations, activeId, view, onNew, onSelect, onDelete, onOpenDocuments, theme, onToggleTheme }) {
  return (
    <aside className="sidebar">
      <div className="brand">MA<span>Scan</span></div>
      <button className="new-chat" onClick={onNew}>+ New chat</button>
      <button
        className={`nav-item${view === "documents" ? " active" : ""}`}
        onClick={onOpenDocuments}
      >
        ▤ Documents
      </button>
      <div className="chat-list">
        {conversations.map((c) => (
          <div
            key={c.id}
            className={`chat-item${view === "chat" && c.id === activeId ? " active" : ""}`}
            onClick={() => onSelect(c.id)}
          >
            <span className="chat-title">{c.title}</span>
            <button
              className="icon-btn"
              aria-label="Delete chat"
              onClick={(e) => { e.stopPropagation(); onDelete(c.id); }}
            >
              ✕
            </button>
          </div>
        ))}
      </div>
      <button className="theme-toggle" onClick={onToggleTheme}>
        {theme === "dark" ? "☀ Light" : "☾ Dark"}
      </button>
    </aside>
  );
}
