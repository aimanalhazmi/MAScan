import { useEffect, useRef, useState } from "react";
import ChatSidebar from "./components/ChatSidebar";
import Conversation from "./components/Conversation";
import GraphRail from "./components/GraphRail";
import NodeDetail from "./components/NodeDetail";
import { useAnalysisStream, emptyRun } from "./useAnalysisStream";
import { loadConversations, saveConversations, newConversation, titleFrom } from "./storage";

export default function App() {
  const [conversations, setConversations] = useState(() => {
    const loaded = loadConversations();
    return loaded.length ? loaded : [newConversation()];
  });
  const [activeId, setActiveId] = useState(() => conversations[0].id);
  const [selectedNode, setSelectedNode] = useState(null);
  const [theme, setTheme] = useState(() => localStorage.getItem("mascan.theme") || "light");
  const [leftW, setLeftW] = useState(() => Number(localStorage.getItem("mascan.leftW")) || 250);
  const [rightW, setRightW] = useState(() => Number(localStorage.getItem("mascan.rightW")) || 340);

  const stream = useAnalysisStream();
  const active = conversations.find((c) => c.id === activeId) || conversations[0];

  useEffect(() => saveConversations(conversations), [conversations]);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("mascan.theme", theme);
  }, [theme]);
  useEffect(() => localStorage.setItem("mascan.leftW", leftW), [leftW]);
  useEffect(() => localStorage.setItem("mascan.rightW", rightW), [rightW]);

  // Mirror the live run into the active conversation, and append the assistant
  // message once the run finishes.
  useEffect(() => {
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== activeId) return c;
        const last = c.messages[c.messages.length - 1];
        const finished =
          stream.run.status === "done" && last && last.role === "user";
        const messages = finished
          ? [...c.messages, { role: "assistant", content: stream.run.finalMarkdown || stream.run.summary || "No report produced." }]
          : c.messages;
        return { ...c, run: stream.run, messages };
      })
    );
  }, [stream.run]); // eslint-disable-line react-hooks/exhaustive-deps

  function selectConversation(id) {
    const conv = conversations.find((c) => c.id === id);
    setActiveId(id);
    setSelectedNode(null);
    stream.reset(conv?.run || emptyRun());
  }

  function createConversation() {
    const conv = newConversation();
    setConversations((prev) => [conv, ...prev]);
    setActiveId(conv.id);
    setSelectedNode(null);
    stream.reset(emptyRun());
  }

  function deleteConversation(id) {
    setConversations((prev) => {
      const next = prev.filter((c) => c.id !== id);
      if (next.length === 0) next.push(newConversation());
      if (id === activeId) {
        setActiveId(next[0].id);
        setSelectedNode(null);
        stream.reset(next[0].run || emptyRun());
      }
      return next;
    });
  }

  function send(query) {
    setConversations((prev) =>
      prev.map((c) =>
        c.id === activeId
          ? {
              ...c,
              title: c.messages.length === 0 ? titleFrom(query) : c.title,
              messages: [...c.messages, { role: "user", content: query }],
            }
          : c
      )
    );
    setSelectedNode(null);
    stream.start(query);
  }

  const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));

  return (
    <div className="app" style={{ "--left-w": `${leftW}px`, "--right-w": `${rightW}px` }}>
      <ChatSidebar
        conversations={conversations}
        activeId={activeId}
        onNew={createConversation}
        onSelect={selectConversation}
        onDelete={deleteConversation}
        theme={theme}
        onToggleTheme={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
      />
      <Resizer onMove={(dx) => setLeftW((w) => clamp(w + dx, 180, 480))} />
      <Conversation
        messages={active.messages}
        run={stream.run}
        onSend={send}
        onResume={(answer) => stream.resume(stream.run.clarification?.thread_id, answer)}
      />
      <Resizer onMove={(dx) => setRightW((w) => clamp(w - dx, 260, 620))} />
      <div className="right-rail">
        <GraphRail run={stream.run} selected={selectedNode} onSelect={setSelectedNode} />
        {selectedNode && (
          <NodeDetail run={stream.run} nodeId={selectedNode} onClose={() => setSelectedNode(null)} />
        )}
      </div>
    </div>
  );
}

// Drag handle between panels. Reports horizontal movement while dragging.
function Resizer({ onMove }) {
  const dragging = useRef(false);
  return (
    <div
      className="resizer"
      onPointerDown={(e) => {
        dragging.current = true;
        e.currentTarget.setPointerCapture(e.pointerId);
      }}
      onPointerMove={(e) => dragging.current && onMove(e.movementX)}
      onPointerUp={(e) => {
        dragging.current = false;
        e.currentTarget.releasePointerCapture(e.pointerId);
      }}
    />
  );
}
