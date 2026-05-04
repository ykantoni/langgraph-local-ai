import { useMemo, useState } from 'react'
import './App.css'

type ChatRole = 'user' | 'assistant'
type ChatMessage = { role: ChatRole; content: string }

const DEFAULT_API_BASE = 'http://127.0.0.1:8000'

function App() {
  const apiBase =
    (import.meta.env.VITE_API_BASE as string | undefined) || DEFAULT_API_BASE

  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const canSend = input.trim().length > 0 && !isSending

  const history = useMemo(
    () =>
      messages.map((m) => ({
        role: m.role,
        content: m.content,
      })),
    [messages],
  )

  async function send() {
    const text = input.trim()
    if (!text || isSending) return

    setError(null)
    setIsSending(true)
    setInput('')

    // Optimistic user message
    setMessages((prev) => [...prev, { role: 'user', content: text }])

    try {
      const res = await fetch(`${apiBase}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          history,
        }),
      })

      if (!res.ok) {
        const bodyText = await res.text()
        throw new Error(`${res.status} ${res.statusText}${bodyText ? `: ${bodyText}` : ''}`)
      }

      const data = (await res.json()) as { response?: string }
      const reply = (data?.response ?? '').toString().trim()
      setMessages((prev) => [...prev, { role: 'assistant', content: reply || '(empty response)' }])
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      setError(msg)
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content:
            'Request failed. Check that the API server is running and CORS is configured.',
        },
      ])
    } finally {
      setIsSending(false)
    }
  }

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">Local Agent Chat</div>
        <div className="meta">
          <span className="pill">API: {apiBase}</span>
          <a className="pill link" href={`${apiBase}/docs`} target="_blank" rel="noreferrer">
            API docs
          </a>
        </div>
      </header>

      <main className="chat">
        {messages.length === 0 ? (
          <div className="empty">
            <div className="emptyTitle">Ask a question about your local docs</div>
            <div className="emptySub">
              The UI sends the full chat history (roles user/assistant) to the backend each turn.
            </div>
          </div>
        ) : null}

        <div className="messages">
          {messages.map((m, idx) => (
            <div key={idx} className={`msgRow ${m.role}`}>
              <div className="msgBubble">
                <div className="msgRole">{m.role}</div>
                <div className="msgText">{m.content}</div>
              </div>
            </div>
          ))}
        </div>
      </main>

      <footer className="composer">
        {error ? <div className="error">{error}</div> : null}
        <div className="composerRow">
          <textarea
            className="input"
            value={input}
            placeholder="Type a message…"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                void send()
              }
            }}
            rows={2}
          />
          <button className="send" onClick={() => void send()} disabled={!canSend}>
            {isSending ? 'Sending…' : 'Send'}
          </button>
        </div>
        <div className="hint">Enter to send, Shift+Enter for newline.</div>
      </footer>
    </div>
  )
}

export default App
