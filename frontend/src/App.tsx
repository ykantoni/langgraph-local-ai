import { useEffect, useRef, useState } from 'react'
import './App.css'

type ChatRole = 'user' | 'assistant'
type ChatMessage = { role: ChatRole; content: string }

const DEFAULT_API_BASE = 'http://127.0.0.1:8000'
const SESSION_KEY = 'local-agent.session-id'

function App() {
  const apiBase =
    (import.meta.env.VITE_API_BASE as string | undefined) || DEFAULT_API_BASE

  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [exitNotice, setExitNotice] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const [sessionId, setSessionId] = useState<string | null>(
    () => window.localStorage.getItem(SESSION_KEY) || null,
  )

  const canSend = input.trim().length > 0 && !isSending

  useEffect(() => {
    if (sessionId) return
    let cancelled = false
    ;(async () => {
      try {
        const res = await fetch(`${apiBase}/session/new`, { method: 'POST' })
        const data = (await res.json()) as { session_id?: string }
        if (!cancelled && data.session_id) {
          window.localStorage.setItem(SESSION_KEY, data.session_id)
          setSessionId(data.session_id)
        }
      } catch {
        // ignore; user can still use stateless history if needed
      }
    })()
    return () => {
      cancelled = true
    }
  }, [apiBase, sessionId])

  function appendAssistantPlaceholder() {
    setMessages((prev) => [...prev, { role: 'assistant', content: '' }])
  }

  function updateLastAssistant(appendText: string) {
    if (!appendText) return
    setMessages((prev) => {
      const next = [...prev]
      for (let i = next.length - 1; i >= 0; i -= 1) {
        if (next[i]?.role === 'assistant') {
          next[i] = { ...next[i], content: (next[i].content ?? '') + appendText }
          return next
        }
      }
      return next
    })
  }

  async function send() {
    const text = input.trim()
    if (!text || isSending) return

    setError(null)
    setIsSending(true)
    setInput('')
    abortRef.current?.abort()
    abortRef.current = new AbortController()

    // Optimistic user message
    setMessages((prev) => [...prev, { role: 'user', content: text }])
    appendAssistantPlaceholder()

    try {
      const res = await fetch(`${apiBase}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: abortRef.current.signal,
        body: JSON.stringify({
          message: text,
          session_id: sessionId,
        }),
      })

      if (!res.ok) {
        const bodyText = await res.text()
        throw new Error(`${res.status} ${res.statusText}${bodyText ? `: ${bodyText}` : ''}`)
      }

      if (!res.body) throw new Error('No response body (stream not supported?)')

      const reader = res.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = ''

      const processFrames = () => {
        // SSE frames end with a blank line
        while (true) {
          const frameEnd = buffer.indexOf('\n\n')
          if (frameEnd === -1) break
          const frame = buffer.slice(0, frameEnd)
          buffer = buffer.slice(frameEnd + 2)

          const dataLine = frame
            .split('\n')
            .find((l) => l.startsWith('data: '))
            ?.slice('data: '.length)

          if (!dataLine) continue

          type StreamEvent =
            | { type: 'start' }
            | { type: 'token'; token: string }
            | { type: 'error'; error: string }
            | { type: 'done' }
            | { type: string; [k: string]: unknown }

          let evt: StreamEvent | null = null
          try {
            evt = JSON.parse(dataLine) as StreamEvent
          } catch {
            continue
          }

          if (evt?.type === 'token') {
            const token = typeof (evt as any).token === 'string' ? (evt as any).token : ''
            updateLastAssistant(token)
          } else if (evt?.type === 'error') {
            const msg =
              typeof (evt as any).error === 'string' ? (evt as any).error : 'Unknown error'
            setError(msg)
          } else if (evt?.type === 'done') {
            // finish
          }
        }
      }

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        processFrames()
      }

      // Flush the TextDecoder to avoid losing multi-byte UTF-8 chars that end exactly
      // at the final chunk boundary.
      buffer += decoder.decode()
      processFrames()
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') {
        updateLastAssistant('\n\n(stopped)')
        return
      }
      const msg = e instanceof Error ? e.message : String(e)
      setError(msg)
      updateLastAssistant(
        '\n\nRequest failed. Check that the API server is running and CORS is configured.',
      )
    } finally {
      setIsSending(false)
      abortRef.current = null
    }
  }

  function stopGenerating() {
    abortRef.current?.abort()
  }

  async function newChat() {
    try {
      const res = await fetch(`${apiBase}/session/new`, { method: 'POST' })
      const data = (await res.json()) as { session_id?: string }
      if (data.session_id) {
        window.localStorage.setItem(SESSION_KEY, data.session_id)
        setSessionId(data.session_id)
        setMessages([])
        setError(null)
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      setError(msg)
    }
  }

  function exitApp() {
    setExitNotice(
      'To stop the app, close this tab and press Ctrl+C in the terminals running the frontend/backend.',
    )
    // Browsers only allow scripts to close windows opened by scripts.
    if (window.opener) window.close()
  }

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">Local Agent Chat</div>
        <div className="meta">
          <span className="pill">API: {apiBase}</span>
          {sessionId ? <span className="pill">session: {sessionId.slice(0, 8)}…</span> : null}
          <a className="pill link" href={`${apiBase}/docs`} target="_blank" rel="noreferrer">
            API docs
          </a>
          <button className="pill button" type="button" onClick={() => void newChat()}>
            New chat
          </button>
          <button className="pill button" type="button" onClick={exitApp}>
            Exit
          </button>
        </div>
      </header>

      <main className="chat">
        {exitNotice ? <div className="notice">{exitNotice}</div> : null}
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
          {isSending ? (
            <button className="stop" onClick={stopGenerating} type="button">
              Stop generating
            </button>
          ) : (
            <button className="send" onClick={() => void send()} disabled={!canSend}>
              Send
            </button>
          )}
        </div>
        <div className="hint">Enter to send, Shift+Enter for newline.</div>
      </footer>
    </div>
  )
}

export default App
