'use client'

import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { askCampusGPT, type AskMetadata, type ChatSource } from '@/services/api'
import {
  ArrowUpRight,
  BookOpen,
  Bot,
  Home,
  Library,
  Menu,
  MessageSquare,
  Search,
  Send,
  X,
  Sun,
  Moon,
} from 'lucide-react'

const nav = [
  ['Dashboard', Home],
  ['Ask CampusGPT', MessageSquare],
] as const

const suggested = [
  'Who is the HOD of the AIML department?',
  'What laboratories are available in ENTC?',
  'What does SY-A have on Monday?',
  'Who teaches Deep Learning?',
  'What are the Semester 3 subjects in CS?',
  'What facilities are available in Mechanical Engineering?',
]

type ChatMessage = {
  from: 'user' | 'ai'
  text: string
  sources?: ChatSource[]
  metadata?: AskMetadata
}

function Logo() {
  return (
    <div className="flex min-w-0 items-center gap-3">
      <div className="flex h-14 w-28 shrink-0 items-center justify-center overflow-hidden rounded-lg border border-border/70 bg-white p-1 shadow-sm">
        <img
          src="https://hebbkx1anhila5yf.public.blob.vercel-storage.com/sit%20logo-ewHkMNLLhSzLoPZAiisjckH74AZHxY.jpg"
          alt="Symbiosis Institute of Technology, Pune"
          className="h-full w-full object-contain"
        />
      </div>

      <div className="min-w-0">
        <div className="truncate font-semibold tracking-tight">CampusGPT</div>
        <div className="text-[11px] text-muted-foreground">Smart University Assistant</div>
      </div>
    </div>
  )
}

/* ============================================================
   MARKDOWN ANSWER
   ============================================================ */

function MarkdownAnswer({ answer }: { answer: string }) {
  return (
    <div className="max-w-none text-sm leading-6">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
          strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
          h1: ({ children }) => (
            <h1 className="mb-3 mt-4 text-xl font-semibold first:mt-0">{children}</h1>
          ),
          h2: ({ children }) => (
            <h2 className="mb-3 mt-4 text-lg font-semibold first:mt-0">{children}</h2>
          ),
          h3: ({ children }) => (
            <h3 className="mb-2 mt-3 text-base font-semibold first:mt-0">{children}</h3>
          ),
          ul: ({ children }) => <ul className="my-2 list-disc space-y-1 pl-5">{children}</ul>,
          ol: ({ children }) => <ol className="my-2 list-decimal space-y-1 pl-5">{children}</ol>,
          li: ({ children }) => <li className="pl-1">{children}</li>,
          blockquote: ({ children }) => (
            <blockquote className="my-3 border-l-4 border-primary/30 pl-4 text-muted-foreground">
              {children}
            </blockquote>
          ),
          code: ({ children }) => (
            <code className="rounded bg-muted px-1.5 py-0.5 text-xs">{children}</code>
          ),
          pre: ({ children }) => (
            <pre className="my-3 overflow-x-auto rounded-xl bg-background p-4 text-xs">
              {children}
            </pre>
          ),
          table: ({ children }) => (
            <div className="my-4 w-full overflow-x-auto rounded-xl border border-border">
              <table className="w-full min-w-[600px] border-collapse text-sm">{children}</table>
            </div>
          ),
          thead: ({ children }) => <thead className="bg-muted/70">{children}</thead>,
          tbody: ({ children }) => <tbody>{children}</tbody>,
          tr: ({ children }) => (
            <tr className="border-b border-border last:border-b-0">{children}</tr>
          ),
          th: ({ children }) => (
            <th className="border-r border-border px-3 py-2 text-left font-semibold last:border-r-0">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="border-r border-border px-3 py-2 align-top last:border-r-0">
              {children}
            </td>
          ),
        }}
      >
        {answer}
      </ReactMarkdown>
    </div>
  )
}

function SourceList({ sources }: { sources?: { title: string; detail: string }[] }) {
  if (!sources?.length) return null
  return (
    <div className="mt-4 border-t border-border/70 pt-3">
      <div className="mb-2 text-xs font-medium text-muted-foreground">Sources</div>
      <div className="flex flex-wrap gap-2">
        {sources.map((source) => (
          <div key={`${source.title}-${source.detail}`} className="rounded-lg border border-border bg-card px-2.5 py-1.5 text-xs text-muted-foreground">
            <div className="max-w-56 truncate font-medium text-foreground">{source.title}</div>
            <div>{source.detail}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

function DetectedContext({ metadata }: { metadata?: AskMetadata }) {
  if (!metadata) return null
  const semester = typeof metadata.detected_semester === 'number'
    ? `Semester ${metadata.detected_semester}`
    : metadata.detected_semester
  const details = [
    metadata.detected_department,
    metadata.document_type?.replaceAll('_', ' '),
    ...(metadata.detected_batch ?? []),
    semester,
  ].filter((value): value is string => Boolean(value))
  if (!details.length) return null

  return <div className="mt-3 text-xs text-muted-foreground">Detected: {details.join(' · ')}</div>
}

/* ============================================================
   MAIN APPLICATION
   ============================================================ */

export function CampusGPT() {
  const [active, setActive] = useState('Dashboard')
  const [mobileOpen, setMobileOpen] = useState(false)
  const [dark, setDark] = useState<boolean | null>(null)
  const [starterQuestion, setStarterQuestion] = useState<string | null>(null)

  const go = (item: string, question?: string) => {
    setActive(item)
    setStarterQuestion(question ?? null)
    setMobileOpen(false)
  }

  useEffect(() => {
    const savedTheme = window.localStorage.getItem('campusgpt-theme')
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
    setDark(savedTheme ? savedTheme === 'dark' : prefersDark)
  }, [])

  useEffect(() => {
    if (dark === null) return
    document.documentElement.classList.toggle('dark', dark)
    document.documentElement.classList.toggle('light', !dark)
    window.localStorage.setItem('campusgpt-theme', dark ? 'dark' : 'light')
  }, [dark])

  return (
    <div className="campus-shell min-h-screen bg-background text-foreground">
      <div aria-hidden="true" className="ai-network-bg" />

      {/* SIDEBAR */}
      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r bg-card px-4 py-5 transition-transform lg:translate-x-0 ${
          mobileOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="mb-8 flex items-center justify-between px-2">
          <Logo />
          <button className="lg:hidden" onClick={() => setMobileOpen(false)} aria-label="Close menu">
            <X size={18} />
          </button>
        </div>

        <nav className="flex flex-1 flex-col gap-1">
          {nav.map(([label, Icon]) => (
            <button
              key={label}
              onClick={() => go(label)}
              className={`flex items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm transition ${
                active === label
                  ? 'bg-primary text-primary-foreground shadow-sm'
                  : 'text-muted-foreground hover:bg-muted hover:text-foreground'
              }`}
            >
              <Icon size={17} />
              {label}
            </button>
          ))}
        </nav>

        <div className="rounded-2xl bg-muted p-4">
          <div className="mb-2 flex items-center gap-2 text-xs font-medium">
            <Library size={14} />
            Knowledge Base
          </div>
          <div className="text-2xl font-semibold">
            <span className="text-sm font-semibold leading-5">Multi-department<br />knowledge base</span>
          </div>
          <div className="mt-2 h-1.5 rounded-full bg-border">
            <div className="h-full w-4/5 rounded-full bg-primary" />
          </div>
        </div>
      </aside>

      <div className="lg:pl-64">
        {/* HEADER */}
        <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b bg-background/90 px-5 backdrop-blur md:px-8">
          <button
            className="rounded-lg p-2 hover:bg-muted lg:hidden"
            onClick={() => setMobileOpen(true)}
            aria-label="Open menu"
          >
            <Menu size={20} />
          </button>

          <div className="hidden text-sm text-muted-foreground lg:block">
            Smart University Assistant
            <span className="mx-2">/</span>
            <span className="text-foreground">{active}</span>
          </div>

          <div className="flex items-center gap-3">
            <button
              className="hidden rounded-lg p-2 text-muted-foreground hover:bg-muted sm:block"
              aria-label="Search"
            >
              <Search size={18} />
            </button>

            <button
              onClick={() => setDark((current) => !(current ?? false))}
              className="grid size-9 place-items-center rounded-lg text-muted-foreground hover:bg-muted"
              aria-label="Toggle dark mode"
            >
              {dark ? <Sun size={18} /> : <Moon size={18} />}
            </button>
          </div>
        </header>

        {/* CONTENT */}
        <main className="mx-auto max-w-7xl p-5 pb-24 md:p-8 md:pb-10">
          {active === 'Dashboard' && <Dashboard onAsk={(question) => go('Ask CampusGPT', question)} />}
          {active === 'Ask CampusGPT' && <Chat initialQuestion={starterQuestion} onQuestionConsumed={() => setStarterQuestion(null)} />}
        </main>
      </div>

      {/* MOBILE NAVIGATION */}
      <div className="fixed inset-x-0 bottom-0 z-30 grid grid-cols-2 border-t bg-card/95 p-2 backdrop-blur lg:hidden">
        {[
          ['Dashboard', Home],
          ['Ask CampusGPT', MessageSquare],
        ].map(([label, Icon]) => (
          <button
            key={String(label)}
            onClick={() => go(String(label))}
            className={`flex flex-col items-center gap-1 py-1 text-[10px] ${
              active === label ? 'text-primary' : 'text-muted-foreground'
            }`}
          >
            <Icon size={18} />
            {String(label)}
          </button>
        ))}
      </div>
    </div>
  )
}

/* ============================================================
   PAGE HEADING
   ============================================================ */

function PageHeading({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string
  title: string
  description: string
}) {
  return (
    <div className="mb-8">
      <div className="mb-2 text-xs font-semibold uppercase tracking-[.18em] text-primary">
        {eyebrow}
      </div>
      <h1 className="text-balance text-3xl font-semibold tracking-tight md:text-4xl">
        {title}
      </h1>
      <p className="mt-2 max-w-2xl text-sm leading-6 text-muted-foreground">{description}</p>
    </div>
  )
}

/* ============================================================
   DASHBOARD
   ============================================================ */

function Dashboard({ onAsk }: { onAsk: (question?: string) => void }) {
  return (
    <>
      <PageHeading
        eyebrow="Good Day"
        title="Your campus, in one place."
        description="Stay on top of your academic journey with a little help from CampusGPT."
      />

      <section className="relative overflow-hidden rounded-3xl bg-primary p-6 text-primary-foreground shadow-lg md:p-8">
        <div className="relative max-w-2xl">
          <div className="mb-4 flex items-center gap-2 text-sm text-primary-foreground/75">
            <Bot size={18} />
            CampusGPT is ready to help
          </div>

          <h2 className="text-2xl font-semibold tracking-tight md:text-3xl">
            What would you like to know?
          </h2>

          <button
            onClick={() => onAsk()}
            className="mt-6 flex w-full items-center justify-between rounded-2xl bg-background px-4 py-4 text-left text-sm text-muted-foreground shadow-sm"
          >
            <span>Ask about your classes, faculty, labs...</span>
            <span className="grid size-8 place-items-center rounded-lg bg-primary text-primary-foreground">
              <ArrowUpRight size={17} />
            </span>
          </button>

          <div className="mt-4 flex flex-wrap gap-2">
            {suggested.slice(0, 2).map((q) => (
              <button
                key={q}
                onClick={() => onAsk(q)}
                className="rounded-full border border-primary-foreground/20 px-3 py-1.5 text-xs text-primary-foreground/80 hover:bg-primary-foreground/10"
              >
                {q}
              </button>
            ))}
          </div>
        </div>

        <div className="absolute -right-10 -top-16 size-64 rounded-full border border-primary-foreground/10" />
      </section>

    </>
  )
}

function Quick({
  title,
  desc,
  icon: Icon,
  onClick,
}: {
  title: string
  desc: string
  icon: typeof BookOpen
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className="group rounded-2xl border bg-card p-5 text-left transition hover:-translate-y-0.5 hover:shadow-md"
    >
      <Icon size={20} className="text-primary" />
      <div className="mt-5 font-medium">{title}</div>
      <div className="mt-1 text-xs text-muted-foreground">{desc}</div>
      <ArrowUpRight size={16} className="mt-5 text-muted-foreground transition group-hover:text-primary" />
    </button>
  )
}

/* ============================================================
   MAIN CHAT
   ============================================================ */

function Chat({
  initialQuestion,
  onQuestionConsumed,
}: {
  initialQuestion: string | null
  onQuestionConsumed: () => void
}) {
  const [input, setInput] = useState('')
  const consumedQuestion = useRef<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      from: 'ai',
      text: "Hi there. I'm CampusGPT, your Smart University Assistant. Ask naturally about departments, curriculum, faculty, laboratories, academic dates, or student information.",
    },
  ])
  const [loading, setLoading] = useState(false)

  const send = async (text = input) => {
    if (!text.trim() || loading) return
    const question = text.trim()

    setMessages((m) => [...m, { from: 'user', text: question }])
    setInput('')
    setLoading(true)

    try {
      const res = await askCampusGPT(question)
      setMessages((m) => [...m, { from: 'ai', text: res.answer, sources: res.sources, metadata: res }])
    } catch (error) {
      setMessages((m) => [
        ...m,
        {
          from: 'ai',
          text: 'I\u2019m unable to connect to the CampusGPT backend right now. Please make sure the FastAPI server is running and try again.',
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!initialQuestion || initialQuestion === consumedQuestion.current) return
    consumedQuestion.current = initialQuestion
    onQuestionConsumed()
    void send(initialQuestion)
  }, [initialQuestion])

  return (
    <div className="mx-auto max-w-4xl">
      <PageHeading
        eyebrow="Your academic assistant"
        title="Ask CampusGPT"
        description="Ask naturally about departments, faculty, curricula, facilities, academic dates, or timetables. CampusGPT identifies the relevant context from your question."
      />

      <div className="rounded-3xl border bg-card shadow-sm">
        <div className="flex min-h-[360px] flex-col gap-5 p-5 md:p-8">
          {messages.map((m, i) => (
            <div key={i} className={`flex gap-3 ${m.from === 'user' ? 'justify-end' : ''}`}>
              {m.from === 'ai' && (
                <div className="grid size-8 shrink-0 place-items-center rounded-xl bg-primary text-primary-foreground">
                  <Bot size={16} />
                </div>
              )}
              <div
                className={`max-w-[90%] rounded-2xl px-4 py-3 ${
                  m.from === 'user' ? 'bg-primary text-primary-foreground' : 'bg-muted'
                }`}
              >
                {m.from === 'ai' ? <><MarkdownAnswer answer={m.text} /><DetectedContext metadata={m.metadata} /><SourceList sources={m.sources} /></> : <div className="text-sm leading-6">{m.text}</div>}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Bot size={16} className="text-primary" />
              CampusGPT is thinking...
            </div>
          )}
        </div>

        <div className="border-t p-4">
          <div className="flex items-end gap-2 rounded-2xl border bg-muted/40 p-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing && e.keyCode !== 229) {
                  e.preventDefault()
                  send()
                }
              }}
              placeholder="Ask a question..."
              rows={2}
              className="min-h-10 flex-1 resize-none bg-transparent px-2 py-2 text-sm outline-none"
            />
            <button
              onClick={() => send()}
              disabled={loading || !input.trim()}
              className="grid size-9 place-items-center rounded-xl bg-primary text-primary-foreground disabled:opacity-40"
              aria-label="Send question"
            >
              <Send size={16} />
            </button>
          </div>

          <div className="mt-3 flex flex-wrap gap-2">
            {suggested.map((q) => (
              <button
                key={q}
                onClick={() => send(q)}
                disabled={loading}
                className="rounded-full border px-3 py-1.5 text-xs text-muted-foreground hover:border-primary hover:text-primary disabled:opacity-50"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

