import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Search, RotateCcw, ChevronDown } from 'lucide-react';
import type { Atom, QueryResult, Evidence } from '../lib/types';
import { VerdictBadge, AtomTypeChip, RiskBadgeChip } from '../components/VerdictBadge';
import AtomInspector from '../components/AtomInspector';
import { DEMO_RESULT, EXAMPLE_QUERIES } from '../demo/demoData';

type Regulator = 'All' | 'RBI' | 'SEBI';
type GroupBy = 'order' | 'type' | 'verdict';

const VERDICT_ORDER = { SUPPORTED: 0, UNCERTAIN: 1, ABSTAIN: 2 } as const;
const ATOM_TYPE_COLOR: Record<string, string> = {
  RATE: 'var(--atom-rate)', THRESHOLD: 'var(--atom-threshold)',
  SECTION: 'var(--atom-section)', DATE: 'var(--atom-date)',
  ENTITY: 'var(--atom-entity)', APPLICABILITY: 'var(--atom-applicability)',
};
const VERDICT_UNDERLINE: Record<string, string> = {
  SUPPORTED: 'var(--ok)', UNCERTAIN: 'var(--warn)', ABSTAIN: 'var(--stop)',
};

/* Stat chip in the hero strip */
function StatChip({ label, value }: { label: string; value: string }) {
  return (
    <div style={{
      padding: '6px 16px', borderRadius: 8, border: 'var(--border)',
      background: 'var(--ink-800)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2,
    }}>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 18, fontWeight: 500, color: 'var(--brass-500)', fontVariantNumeric: 'tabular-nums' }}>{value}</span>
      <span style={{ fontSize: 11, color: 'var(--paper-300)' }}>{label}</span>
    </div>
  );
}

/* Highlighted answer where atoms are underlined by verdict color */
function HighlightedAnswer({ answer, atoms, onAtomClick, selectedId }: {
  answer: string; atoms: Atom[]; onAtomClick: (a: Atom) => void; selectedId: string | null;
}) {
  // Build segments from spans
  const sorted = [...atoms].sort((a, b) => a.span[0] - b.span[0]);
  const segments: { text: string; atom?: Atom }[] = [];
  let cursor = 0;
  for (const atom of sorted) {
    const [start, end] = atom.span;
    if (start > cursor) segments.push({ text: answer.slice(cursor, start) });
    segments.push({ text: answer.slice(start, end), atom });
    cursor = end;
  }
  if (cursor < answer.length) segments.push({ text: answer.slice(cursor) });

  return (
    <p style={{ fontSize: 15, lineHeight: 1.8, color: 'var(--paper-100)' }}>
      {segments.map((seg, i) =>
        seg.atom ? (
          <span
            key={i}
            onClick={() => onAtomClick(seg.atom!)}
            title={`${seg.atom.type} — ${seg.atom.decision}`}
            style={{
              borderBottom: `2px solid ${VERDICT_UNDERLINE[seg.atom.decision]}`,
              cursor: 'pointer',
              background: selectedId === seg.atom.id ? VERDICT_UNDERLINE[seg.atom.decision] + '22' : 'transparent',
              borderRadius: 2, padding: '0 2px',
              transition: 'background 200ms',
            }}
          >
            {seg.text}
          </span>
        ) : (
          <span key={i}>{seg.text}</span>
        )
      )}
    </p>
  );
}

/* Pipeline progress ring */
function ProgressRing({ stage, total = 10 }: { stage: number; total?: number }) {
  const r = 20, stroke = 3;
  const circ = 2 * Math.PI * r;
  const offset = circ - (stage / total) * circ;
  return (
    <svg width={52} height={52} aria-label={`Stage ${stage} of ${total}`}>
      <circle cx={26} cy={26} r={r} fill="none" stroke="var(--ink-700)" strokeWidth={stroke} />
      <circle cx={26} cy={26} r={r} fill="none" stroke="var(--brass-500)" strokeWidth={stroke}
        strokeDasharray={circ} strokeDashoffset={offset}
        strokeLinecap="round" transform="rotate(-90 26 26)"
        style={{ transition: 'stroke-dashoffset 300ms' }}
      />
      <text x={26} y={30} textAnchor="middle" fill="var(--brass-500)" fontSize={12} fontFamily="var(--font-mono)">{stage}/{total}</text>
    </svg>
  );
}

export default function LiveConsole() {
  const [query, setQuery] = useState('');
  const [regulator, setRegulator] = useState<Regulator>('All');
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState(0);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [selectedAtom, setSelectedAtom] = useState<Atom | null>(null);
  const [groupBy, setGroupBy] = useState<GroupBy>('order');
  const [showRaw, setShowRaw] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [history, setHistory] = useState<string[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Keyboard shortcut: / to focus query
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === '/' && document.activeElement?.tagName !== 'TEXTAREA') {
        e.preventDefault();
        textareaRef.current?.focus();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  const runQuery = async (q = query, reg = regulator) => {
    if (!q.trim()) return;
    setLoading(true);
    setResult(null);
    setSelectedAtom(null);
    setHistory(h => [q, ...h.filter(x => x !== q)].slice(0, 20));
    // Simulate pipeline stages with increasing delay
    for (let i = 1; i <= 10; i++) {
      await new Promise(r => setTimeout(r, 180 + i * 60));
      setStage(i);
    }
    // In a real app: fetch from /api/query with SSE streaming
    await new Promise(r => setTimeout(r, 300));
    setResult({ ...DEMO_RESULT, query: q, regulator_filter: reg });
    setLoading(false);
    setStage(0);
  };

  const counts = result ? {
    supported: result.atoms.filter(a => a.decision === 'SUPPORTED').length,
    uncertain: result.atoms.filter(a => a.decision === 'UNCERTAIN').length,
    abstain:   result.atoms.filter(a => a.decision === 'ABSTAIN').length,
    total:     result.atoms.length,
  } : null;

  const sortedAtoms = result ? [...result.atoms].sort((a, b) => {
    if (groupBy === 'verdict') return VERDICT_ORDER[a.decision] - VERDICT_ORDER[b.decision];
    if (groupBy === 'type')   return a.type.localeCompare(b.type);
    return a.span[0] - b.span[0];
  }) : [];

  const evidenceMap: Evidence[] = result?.retrieved_evidence ?? [];

  return (
    <div style={{ maxWidth: 1400, margin: '0 auto', padding: '24px 24px 48px' }}>
      {/* Hero strip */}
      <div style={{ marginBottom: 32, textAlign: 'center' }}>
        <h1 style={{ fontFamily: 'var(--font-display)', fontSize: 40, fontWeight: 700, color: 'var(--paper-100)', marginBottom: 8, lineHeight: 1.1 }}>
          Every claim earns its verdict.
        </h1>
        <p style={{ color: 'var(--paper-300)', fontSize: 16, marginBottom: 20 }}>
          Risk-calibrated hallucination control for RBI &amp; SEBI regulatory RAG.
        </p>
        <div style={{ display: 'flex', gap: 12, justifyContent: 'center', flexWrap: 'wrap' }}>
          <StatChip label="Atoms verified today" value="2,841" />
          <StatChip label="Abstention rate" value="13.0%" />
          <StatChip label="Selective risk ≤ ε" value="4.3%" />
        </div>
        <p style={{ marginTop: 10, fontSize: 11, color: 'var(--ink-500)', fontStyle: 'italic' }}>Sample data — demo mode</p>
      </div>

      {/* Main 2-col layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '40% 1fr', gap: 20, alignItems: 'start' }}>
        {/* LEFT COLUMN */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* Query bar */}
          <div className="card" style={{ padding: 16 }}>
            <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
              {(['All', 'RBI', 'SEBI'] as Regulator[]).map(r => (
                <button key={r} onClick={() => setRegulator(r)}
                  style={{
                    padding: '4px 14px', borderRadius: 6, fontSize: 12, fontWeight: 600, cursor: 'pointer',
                    background: regulator === r ? 'var(--brass-500)' : 'var(--ink-700)',
                    color: regulator === r ? 'var(--ink-950)' : 'var(--paper-300)',
                    border: 'none', transition: 'all 150ms',
                  }}
                  aria-pressed={regulator === r}
                >{r}</button>
              ))}
            </div>
            <div style={{ position: 'relative' }}>
              <Search size={16} style={{ position: 'absolute', left: 12, top: 14, color: 'var(--ink-500)' }} aria-hidden />
              <textarea
                ref={textareaRef}
                id="query-input"
                rows={3}
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); runQuery(); } }}
                placeholder="Ask an RBI or SEBI question… (press / to focus)"
                aria-label="Regulatory question input"
                style={{
                  width: '100%', background: 'var(--ink-900)', border: 'var(--border)',
                  borderRadius: 8, padding: '10px 12px 10px 36px', color: 'var(--paper-100)',
                  fontFamily: 'var(--font-ui)', fontSize: 14, resize: 'none',
                  outline: 'none', transition: 'border-color 150ms',
                }}
              />
            </div>
            <div style={{ display: 'flex', gap: 8, marginTop: 10, justifyContent: 'space-between', alignItems: 'center' }}>
              <button
                onClick={() => setHistoryOpen(h => !h)}
                className="btn btn-ghost"
                style={{ padding: '6px 10px', fontSize: 12 }}
                aria-expanded={historyOpen}
              >
                History <ChevronDown size={13} style={{ transform: historyOpen ? 'rotate(180deg)' : '', transition: 'transform 200ms' }} />
              </button>
              <button
                id="run-query-btn"
                onClick={() => runQuery()}
                disabled={loading || !query.trim()}
                className="btn btn-brass"
                style={{ gap: 6, opacity: loading || !query.trim() ? 0.6 : 1 }}
              >
                {loading ? <ProgressRing stage={stage} /> : <><Send size={15} aria-hidden /> Run</>}
              </button>
            </div>

            {/* History drawer */}
            <AnimatePresence>
              {historyOpen && history.length > 0 && (
                <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} style={{ overflow: 'hidden' }}>
                  <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {history.map((h, i) => (
                      <button key={i} onClick={() => { setQuery(h); runQuery(h); }}
                        style={{ textAlign: 'left', background: 'var(--ink-900)', border: 'var(--border)', borderRadius: 6, padding: '7px 12px', fontSize: 12, color: 'var(--paper-300)', cursor: 'pointer' }}>
                        {h}
                      </button>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Example query chips */}
          <div>
            <p style={{ fontSize: 11, color: 'var(--ink-500)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Example queries</p>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {EXAMPLE_QUERIES.map((eq, i) => (
                <button key={i}
                  onClick={() => { setQuery(eq.query); setRegulator(eq.regulator); }}
                  style={{
                    padding: '5px 12px', borderRadius: 6, fontSize: 12, border: 'var(--border)',
                    background: 'var(--ink-800)', color: 'var(--paper-300)', cursor: 'pointer',
                    transition: 'all 150ms',
                  }}
                  onMouseOver={e => { (e.currentTarget as HTMLElement).style.borderColor = 'var(--brass-500)'; (e.currentTarget as HTMLElement).style.color = 'var(--brass-300)'; }}
                  onMouseOut={e => { (e.currentTarget as HTMLElement).style.borderColor = 'var(--ink-700)'; (e.currentTarget as HTMLElement).style.color = 'var(--paper-300)'; }}
                >
                  {eq.label}
                </button>
              ))}
            </div>
          </div>

          {/* Answer panel */}
          {result && (
            <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
              {/* Verdict banner */}
              <div style={{
                padding: '12px 16px', borderRadius: 10, marginBottom: 12,
                background: result.final_decision === 'SUPPORTED' ? 'var(--ok-bg)' :
                  result.final_decision === 'UNCERTAIN' ? 'var(--warn-bg)' : 'var(--stop-bg)',
                border: `1px solid ${result.final_decision === 'SUPPORTED' ? 'var(--ok)' : result.final_decision === 'UNCERTAIN' ? 'var(--warn)' : 'var(--stop)'}44`,
                display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap',
              }}>
                <VerdictBadge verdict={result.final_decision} size="md" />
                <RiskBadgeChip badge={result.risk_badge} />
                {result.abstain_reason && (
                  <p style={{ fontSize: 12, color: 'var(--paper-300)', flex: 1 }}>{result.abstain_reason}</p>
                )}
                {result.final_decision === 'UNCERTAIN' && (
                  <p style={{ fontSize: 12, color: 'var(--paper-300)', flex: 1 }}>Some claims need a closer check. A stronger model reviewed them.</p>
                )}
              </div>

              {/* Answer card */}
              <div className="card">
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
                  <button onClick={() => setShowRaw(false)}
                    style={{ fontSize: 12, fontWeight: 600, cursor: 'pointer', color: !showRaw ? 'var(--brass-500)' : 'var(--paper-300)', background: 'none', border: 'none' }}>
                    Verified answer
                  </button>
                  <span style={{ color: 'var(--ink-500)' }}>⇄</span>
                  <button onClick={() => setShowRaw(true)}
                    style={{ fontSize: 12, fontWeight: 600, cursor: 'pointer', color: showRaw ? 'var(--brass-500)' : 'var(--paper-300)', background: 'none', border: 'none' }}>
                    Raw answer
                  </button>
                </div>
                {showRaw ? (
                  <p style={{ fontSize: 14, color: 'var(--paper-300)', lineHeight: 1.8 }}>{result.raw_answer}</p>
                ) : (
                  <HighlightedAnswer
                    answer={result.raw_answer}
                    atoms={result.atoms}
                    onAtomClick={a => setSelectedAtom(a)}
                    selectedId={selectedAtom?.id ?? null}
                  />
                )}
              </div>

              {/* Atom splitting view */}
              <div className="card" style={{ marginTop: 12 }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12, flexWrap: 'wrap', gap: 8 }}>
                  <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--paper-100)' }}>
                    {counts!.total} atoms: {counts!.supported} supported · {counts!.uncertain} uncertain · {counts!.abstain} abstained
                  </span>
                  <div style={{ display: 'flex', gap: 4 }}>
                    {(['order', 'type', 'verdict'] as GroupBy[]).map(g => (
                      <button key={g} onClick={() => setGroupBy(g)}
                        style={{
                          padding: '3px 10px', borderRadius: 5, fontSize: 11, cursor: 'pointer', border: 'var(--border)',
                          background: groupBy === g ? 'var(--brass-500)1A' : 'transparent',
                          color: groupBy === g ? 'var(--brass-500)' : 'var(--paper-300)',
                        }}
                      >{g}</button>
                    ))}
                  </div>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <AnimatePresence mode="wait">
                    {sortedAtoms.map((atom, i) => (
                      <motion.div
                        key={atom.id}
                        initial={{ opacity: 0, x: -8 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.06, duration: 0.25 }}
                        onClick={() => setSelectedAtom(atom)}
                        style={{
                          display: 'flex', alignItems: 'flex-start', gap: 10, padding: '10px 14px',
                          borderRadius: 8, cursor: 'pointer', border: '1px solid',
                          borderColor: selectedAtom?.id === atom.id ? ATOM_TYPE_COLOR[atom.type] + '88' : 'var(--ink-700)',
                          background: selectedAtom?.id === atom.id ? ATOM_TYPE_COLOR[atom.type] + '11' : 'var(--ink-900)',
                          transition: 'all 150ms',
                        }}
                      >
                        <AtomTypeChip type={atom.type} />
                        <p style={{ flex: 1, fontSize: 12, color: 'var(--paper-300)', lineHeight: 1.5 }}>{atom.text}</p>
                        <VerdictBadge verdict={atom.decision} size="sm" />
                      </motion.div>
                    ))}
                  </AnimatePresence>
                </div>
              </div>

              {/* Final output card */}
              <div className="card" style={{ marginTop: 12 }}>
                <p style={{ fontSize: 12, color: 'var(--ink-500)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 10 }}>Final output</p>
                <p style={{ fontSize: 14, lineHeight: 1.7, color: 'var(--paper-100)', marginBottom: 14 }}>{result.final_answer}</p>

                {/* Citations */}
                {result.citations.length > 0 && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 12 }}>
                    {result.citations.map(c => (
                      <div key={c.chunk_id} style={{ padding: '8px 12px', borderRadius: 8, background: 'var(--ink-900)', border: 'var(--border)', display: 'flex', gap: 10, alignItems: 'center' }}>
                        <span className="chip" style={{ color: 'var(--brass-500)', background: 'var(--brass-500)1A', borderColor: 'var(--brass-500)44' }}>{c.source}</span>
                        <div>
                          <p style={{ fontSize: 12, color: 'var(--paper-100)', fontWeight: 500 }}>{c.document}</p>
                          <p style={{ fontSize: 11, color: 'var(--paper-300)' }}>{c.section} · {c.effective_date}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Actions */}
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <button className="btn btn-ghost" style={{ fontSize: 12 }} onClick={() => navigator.clipboard.writeText(result.final_answer)}>Copy answer</button>
                  <button className="btn btn-ghost" style={{ fontSize: 12 }}>Export JSON</button>
                  <button className="btn btn-ghost" style={{ fontSize: 12 }}>Export PDF</button>
                  <button className="btn btn-ghost" style={{ fontSize: 12 }} onClick={() => { setResult(null); setQuery(''); setSelectedAtom(null); }}>
                    <RotateCcw size={12} /> Reset
                  </button>
                </div>

                {/* Latency */}
                <div style={{ marginTop: 12, display: 'flex', gap: 16, fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--ink-500)' }}>
                  <span>Retrieval: {result.latency_ms.retrieval}ms</span>
                  <span>Generation: {result.latency_ms.generation}ms</span>
                  <span>Verification: {result.latency_ms.verification}ms</span>
                  <span style={{ color: 'var(--paper-300)' }}>Total: {result.latency_ms.total}ms</span>
                </div>
              </div>
            </motion.div>
          )}

          {/* Loading skeleton */}
          {loading && (
            <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center', padding: 32 }}>
              <ProgressRing stage={stage} />
              <p style={{ fontSize: 13, color: 'var(--paper-300)' }}>Running pipeline stage {stage}/10…</p>
              {[80, 60, 70].map((w, i) => (
                <div key={i} style={{ height: 14, width: `${w}%`, borderRadius: 4, background: 'var(--ink-700)', animation: 'pulse 1.4s ease-in-out infinite', animationDelay: `${i * 0.2}s` }} />
              ))}
            </div>
          )}
        </div>

        {/* RIGHT COLUMN — Atom Inspector */}
        <div style={{ position: 'sticky', top: 72, height: 'calc(100vh - 90px)', overflow: 'hidden' }}>
          <AtomInspector atom={selectedAtom} evidences={evidenceMap} />
        </div>
      </div>

      <style>{`
        @keyframes pulse { 0%,100%{opacity:.4} 50%{opacity:.9} }
        textarea:focus { border-color: var(--brass-500) !important; }
      `}</style>
    </div>
  );
}
