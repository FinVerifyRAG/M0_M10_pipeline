import { useState } from 'react';
import { motion } from 'framer-motion';
import { Play, RefreshCw, Filter } from 'lucide-react';
import { DEMO_TESTS } from '../demo/demoData';
import type { TestQuery } from '../lib/types';
import { VerdictBadge } from '../components/VerdictBadge';

const STATUS_COLOR = { pass: 'var(--ok)', fail: 'var(--stop)', not_run: 'var(--ink-500)' } as const;
const STATUS_BG    = { pass: 'var(--ok-bg)', fail: 'var(--stop-bg)', not_run: 'transparent' } as const;

const CATEGORIES = ['All', ...Array.from(new Set(DEMO_TESTS.map(t => t.category.split(' – ')[0].trim())))];

export default function TestSuite() {
  const [filter, setFilter] = useState<string>('All');
  const [statusFilter, setStatusFilter] = useState<'all' | 'pass' | 'fail' | 'not_run'>('all');
  const [tests, setTests] = useState<TestQuery[]>(DEMO_TESTS);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);

  const visible = tests.filter(t => {
    const catMatch = filter === 'All' || t.category.startsWith(filter);
    const statusMatch = statusFilter === 'all' || t.status === statusFilter;
    return catMatch && statusMatch;
  });

  const runAll = async () => {
    setRunning(true);
    setProgress(0);
    const notRun = tests.filter(t => t.status === 'not_run');
    for (let i = 0; i < notRun.length; i++) {
      await new Promise(r => setTimeout(r, 400));
      setProgress(Math.round(((i + 1) / notRun.length) * 100));
      setTests(prev => prev.map(t => t.id === notRun[i].id ? { ...t, status: 'pass', verdict: 'SUPPORTED' } : t));
    }
    setRunning(false);
    setProgress(0);
  };

  const summary = {
    pass:    tests.filter(t => t.status === 'pass').length,
    fail:    tests.filter(t => t.status === 'fail').length,
    not_run: tests.filter(t => t.status === 'not_run').length,
  };

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: '24px 24px 64px' }}>
      <h1 style={{ fontFamily: 'var(--font-display)', fontSize: 32, fontWeight: 700, color: 'var(--paper-100)', marginBottom: 6 }}>
        Test Suite
      </h1>
      <p style={{ color: 'var(--paper-300)', fontSize: 14, marginBottom: 4 }}>
        Edge-case queries covering categories A to K.
      </p>
      <p style={{ fontSize: 11, color: 'var(--ink-500)', fontStyle: 'italic', marginBottom: 24 }}>Sample data — demo mode</p>

      {/* Summary tiles */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 24, flexWrap: 'wrap' }}>
        {(['pass', 'fail', 'not_run'] as const).map(s => (
          <div key={s} style={{ padding: '12px 20px', borderRadius: 10, background: STATUS_BG[s] || 'var(--ink-800)', border: `1px solid ${STATUS_COLOR[s]}44`, textAlign: 'center' }}>
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: 24, fontWeight: 700, color: STATUS_COLOR[s] }}>{summary[s]}</p>
            <p style={{ fontSize: 11, color: 'var(--paper-300)', textTransform: 'capitalize' }}>{s.replace('_', ' ')}</p>
          </div>
        ))}
        {/* Progress bar when running */}
        {running && (
          <div style={{ flex: 1, minWidth: 200, display: 'flex', flexDirection: 'column', gap: 6, justifyContent: 'center' }}>
            <p style={{ fontSize: 12, color: 'var(--paper-300)' }}>Running… {progress}%</p>
            <div style={{ height: 8, background: 'var(--ink-700)', borderRadius: 4 }}>
              <div style={{ width: `${progress}%`, height: '100%', background: 'var(--brass-500)', borderRadius: 4, transition: 'width 300ms' }} />
            </div>
          </div>
        )}
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 16, flexWrap: 'wrap', alignItems: 'center' }}>
        <button id="run-all-tests-btn" onClick={runAll} disabled={running} className="btn btn-brass" style={{ gap: 6 }}>
          <Play size={14} aria-hidden /> Run all
        </button>
        <button className="btn btn-ghost" style={{ gap: 6, fontSize: 12 }} onClick={() => setTests(DEMO_TESTS)}>
          <RefreshCw size={13} aria-hidden /> Reset
        </button>

        <div style={{ display: 'flex', gap: 4, alignItems: 'center', marginLeft: 'auto' }}>
          <Filter size={13} color="var(--ink-500)" aria-hidden />
          {(['all', 'pass', 'fail', 'not_run'] as const).map(s => (
            <button key={s} onClick={() => setStatusFilter(s)}
              style={{
                padding: '3px 10px', borderRadius: 5, fontSize: 11, cursor: 'pointer', border: 'var(--border)',
                background: statusFilter === s ? 'var(--brass-500)1A' : 'transparent',
                color: statusFilter === s ? 'var(--brass-500)' : 'var(--paper-300)',
              }}
            >{s.replace('_', ' ')}</button>
          ))}
        </div>
      </div>

      {/* Category filter */}
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 20 }}>
        {CATEGORIES.map(c => (
          <button key={c} onClick={() => setFilter(c)}
            style={{
              padding: '4px 12px', borderRadius: 6, fontSize: 12, border: 'var(--border)', cursor: 'pointer',
              background: filter === c ? 'var(--brass-500)1A' : 'var(--ink-800)',
              color: filter === c ? 'var(--brass-500)' : 'var(--paper-300)',
              borderColor: filter === c ? 'var(--brass-500)44' : 'var(--ink-700)',
            }}
          >{c}</button>
        ))}
      </div>

      {/* Test rows */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {visible.map((t, i) => (
          <motion.div
            key={t.id}
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.04 }}
            style={{
              padding: '14px 16px', borderRadius: 10, border: `1px solid ${STATUS_COLOR[t.status]}44`,
              background: `${STATUS_BG[t.status] || 'var(--ink-800)'}`,
              display: 'grid', gridTemplateColumns: '1fr auto auto', gap: 12, alignItems: 'start',
            }}
          >
            <div>
              <div style={{ display: 'flex', gap: 8, marginBottom: 6, flexWrap: 'wrap' }}>
                <span className="chip" style={{ fontSize: 10, color: 'var(--paper-300)', background: 'var(--ink-700)' }}>{t.category}</span>
              </div>
              <p style={{ fontSize: 13, color: 'var(--paper-100)', marginBottom: 4 }}>{t.query}</p>
              <p style={{ fontSize: 11, color: 'var(--ink-500)', fontStyle: 'italic' }}>Expected: {t.expected_behavior}</p>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'flex-end' }}>
              {t.verdict && <VerdictBadge verdict={t.verdict} size="sm" />}
            </div>
            <div>
              <span
                className="chip"
                style={{
                  color: STATUS_COLOR[t.status],
                  background: `${STATUS_COLOR[t.status]}22`,
                  borderColor: `${STATUS_COLOR[t.status]}44`,
                  textTransform: 'capitalize',
                }}
              >
                {t.status.replace('_', ' ')}
              </span>
            </div>
          </motion.div>
        ))}
        {visible.length === 0 && (
          <p style={{ color: 'var(--ink-500)', fontSize: 14, textAlign: 'center', padding: '32px 0' }}>No tests match the current filter.</p>
        )}
      </div>
    </div>
  );
}
