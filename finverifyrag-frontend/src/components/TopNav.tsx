import { NavLink } from 'react-router-dom';
import { Shield, Sun, Moon } from 'lucide-react';
import { useState } from 'react';

const NAV_LINKS = [
  { to: '/',             label: 'Live Console' },
  { to: '/pipeline',     label: 'Pipeline' },
  { to: '/benchmark',    label: 'Benchmark' },
  { to: '/calibration',  label: 'Calibration & Drift' },
  { to: '/test-suite',   label: 'Test Suite' },
  { to: '/architecture', label: 'Architecture' },
];

export default function TopNav() {
  const [light, setLight] = useState(false);

  // Toggle light class on <html>
  const toggleTheme = () => {
    setLight(l => {
      document.documentElement.classList.toggle('light', !l);
      return !l;
    });
  };

  return (
    <header
      style={{
        position: 'sticky',
        top: 0,
        zIndex: 100,
        background: 'var(--ink-900)',
        borderBottom: 'var(--border)',
        backdropFilter: 'blur(12px)',
      }}
    >
      <div
        style={{
          maxWidth: 1400,
          margin: '0 auto',
          padding: '0 24px',
          height: 56,
          display: 'flex',
          alignItems: 'center',
          gap: 32,
        }}
      >
        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 }}>
          <div
            style={{
              width: 32,
              height: 32,
              background: 'var(--brass-500)',
              borderRadius: 8,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Shield size={18} color="var(--ink-950)" aria-hidden />
          </div>
          <span
            style={{
              fontFamily: 'var(--font-display)',
              fontWeight: 700,
              fontSize: 18,
              color: 'var(--paper-100)',
              letterSpacing: '-0.01em',
            }}
          >
            FinVerify<span style={{ color: 'var(--brass-500)' }}>RAG</span>
          </span>
        </div>

        {/* Nav links */}
        <nav style={{ display: 'flex', gap: 4, flex: 1, overflowX: 'auto' }} aria-label="Main navigation">
          {NAV_LINKS.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              style={({ isActive }) => ({
                padding: '6px 14px',
                borderRadius: 8,
                fontSize: 13,
                fontWeight: 500,
                color: isActive ? 'var(--brass-500)' : 'var(--paper-300)',
                background: isActive ? 'var(--brass-500)1A' : 'transparent',
                textDecoration: 'none',
                transition: 'all 150ms',
                whiteSpace: 'nowrap',
                border: isActive ? '1px solid var(--brass-500)44' : '1px solid transparent',
              })}
            >
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Theme toggle */}
        <button
          onClick={toggleTheme}
          className="btn btn-ghost"
          style={{ padding: '6px 10px', gap: 0, flexShrink: 0 }}
          aria-label={light ? 'Switch to dark theme' : 'Switch to light theme'}
        >
          {light ? <Moon size={16} /> : <Sun size={16} />}
        </button>
      </div>
    </header>
  );
}
