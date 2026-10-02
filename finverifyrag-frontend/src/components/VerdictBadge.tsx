import type { VerdictType, AtomType, RiskBadge } from '../lib/types';
import { CheckCircle, AlertTriangle, XCircle } from 'lucide-react';

const verdictMeta: Record<VerdictType, { label: string; color: string; bg: string; Icon: typeof CheckCircle }> = {
  SUPPORTED: { label: 'Supported', color: 'var(--ok)',   bg: 'var(--ok-bg)',   Icon: CheckCircle  },
  UNCERTAIN: { label: 'Uncertain', color: 'var(--warn)', bg: 'var(--warn-bg)', Icon: AlertTriangle },
  ABSTAIN:   { label: 'Abstain',   color: 'var(--stop)', bg: 'var(--stop-bg)', Icon: XCircle      },
};

const atomColors: Record<AtomType, string> = {
  RATE:          'var(--atom-rate)',
  THRESHOLD:     'var(--atom-threshold)',
  SECTION:       'var(--atom-section)',
  DATE:          'var(--atom-date)',
  ENTITY:        'var(--atom-entity)',
  APPLICABILITY: 'var(--atom-applicability)',
};

const riskColors: Record<RiskBadge, string> = {
  Low:    'var(--ok)',
  Medium: 'var(--warn)',
  High:   'var(--stop)',
};

export function VerdictBadge({ verdict, size = 'md' }: { verdict: VerdictType; size?: 'sm' | 'md' | 'lg' }) {
  const { label, color, bg, Icon } = verdictMeta[verdict];
  const iconSize = size === 'sm' ? 12 : size === 'lg' ? 20 : 15;
  const fontSize = size === 'sm' ? 11 : size === 'lg' ? 15 : 13;
  return (
    <span
      className="chip"
      style={{ color, background: bg, borderColor: color + '44', fontSize }}
      aria-label={`Verdict: ${label}`}
    >
      <Icon size={iconSize} aria-hidden />
      {label}
    </span>
  );
}

export function AtomTypeChip({ type }: { type: AtomType }) {
  const color = atomColors[type];
  return (
    <span
      className="chip"
      style={{ color, background: color + '22', borderColor: color + '44' }}
    >
      {type}
    </span>
  );
}

export function RiskBadgeChip({ badge }: { badge: RiskBadge }) {
  const color = riskColors[badge];
  return (
    <span className="chip" style={{ color, background: color + '22', borderColor: color + '44' }}>
      {badge} Risk
    </span>
  );
}
