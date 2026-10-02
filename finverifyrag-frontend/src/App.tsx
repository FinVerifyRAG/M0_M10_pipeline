import { BrowserRouter, Routes, Route } from 'react-router-dom';
import TopNav from './components/TopNav';
import LiveConsole     from './pages/LiveConsole';
import Pipeline        from './pages/Pipeline';
import Benchmark       from './pages/Benchmark';
import CalibrationDrift from './pages/CalibrationDrift';
import TestSuite       from './pages/TestSuite';
import Architecture    from './pages/Architecture';

function Footer() {
  return (
    <footer style={{
      borderTop: 'var(--border)',
      padding: '20px 24px',
      textAlign: 'center',
      fontSize: 11,
      color: 'var(--ink-500)',
      fontStyle: 'italic',
    }}>
      For research and compliance assistance. Not legal advice. Always confirm against the official RBI and SEBI publications.
    </footer>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
        <TopNav />
        <main style={{ flex: 1 }}>
          <Routes>
            <Route path="/"             element={<LiveConsole />} />
            <Route path="/pipeline"     element={<Pipeline />} />
            <Route path="/benchmark"    element={<Benchmark />} />
            <Route path="/calibration"  element={<CalibrationDrift />} />
            <Route path="/test-suite"   element={<TestSuite />} />
            <Route path="/architecture" element={<Architecture />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </BrowserRouter>
  );
}
