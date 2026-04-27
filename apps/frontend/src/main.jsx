import React, { useEffect, useMemo, useState, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  BarChart3,
  CheckCircle2,
  ChevronRight,
  Clock3,
  ClipboardCheck,
  Database,
  FileText,
  Gauge,
  GitBranch,
  Loader2,
  MessageSquareText,
  RefreshCw,
  Rocket,
  Send,
  Server,
  ShieldCheck,
  Sparkles,
  Star,
  ThumbsDown,
  ThumbsUp,
  Minus,
  TrendingUp,
  Workflow,
  XCircle,
  X,
  Zap,
  Brain,
  Target,
  Layers,
  Map,
} from 'lucide-react';
import './styles.css';

const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const MAX_LEN = 5000;

const EXAMPLES = [
  { label: 'Positive', emoji: '😊', sentiment: 'positive', text: 'Excellent quality and the delivery was faster than expected. I would buy this again.' },
  { label: 'Neutral', emoji: '😐', sentiment: 'neutral', text: 'The product is okay and works as described, but there is nothing special about it.' },
  { label: 'Negative', emoji: '😞', sentiment: 'negative', text: 'The package arrived damaged and the product stopped working after two days.' },
];

const SENTIMENT_MAP = {
  positive: { label: 'Positive', icon: CheckCircle2, tone: 'positive' },
  neutral: { label: 'Neutral', icon: Gauge, tone: 'neutral' },
  negative: { label: 'Negative', icon: XCircle, tone: 'negative' },
};

/* ═══════════════════════════════════════════════════════
   PRODUCT TOUR DEFINITION
   ═══════════════════════════════════════════════════════ */
const TOUR_STEPS = [
  {
    target: null,
    title: 'Welcome to SentinelAI 🚀',
    body: 'Let\'s take a quick 30-second tour of the sentiment analyzer. You can skip anytime.',
    placement: 'center',
  },
  {
    target: '[data-tour="nav"]',
    title: 'Navigation',
    body: 'Switch between the AI Analyzer for predictions and the MLOps dashboard for pipeline monitoring.',
    placement: 'bottom',
    padding: 6,
  },
  {
    target: '[data-tour="textarea"]',
    title: 'Review Input',
    body: 'Paste any product review here — from Amazon, Flipkart, or any e-commerce platform. Supports up to 5,000 characters.',
    placement: 'bottom',
    padding: 8,
  },
  {
    target: '[data-tour="examples"]',
    title: 'Quick Examples',
    body: 'Don\'t have a review handy? Click any of these pre-loaded examples to try it instantly.',
    placement: 'bottom',
    padding: 6,
  },
  {
    target: '[data-tour="analyze-btn"]',
    title: 'Run Analysis',
    body: 'Hit this button to send the review to the AI model. You\'ll get sentiment, confidence, explainable tokens, and latency metrics.',
    placement: 'top',
    padding: 10,
  },
  {
    target: '[data-tour="results-area"]',
    title: 'Results Appear Here',
    body: 'After analysis, your results show up here — a verdict card with confidence ring, probability breakdown, influential tokens, and model metadata.',
    placement: 'top',
    padding: 10,
  },
  {
    target: '[data-tour="guide-btn"]',
    title: 'User Guide',
    body: 'Need help later? The guide panel slides out with step-by-step instructions for every feature.',
    placement: 'bottom-end',
    padding: 6,
  },
  {
    target: null,
    title: 'You\'re all set! ✨',
    body: 'Start by pasting a review or picking an example. You can restart this tour anytime from the top bar.',
    placement: 'center',
  },
];

/* ═══════════════════════════════════════════════════════
   APP SHELL
   ═══════════════════════════════════════════════════════ */
function App() {
  const [screen, setScreen] = useState('predict');
  const [manualOpen, setManualOpen] = useState(false);
  const [tourActive, setTourActive] = useState(() => {
    return !localStorage.getItem('sentinel-tour-done');
  });

  function startTour() {
    setScreen('predict');
    setTourActive(true);
  }

  function endTour() {
    localStorage.setItem('sentinel-tour-done', '1');
    setTourActive(false);
  }

  return (
    <div className="app-shell">
      {/* Animated Background */}
      <div className="bg-scene" aria-hidden="true">
        <div className="blob blob-1" />
        <div className="blob blob-2" />
        <div className="blob blob-3" />
        <div className="blob blob-4" />
        <div className="bg-noise" />
      </div>

      {/* Top Bar */}
      <header className="topbar">
        <div className="topbar-left">
          <div className="brand-mark" aria-hidden="true"><Sparkles size={20} /></div>
          <h1 className="brand-name">SentinelAI</h1>
        </div>
        <nav className="topbar-nav" data-tour="nav" aria-label="Main">
          <button className={screen === 'predict' ? 'nav-pill active' : 'nav-pill'} onClick={() => setScreen('predict')}>
            <Brain size={16} /><span>Analyzer</span>
          </button>
          <button className={screen === 'ops' ? 'nav-pill active' : 'nav-pill'} onClick={() => setScreen('ops')}>
            <Activity size={16} /><span>MLOps</span>
          </button>
        </nav>
        <div className="topbar-right">
          <button className="topbar-btn tour-trigger" onClick={startTour} title="Product Tour">
            <Map size={16} /><span>Tour</span>
          </button>
          <button className="topbar-btn" data-tour="guide-btn" onClick={() => setManualOpen(true)}>
            <FileText size={16} /><span>Guide</span>
          </button>
        </div>
      </header>

      {screen === 'predict' ? <AnalyzerScreen /> : <OpsScreen />}
      {manualOpen && <GuidePanel onClose={() => setManualOpen(false)} />}
      {tourActive && <ProductTour onComplete={endTour} />}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════
   PRODUCT TOUR COMPONENT
   ═══════════════════════════════════════════════════════ */
function ProductTour({ onComplete }) {
  const [step, setStep] = useState(0);
  const [rect, setRect] = useState(null);
  const [transitioning, setTransitioning] = useState(false);
  const tooltipRef = useRef(null);

  const current = TOUR_STEPS[step];
  const total = TOUR_STEPS.length;
  const isFirst = step === 0;
  const isLast = step === total - 1;
  const isCentered = current.placement === 'center';

  // Measure the target element
  useEffect(() => {
    if (!current.target) {
      setRect(null);
      return;
    }

    const el = document.querySelector(current.target);
    if (!el) { setRect(null); return; }

    const pad = current.padding || 8;

    function measure() {
      const r = el.getBoundingClientRect();
      setRect({
        top: r.top - pad,
        left: r.left - pad,
        width: r.width + pad * 2,
        height: r.height + pad * 2,
      });
    }

    // Slight delay to allow DOM to settle
    const timer = setTimeout(measure, 50);
    window.addEventListener('resize', measure);
    window.addEventListener('scroll', measure, true);

    return () => {
      clearTimeout(timer);
      window.removeEventListener('resize', measure);
      window.removeEventListener('scroll', measure, true);
    };
  }, [step, current.target, current.padding]);

  // Scroll target into view if needed
  useEffect(() => {
    if (!current.target) return;
    const el = document.querySelector(current.target);
    if (!el) return;

    const r = el.getBoundingClientRect();
    const inView = r.top >= 60 && r.bottom <= window.innerHeight - 40;
    if (!inView) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [step, current.target]);

  function goTo(nextStep) {
    setTransitioning(true);
    setTimeout(() => {
      setStep(nextStep);
      setTransitioning(false);
    }, 200);
  }

  function next() {
    if (isLast) onComplete();
    else goTo(step + 1);
  }

  function back() {
    if (!isFirst) goTo(step - 1);
  }

  // Calculate tooltip position
  const tooltipStyle = useMemo(() => {
    if (isCentered || !rect) return {};

    const gap = 16;
    const tooltipW = 360;
    const tooltipEstH = 200;
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    let top, left;
    const placement = current.placement || 'bottom';

    // Vertical
    if (placement.startsWith('top')) {
      top = rect.top - tooltipEstH - gap;
      if (top < 10) top = rect.top + rect.height + gap; // flip to bottom
    } else {
      top = rect.top + rect.height + gap;
      if (top + tooltipEstH > vh - 10) top = rect.top - tooltipEstH - gap; // flip to top
    }

    // Horizontal
    if (placement.endsWith('-end')) {
      left = rect.left + rect.width - tooltipW;
    } else if (placement.endsWith('-start')) {
      left = rect.left;
    } else {
      left = rect.left + rect.width / 2 - tooltipW / 2;
    }

    // Clamp
    if (left < 12) left = 12;
    if (left + tooltipW > vw - 12) left = vw - tooltipW - 12;
    if (top < 10) top = 10;

    return { top: `${top}px`, left: `${left}px`, width: `${tooltipW}px` };
  }, [rect, isCentered, current.placement]);

  // Keyboard controls
  useEffect(() => {
    function onKey(e) {
      if (e.key === 'Escape') onComplete();
      if (e.key === 'ArrowRight' || e.key === 'Enter') next();
      if (e.key === 'ArrowLeft') back();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  return (
    <div className={`tour-overlay ${transitioning ? 'tour-transitioning' : ''}`}>
      {/* Full-screen dim for centered steps (welcome/completion) */}
      {isCentered && <div className="tour-backdrop" />}

      {/* Spotlight cutout (only for targeted steps) */}
      {rect && !isCentered && (
        <div
          className="tour-spotlight"
          style={{
            top: `${rect.top}px`,
            left: `${rect.left}px`,
            width: `${rect.width}px`,
            height: `${rect.height}px`,
          }}
        />
      )}

      {/* Tooltip */}
      <div
        className={`tour-tooltip ${isCentered ? 'tour-tooltip-center' : ''}`}
        style={isCentered ? {} : tooltipStyle}
        ref={tooltipRef}
      >
        {/* Progress bar */}
        <div className="tour-progress">
          <div className="tour-progress-fill" style={{ width: `${((step + 1) / total) * 100}%` }} />
        </div>

        {/* Icon for centered steps */}
        {isCentered && (
          <div className="tour-center-icon">
            {isLast ? <CheckCircle2 size={32} /> : <Rocket size={32} />}
          </div>
        )}

        <div className="tour-body">
          <h3 className="tour-title">{current.title}</h3>
          <p className="tour-desc">{current.body}</p>
        </div>

        {/* Controls */}
        <div className="tour-controls">
          <div className="tour-dots">
            {TOUR_STEPS.map((_, i) => (
              <button
                key={i}
                className={`tour-dot ${i === step ? 'active' : ''} ${i < step ? 'done' : ''}`}
                onClick={() => goTo(i)}
                aria-label={`Go to step ${i + 1}`}
              />
            ))}
          </div>
          <div className="tour-btns">
            {!isFirst && (
              <button className="tour-btn-back" onClick={back}>
                <ArrowLeft size={15} /> Back
              </button>
            )}
            {isFirst && (
              <button className="tour-btn-skip" onClick={onComplete}>
                Skip tour
              </button>
            )}
            <button className="tour-btn-next" onClick={next}>
              {isLast ? 'Get Started' : 'Next'}
              {!isLast && <ArrowRight size={15} />}
            </button>
          </div>
        </div>

        {/* Step counter */}
        <span className="tour-counter">{step + 1} of {total}</span>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════
   ANALYZER SCREEN
   ═══════════════════════════════════════════════════════ */
function AnalyzerScreen() {
  const [review, setReview] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [feedbackStatus, setFeedbackStatus] = useState('');
  const [feedbackLoading, setFeedbackLoading] = useState(false);
  const [selectedFeedback, setSelectedFeedback] = useState('');
  const resultRef = useRef(null);

  const trimmed = review.trim();
  const tooLong = review.length > MAX_LEN;
  const canAnalyze = Boolean(trimmed) && !tooLong && !loading;

  async function analyze() {
    setError(''); setFeedbackStatus(''); setSelectedFeedback('');
    if (!trimmed) { setError('Enter a product review first.'); return; }
    if (tooLong) { setError(`Review must be ${MAX_LEN} characters or fewer.`); return; }
    setLoading(true);
    try {
      const r = await fetch(`${API}/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ review_text: trimmed }) });
      const data = await safeJson(r);
      if (!r.ok) throw new Error(data?.detail || `API ${r.status}`);
      setResult(data);
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
    } catch (e) { setError(e.message || 'API unavailable. Start the FastAPI service.'); }
    finally { setLoading(false); }
  }

  async function sendFeedback(label) {
    if (!result || feedbackLoading) return;
    setSelectedFeedback(label); setFeedbackLoading(true); setFeedbackStatus('');
    try {
      const r = await fetch(`${API}/feedback`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ review_text: trimmed, predicted_sentiment: result.sentiment, actual_sentiment: label, source: 'frontend-demo' }) });
      setFeedbackStatus(r.ok ? 'Feedback saved ✓' : 'Could not save feedback.');
    } catch { setFeedbackStatus('Could not save feedback.'); }
    finally { setFeedbackLoading(false); }
  }

  function reset() { setReview(''); setResult(null); setError(''); setFeedbackStatus(''); setSelectedFeedback(''); }

  return (
    <main className="analyzer-main">
      <section className="analyzer-hero">
        <div className="hero-badge"><Zap size={14} /> AI-Powered Analysis</div>
        <h2>What do your customers <em>really</em> think?</h2>
        <p>Paste a product review below. Get sentiment, confidence, explainable tokens, and latency — in milliseconds.</p>
        <div className="hero-stats">
          <StatChip icon={<Zap size={14} />} value="< 200ms" />
          <StatChip icon={<Target size={14} />} value="F1 ≥ 0.75" />
          <StatChip icon={<Layers size={14} />} value="3 classes" />
        </div>
      </section>

      <section className="input-card">
        <div className="input-card-header">
          <span className="step-num">1</span>
          <div>
            <h3>Enter a product review</h3>
            <p className="input-sub">Or pick an example below</p>
          </div>
          {review && <button className="btn-icon" onClick={reset} title="Clear"><RefreshCw size={15} /></button>}
        </div>

        <div className="textarea-wrap" data-tour="textarea">
          <textarea
            value={review}
            maxLength={MAX_LEN + 1}
            onChange={e => setReview(e.target.value)}
            placeholder="The product quality was amazing and shipping was fast..."
            aria-label="Product review text"
            rows={5}
          />
          <span className={`char-count ${tooLong ? 'over' : ''}`}>{review.length.toLocaleString()}/{MAX_LEN.toLocaleString()}</span>
        </div>

        <div className="examples-row" data-tour="examples">
          {EXAMPLES.map(ex => (
            <button key={ex.sentiment} className={`example-chip ${review === ex.text ? 'active ' + ex.sentiment : ''}`} onClick={() => { setReview(ex.text); setError(''); }}>
              <span className="ex-emoji">{ex.emoji}</span>{ex.label}
            </button>
          ))}
        </div>

        <div className="analyze-row">
          <span className="word-count">{wordCount(review)} words</span>
          <button className="btn-cta" data-tour="analyze-btn" onClick={analyze} disabled={!canAnalyze}>
            {loading ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
            {loading ? 'Analyzing…' : 'Analyze Sentiment'}
          </button>
        </div>

        {error && <div className="alert-err" role="alert"><AlertTriangle size={16} /><span>{error}</span></div>}
      </section>

      {result && (
        <section className="results-zone" ref={resultRef}>
          <div className="results-head"><span className="step-num">2</span><h3>Analysis Results</h3></div>
          <VerdictCard result={result} />
          <div className="details-duo">
            <ConfidenceCard probabilities={result.class_probabilities} />
            <TokensCard tokens={result.explanation} />
          </div>
          <MetaRow result={result} />
          <FeedbackRow onFeedback={sendFeedback} feedbackStatus={feedbackStatus} feedbackLoading={feedbackLoading} selectedFeedback={selectedFeedback} />
        </section>
      )}

      {!result && !loading && (
        <div className="empty-hint" data-tour="results-area">
          <div className="empty-icon"><Gauge size={28} /></div>
          <p>Results will appear here after analysis</p>
        </div>
      )}
    </main>
  );
}

function VerdictCard({ result }) {
  const s = SENTIMENT_MAP[result.sentiment] || SENTIMENT_MAP.neutral;
  const Icon = s.icon;
  const pct = Math.round(result.confidence * 100);
  return (
    <div className={`verdict ${s.tone}`}>
      <div className="verdict-left">
        <div className="verdict-icon"><Icon size={26} /></div>
        <div><span className="verdict-label">Predicted Sentiment</span><strong className="verdict-val">{s.label}</strong></div>
      </div>
      <div className="verdict-right">
        <div className="donut" style={{ '--pct': `${pct}%` }}><span>{pct}<small>%</small></span></div>
        <span className="donut-label">Confidence</span>
      </div>
    </div>
  );
}

function ConfidenceCard({ probabilities }) {
  const rows = useMemo(() => Object.entries(probabilities || {}).sort((a, b) => b[1] - a[1]), [probabilities]);
  return (
    <div className="detail-card">
      <div className="dc-head"><BarChart3 size={16} /><h4>Confidence Breakdown</h4></div>
      <div className="prob-list">
        {rows.map(([label, value]) => (
          <div className="prob-row" key={label}>
            <span className="prob-lbl">{titleCase(label)}</span>
            <div className="prob-track"><div className={`prob-fill ${label}`} style={{ width: `${Math.round(value * 100)}%` }} /></div>
            <span className="prob-pct">{Math.round(value * 100)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function TokensCard({ tokens }) {
  const items = tokens || [];
  return (
    <div className="detail-card">
      <div className="dc-head"><Sparkles size={16} /><h4>Influential Words</h4><span className="dc-count">{items.length}</span></div>
      <div className="token-cloud">
        {items.length ? items.map((t, i) => (
          <span className="token" key={`${t.token}-${i}`}>{t.token}<small>{fmtWeight(t.weight)}</small></span>
        )) : <span className="token-none">No notable tokens</span>}
      </div>
    </div>
  );
}

function MetaRow({ result }) {
  return (
    <div className="meta-row">
      <MiniMeta label="Latency" value={`${Number(result.latency_ms || 0).toFixed(1)} ms`} />
      <MiniMeta label="Model" value={result.model_version || 'unknown'} />
      <MiniMeta label="MLflow" value={shortText(result.mlflow_run_id, 10)} />
      <MiniMeta label="API" value={API.replace(/^https?:\/\//, '')} />
    </div>
  );
}
function MiniMeta({ label, value }) {
  return <div className="mini-meta"><span>{label}</span><strong>{value}</strong></div>;
}

function FeedbackRow({ onFeedback, feedbackStatus, feedbackLoading, selectedFeedback }) {
  return (
    <div className="fb-row">
      <span className="fb-q">Was this correct?</span>
      <div className="fb-btns">
        {['positive', 'neutral', 'negative'].map(l => {
          const icons = { positive: ThumbsUp, neutral: Minus, negative: ThumbsDown };
          const I = icons[l];
          return <button key={l} className={`fb-b ${selectedFeedback === l ? 'sel ' + l : ''}`} disabled={feedbackLoading} onClick={() => onFeedback(l)} title={titleCase(l)}><I size={15} /><span>{titleCase(l)}</span></button>;
        })}
      </div>
      {feedbackStatus && <small className="fb-msg">{feedbackStatus}</small>}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════
   OPS SCREEN — Premium Dashboard
   ═══════════════════════════════════════════════════════ */
function OpsScreen() {
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  async function load(refresh = false) {
    setError(''); setLoading(true);
    try {
      if (refresh) await fetch(`${API}/monitoring/refresh`, { method: 'POST' });
      const r = await fetch(`${API}/metrics-summary`);
      const data = await safeJson(r);
      if (!r.ok) throw new Error(data?.detail || `API ${r.status}`);
      setSummary(data);
    } catch (e) { setError(e.message || 'Cannot load MLOps summary.'); }
    finally { setLoading(false); }
  }

  useEffect(() => { load(); }, []);

  const links = summary?.links || { airflow: 'http://localhost:8080', mlflow: 'http://localhost:5001', prometheus: 'http://localhost:9091', grafana: 'http://localhost:3001', alertmanager: 'http://localhost:19093' };
  const metrics = summary?.evaluation?.metrics || summary?.evaluation?.test || {};
  const drift = summary?.drift || {};
  const model = summary?.model?.metadata || {};
  const ingestion = summary?.ingestion || {};
  const preprocessing = summary?.preprocessing || {};
  const comparison = summary?.model_comparison || {};
  const performance = summary?.pipeline_performance || {};
  const ps = summary?.pipeline_summary || {};
  const validationWarnings = summary?.validation?.warnings || [];
  const rawRows = ingestion.rows || ps.raw_rows;
  const processedRows = preprocessing.final_rows || ps.processed_rows;
  const rejectedRows = preprocessing.rejected_rows || ps.rejected_rows || 0;
  const rejectedRatio = rawRows ? rejectedRows / rawRows : 0;
  const accepted = summary?.evaluation?.accepted ?? ps.accepted;
  const fallbackMode = summary?.model?.fallback_mode;
  const ready = summary?.api?.ready;
  const f1 = metrics.macro_f1 || ps.test_macro_f1;
  const f1Pct = typeof f1 === 'number' ? Math.round(f1 * 100) : 0;

  const stages = [
    { name: 'Ingest', key: 'ingest_data', report: 'ingestion_report.json', icon: Database },
    { name: 'Validate', key: 'validate_raw_data', report: 'data_validation.json', icon: ShieldCheck },
    { name: 'EDA', key: 'run_eda', report: 'eda_report.json', icon: BarChart3 },
    { name: 'Preprocess', key: 'preprocess_data', report: 'preprocessing_report.json', icon: Layers },
    { name: 'Baseline', key: 'compute_drift_baseline', report: 'feature_baseline_report.json', icon: Target },
    { name: 'Compare', key: 'train_and_compare_models', report: 'model_comparison.json', icon: GitBranch },
    { name: 'Evaluate', key: 'evaluate_selected_model', report: 'evaluation.json', icon: CheckCircle2 },
    { name: 'Drift', key: 'run_batch_drift_check', report: 'drift_report.json', icon: AlertTriangle },
    { name: 'Publish', key: 'publish_pipeline_report', report: 'pipeline_report.json', icon: Send },
  ];
  const stageStates = stages.map((stage) => getPipelineStageState(stage, summary));
  const failedStages = stageStates.filter((s) => s.state === 'failed').length;
  const warningStages = stageStates.filter((s) => s.state === 'warning').length;
  const finishedStages = stageStates.filter((s) => ['success', 'warning', 'failed'].includes(s.state)).length;
  const pipelineTone = failedStages ? 'failed' : warningStages ? 'warning' : finishedStages === stages.length ? 'success' : 'pending';
  const pipelineEvents = buildPipelineEvents({ error, summary, validationWarnings, rejectedRows, rejectedRatio, accepted, drift, performance, ps, stageStates });
  const linkIcons = { airflow: Workflow, mlflow: Brain, prometheus: Activity, grafana: BarChart3, alertmanager: AlertTriangle };
  const healthy = summary?.api?.healthy;

  return (
    <main className="ops-main">
      {/* ━━━ STATUS BANNER ━━━ */}
      <section className="status-banner">
        <div className="sb-gradient" />
        <div className="sb-content">
          <div className="sb-left">
            <div className={`sb-health-dot ${healthy ? 'live' : ''}`} />
            <div>
              <span className="sb-health-label">{healthy ? 'All Systems Operational' : 'Status Unknown'}</span>
              <h2 className="sb-title">MLOps Command Center</h2>
            </div>
          </div>
          <div className="sb-metrics">
            <div className="sb-hero-metric">
              <div className="f1-ring" style={{ '--f1': f1Pct }}>
                <svg viewBox="0 0 80 80">
                  <circle className="f1-track" cx="40" cy="40" r="34" />
                  <circle className="f1-fill" cx="40" cy="40" r="34" strokeDasharray={`${f1Pct * 2.136} 999`} />
                </svg>
                <span className="f1-val">{typeof f1 === 'number' ? f1.toFixed(2) : '—'}</span>
              </div>
              <div className="sb-hero-label">
                <strong>Macro F1</strong>
                <small>Primary metric</small>
              </div>
            </div>
            <div className="sb-pills">
              <span className={`sb-pill ${ready ? 'positive' : 'warning'}`}><ShieldCheck size={12} /> {ready ? 'Ready' : 'Review'}</span>
              <span className={`sb-pill ${drift.drift_detected ? 'warning' : 'positive'}`}><Activity size={12} /> {drift.drift_detected ? 'Drift' : 'Stable'}</span>
              <span className={`sb-pill ${accepted ? 'positive' : 'warning'}`}><Target size={12} /> {accepted ? 'Accepted' : 'Pending'}</span>
            </div>
          </div>
          <button className="sb-refresh" onClick={() => load(true)} disabled={loading}>
            {loading ? <Loader2 className="spin" size={16} /> : <RefreshCw size={16} />}
          </button>
        </div>
      </section>

      {error && <div className="alert-err" role="alert"><AlertTriangle size={16} /><span>{error}</span></div>}

      {/* ━━━ PIPELINE TIMELINE ━━━ */}
      <section className="pipe-timeline">
        <div className="pt-head">
          <h3>Pipeline Lifecycle</h3>
          <div className="pt-meta">
            <span className="pt-chip"><Clock3 size={12} /> {fmtSec(performance.total_duration_seconds || ps.total_duration_seconds)}</span>
            <span className={`pt-chip ${pipelineTone}`}>{finishedStages}/{stages.length} complete</span>
          </div>
        </div>
        <div className="pt-scroll">
          <div className="pt-track">
            {stageStates.map((s, i) => {
              const SIcon = s.icon;
              return (
                <React.Fragment key={s.key}>
                  {i > 0 && (
                    <div className={`pt-connector ${s.state === 'pending' ? 'pending' : 'done'}`}>
                      <div className="pt-conn-inner" />
                    </div>
                  )}
                  <div className={`pt-node ${s.state}`}>
                    <div className="pt-node-head">
                      <div className="pt-num">{i + 1}</div>
                      <div className="pt-icon"><SIcon size={15} /></div>
                    </div>
                    <strong className="pt-name">{s.name}</strong>
                    <small className="pt-timing">{fmtStage(s.stageData)}</small>
                    <span className={`pt-badge ${s.state}`}>{s.label}</span>
                  </div>
                </React.Fragment>
              );
            })}
          </div>
        </div>
      </section>

      {/* ━━━ DATA FUNNEL ━━━ */}
      <section className="data-funnel">
        <h3 className="df-title"><Database size={16} /> Data Throughput</h3>
        <div className="df-bar">
          <div className="df-seg df-raw" style={{ flex: 1 }}>
            <strong>{fmtInt(rawRows)}</strong>
            <small>Ingested</small>
          </div>
          <div className="df-seg df-processed" style={{ flex: rawRows ? (processedRows || 0) / rawRows : 0.8 }}>
            <strong>{fmtInt(processedRows)}</strong>
            <small>Processed</small>
          </div>
          {rejectedRows > 0 && (
            <div className="df-seg df-rejected" style={{ flex: rawRows ? rejectedRows / rawRows : 0.05 }}>
              <strong>{fmtInt(rejectedRows)}</strong>
              <small>Rejected</small>
            </div>
          )}
        </div>
        <div className="df-stats">
          <span>Acceptance: <strong>{rawRows ? fmtPct(1 - rejectedRatio) : 'n/a'}</strong></span>
          <span>Pipeline: <strong>{fmtSec(performance.total_duration_seconds || ps.total_duration_seconds)}</strong></span>
        </div>
      </section>

      {/* ━━━ INTELLIGENCE GRID ━━━ */}
      <section className="intel-grid">
        <div className="intel-events">
          <div className="ie-head">
            <h3>Pipeline Events</h3>
            <span className={`pill ${eventToneToPill(pipelineTone)}`}>
              {failedStages ? `${failedStages} failed` : warningStages ? `${warningStages} warnings` : 'healthy'}
            </span>
          </div>
          <div className="ie-list">
            {pipelineEvents.map((ev, i) => <EventItem event={ev} key={`${ev.title}-${i}`} />)}
          </div>
        </div>
        <div className="intel-cards">
          <InfoCard accent="sky" icon={<Brain size={17} />} title="Model" badge={fallbackMode ? 'fallback' : 'trained'} badgeTone={fallbackMode ? 'warning' : 'positive'}
            rows={[['Name', model.model_name || 'not trained'], ['Version', model.model_version || 'fallback'], ['MLflow', shortText(model.mlflow_run_id, 14)], ['Git', shortText(model.git_commit, 12)], ['Selected', comparison.selected_candidate || ps.selected_model || 'n/a'], ['Test F1', fmtMetric(f1)]]}
          />
          <InfoCard accent="amber" icon={<Database size={17} />} title="Data Quality" badge={validationWarnings.length ? `${validationWarnings.length} warnings` : 'clean'} badgeTone={validationWarnings.length ? 'warning' : 'positive'}
            rows={[['Dataset', ingestion.dataset_name || ps.dataset_name || 'n/a'], ['Drift Score', fmtMetric(drift.drift_score || ps.drift_score)], ['Candidates', fmtInt((comparison.candidates || []).length || ps.candidate_count)], ['Latency', fmtMs(summary?.evaluation?.latency_ms_per_review)], ['Fallback', String(ingestion.fallback_used ?? false)], ['EDA Rows', fmtInt(summary?.eda?.rows || ps.eda_rows)]]}
          />
        </div>
      </section>

      {/* ━━━ INFRASTRUCTURE BAR ━━━ */}
      <section className="infra-bar">
        <span className="infra-label"><Workflow size={14} /> Infrastructure</span>
        <div className="infra-links">
          {Object.entries(links).map(([label, href]) => {
            const LIcon = linkIcons[label] || GitBranch;
            return (
              <a href={href} key={label} target="_blank" rel="noreferrer" className="infra-link">
                <LIcon size={14} /><span>{titleCase(label)}</span>
              </a>
            );
          })}
        </div>
      </section>
    </main>
  );
}

function getPipelineStageState(stage, summary) {
  const reports = summary?.pipeline?.reports || {};
  const report = reports[stage.report] || {};
  const stageData = summary?.pipeline_performance?.stages?.[stage.key];
  const validationWarnings = summary?.validation?.warnings || [];
  const preprocessing = summary?.preprocessing || {};
  const drift = summary?.drift || {};
  const evaluation = summary?.evaluation || {};
  const ingestion = summary?.ingestion || {};
  const pipeline = summary?.pipeline || {};
  const rawStatus = stage.report === 'pipeline_report.json' ? pipeline.status : report.status;
  const hasErrors = Array.isArray(report.errors) && report.errors.length > 0;

  if (!summary) return { ...stage, stageData, state: 'pending', label: 'waiting' };
  if (hasErrors || ['failed', 'failure', 'error'].includes(String(rawStatus || '').toLowerCase())) {
    return { ...stage, stageData, state: 'failed', label: 'failed' };
  }
  if (stage.key === 'validate_raw_data' && validationWarnings.length) {
    return { ...stage, stageData, state: 'warning', label: `${validationWarnings.length} warnings` };
  }
  if (stage.key === 'preprocess_data' && Number(preprocessing.rejected_rows || 0) > 0) {
    return { ...stage, stageData, state: 'warning', label: `${preprocessing.rejected_rows} rejected` };
  }
  if (stage.key === 'run_batch_drift_check' && drift.drift_detected) {
    return { ...stage, stageData, state: 'warning', label: 'drift' };
  }
  if (stage.key === 'evaluate_selected_model' && evaluation.accepted === false) {
    return { ...stage, stageData, state: 'warning', label: 'not accepted' };
  }
  if (stage.key === 'ingest_data' && ingestion.fallback_used) {
    return { ...stage, stageData, state: 'warning', label: 'fallback' };
  }
  if (rawStatus === 'success' || stageData) {
    return { ...stage, stageData, state: 'success', label: 'success' };
  }
  return { ...stage, stageData, state: 'pending', label: 'pending' };
}

function buildPipelineEvents({ error, summary, validationWarnings, rejectedRows, rejectedRatio, accepted, drift, performance, ps, stageStates }) {
  if (!summary && !error) {
    return [{ tone: 'pending', title: 'Waiting for live reports', detail: 'Refresh the dashboard after the API is running to load pipeline status.' }];
  }
  const events = [];
  if (error) events.push({ tone: 'failed', title: 'Dashboard refresh failed', detail: error });

  stageStates.filter((s) => s.state === 'failed').forEach((s) => {
    events.push({ tone: 'failed', title: `${s.name} stage failed`, detail: s.label });
  });

  if (summary?.api?.ready) {
    events.push({ tone: 'success', title: 'API and trained model are ready', detail: 'The serving endpoint reports healthy readiness.' });
  } else if (summary) {
    events.push({ tone: 'warning', title: 'API readiness needs review', detail: 'The model may be missing, fallback-only, or still starting.' });
  }

  if (summary?.pipeline?.status === 'success' || ps?.accepted) {
    events.push({ tone: 'success', title: 'DVC lifecycle published successfully', detail: `${fmtSec(performance.total_duration_seconds || ps.total_duration_seconds)} total runtime across ${stageStates.length} stages.` });
  }

  validationWarnings.slice(0, 3).forEach((w) => events.push({ tone: 'warning', title: 'Validation warning', detail: w }));

  if (Number(rejectedRows || 0) > 0) {
    events.push({ tone: 'warning', title: 'Preprocessing rejected rows', detail: `${fmtInt(rejectedRows)} rows removed during cleaning (${fmtPct(rejectedRatio)} of raw data).` });
  }

  if (accepted === false) {
    events.push({ tone: 'warning', title: 'Model acceptance gate failed', detail: 'Macro F1 or latency did not satisfy the configured threshold.' });
  } else if (accepted === true) {
    events.push({ tone: 'success', title: 'Model acceptance gate passed', detail: `Macro F1 ${fmtMetric(summary?.evaluation?.metrics?.macro_f1 || ps.test_macro_f1)}, latency ${fmtMs(summary?.evaluation?.latency_ms_per_review)}.` });
  }

  if (drift?.drift_detected) {
    events.push({ tone: 'warning', title: 'Data drift detected', detail: `Latest drift score is ${fmtMetric(drift.drift_score)}.` });
  } else if (summary?.drift?.status === 'success') {
    events.push({ tone: 'success', title: 'Drift check normal', detail: `Latest drift score is ${fmtMetric(drift.drift_score)}.` });
  }

  const batch = summary?.batch_pipeline;
  if (batch?.status === 'quarantined') {
    events.push({ tone: 'failed', title: 'Incoming batch quarantined', detail: batch.reason || 'Malformed input was isolated from the pipeline.' });
  } else if (Number(batch?.failed_chunks || 0) > 0) {
    events.push({ tone: 'failed', title: 'Batch chunks failed', detail: `${batch.failed_chunks} chunks failed during batch processing.` });
  } else if (batch?.status === 'success') {
    events.push({ tone: 'success', title: 'Batch pipeline completed', detail: `${fmtInt(batch.rows_processed)} rows processed across ${fmtInt(batch.chunk_count)} chunks.` });
  } else {
    events.push({ tone: 'pending', title: 'No recent batch file event', detail: 'The incoming-file Airflow DAG has no latest batch report yet.' });
  }

  return events.slice(0, 8);
}

function EventItem({ event }) {
  const icons = { success: CheckCircle2, warning: AlertTriangle, failed: XCircle, pending: Clock3 };
  const Icon = icons[event.tone] || Clock3;
  return (
    <div className={`event-item ${event.tone}`}>
      <div className="event-icon"><Icon size={16} /></div>
      <div className="event-copy">
        <strong>{event.title}</strong>
        <small>{event.detail}</small>
      </div>
    </div>
  );
}

function eventToneToPill(tone) {
  if (tone === 'failed') return 'danger';
  if (tone === 'warning') return 'warning';
  if (tone === 'success') return 'positive';
  return 'neutral';
}

function InfoCard({ accent, icon, title, badge, badgeTone, rows }) {
  return (
    <div className={`info-card ic-${accent}`}>
      <div className="ic-head">
        <div className="ic-title">{icon}<h4>{title}</h4></div>
        <span className={`pill ${badgeTone}`}>{badge}</span>
      </div>
      <div className="ic-rows">
        {rows.map(([k, v]) => (
          <div className="ic-row" key={k}><span>{k}</span><strong>{v}</strong></div>
        ))}
      </div>
    </div>
  );
}

function StatChip({ icon, value }) {
  return <div className="stat-chip">{icon}<strong>{value}</strong></div>;
}

/* ═══════════════════════════════════════════════════════
   GUIDE PANEL
   ═══════════════════════════════════════════════════════ */
function GuidePanel({ onClose }) {
  return (
    <div className="guide-bg" onClick={onClose}>
      <aside className="guide-panel" onClick={e => e.stopPropagation()} role="dialog" aria-modal="true">
        <div className="guide-top"><h2>How to use SentinelAI</h2><button className="btn-icon" onClick={onClose}><X size={18} /></button></div>
        <div className="guide-list">
          <GStep icon={<ClipboardCheck size={20} />} title="Enter a review" desc="Paste any product review or pick an example." />
          <GStep icon={<Send size={20} />} title="Hit Analyze" desc="Send the review to the AI model for classification." />
          <GStep icon={<TrendingUp size={20} />} title="Read results" desc="See sentiment, confidence ring, probability bars, and influential tokens." />
          <GStep icon={<Star size={20} />} title="Give feedback" desc="Select the real label to help track prediction quality." />
          <GStep icon={<Activity size={20} />} title="Check MLOps" desc="Switch tabs to see pipeline health, drift, and infrastructure links." />
        </div>
      </aside>
    </div>
  );
}
function GStep({ icon, title, desc }) {
  return (
    <div className="g-step">
      <div className="g-step-icon">{icon}</div>
      <div><strong>{title}</strong><p>{desc}</p></div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════
   UTILITIES
   ═══════════════════════════════════════════════════════ */
async function safeJson(r) { try { return await r.json(); } catch { return null; } }
function wordCount(v) { const t = v.trim(); return t ? t.split(/\s+/).length : 0; }
function fmtMetric(v) { return typeof v === 'number' ? v.toFixed(3) : 'n/a'; }
function fmtInt(v) { return typeof v === 'number' ? new Intl.NumberFormat().format(v) : 'n/a'; }
function fmtPct(v) { return typeof v === 'number' ? `${(v * 100).toFixed(v < 0.01 ? 2 : 1)}%` : 'n/a'; }
function fmtSec(v) { if (typeof v !== 'number') return 'n/a'; return v < 1 ? `${Math.round(v * 1000)} ms` : `${v.toFixed(1)} s`; }
function fmtMs(v) { return typeof v === 'number' ? `${v.toFixed(2)} ms` : 'n/a'; }
function fmtStage(s) { if (!s) return 'pending'; const t = s.throughput_rows_per_second; return typeof t === 'number' ? `${fmtSec(s.duration_seconds)} · ${Math.round(t)}/s` : fmtSec(s.duration_seconds); }
function shortText(v, n) { return v ? (v.length > n ? v.slice(0, n) + '…' : v) : 'n/a'; }
function titleCase(v) { return v ? v[0].toUpperCase() + v.slice(1) : ''; }
function fmtWeight(v) { return typeof v === 'number' ? (v >= 0 ? `+${v.toFixed(2)}` : v.toFixed(2)) : ''; }

createRoot(document.getElementById('root')).render(<App />);
