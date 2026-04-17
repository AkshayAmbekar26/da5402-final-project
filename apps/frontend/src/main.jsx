import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Gauge,
  GitBranch,
  HeartHandshake,
  Loader2,
  Play,
  RefreshCw,
  Send,
  Server,
} from 'lucide-react';
import './styles.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const examples = [
  'Excellent quality and the delivery was faster than expected. I would buy this again.',
  'The product is okay and works as described, but there is nothing special about it.',
  'The package arrived damaged and the product stopped working after two days.',
];

const sentimentCopy = {
  positive: { label: 'Positive', tone: 'positive' },
  neutral: { label: 'Neutral', tone: 'neutral' },
  negative: { label: 'Negative', tone: 'negative' },
};

function App() {
  const [screen, setScreen] = useState('predict');
  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">MLOps AI Application</p>
          <h1>Product Review Sentiment Analyzer</h1>
        </div>
        <nav className="tabs" aria-label="Application views">
          <button className={screen === 'predict' ? 'active' : ''} onClick={() => setScreen('predict')}>
            <HeartHandshake size={18} />
            Analyzer
          </button>
          <button className={screen === 'ops' ? 'active' : ''} onClick={() => setScreen('ops')}>
            <Activity size={18} />
            MLOps
          </button>
        </nav>
      </header>
      {screen === 'predict' ? <PredictScreen /> : <OpsScreen />}
    </div>
  );
}

function PredictScreen() {
  const [review, setReview] = useState(examples[0]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [feedbackStatus, setFeedbackStatus] = useState('');

  async function submitReview() {
    setError('');
    setFeedbackStatus('');
    const trimmed = review.trim();
    if (!trimmed) {
      setError('Enter a product review before running the analyzer.');
      return;
    }
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review_text: trimmed }),
      });
      if (!response.ok) throw new Error(`API returned ${response.status}`);
      setResult(await response.json());
    } catch (err) {
      setError('The API is unavailable. Start the FastAPI service and try again.');
    } finally {
      setLoading(false);
    }
  }

  async function sendFeedback(actualSentiment) {
    if (!result) return;
    setFeedbackStatus('');
    const response = await fetch(`${API_BASE_URL}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        review_text: review,
        predicted_sentiment: result.sentiment,
        actual_sentiment: actualSentiment,
        source: 'frontend-demo',
      }),
    });
    setFeedbackStatus(response.ok ? 'Feedback saved for monitoring.' : 'Feedback could not be saved.');
  }

  return (
    <main className="workspace analyzer-grid">
      <section className="panel input-panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Customer Review</p>
            <h2>Analyze sentiment in real time</h2>
          </div>
          <button className="icon-button" onClick={() => setReview('')} title="Clear review">
            <RefreshCw size={18} />
          </button>
        </div>
        <textarea
          value={review}
          maxLength={5000}
          onChange={(event) => setReview(event.target.value)}
          placeholder="Paste a product review here..."
        />
        <div className="helper-row">
          <span>{review.length}/5000 characters</span>
          <button className="primary-button" onClick={submitReview} disabled={loading}>
            {loading ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
            Analyze
          </button>
        </div>
        <div className="example-row">
          {examples.map((example) => (
            <button key={example} onClick={() => setReview(example)}>
              {example.slice(0, 44)}...
            </button>
          ))}
        </div>
        {error && <p className="error-text">{error}</p>}
      </section>

      <section className="panel result-panel">
        {!result ? (
          <div className="empty-state">
            <Gauge size={42} />
            <h2>Prediction output appears here</h2>
            <p>Run the analyzer to see sentiment, confidence, explanation, latency, and model version.</p>
          </div>
        ) : (
          <ResultCard result={result} onFeedback={sendFeedback} feedbackStatus={feedbackStatus} />
        )}
      </section>
    </main>
  );
}

function ResultCard({ result, onFeedback, feedbackStatus }) {
  const sentiment = sentimentCopy[result.sentiment] || sentimentCopy.neutral;
  const probabilityRows = Object.entries(result.class_probabilities || {});
  return (
    <div className="result-content">
      <div className={`sentiment-badge ${sentiment.tone}`}>{sentiment.label}</div>
      <h2>{Math.round(result.confidence * 100)}% confidence</h2>
      <div className="probabilities">
        {probabilityRows.map(([label, value]) => (
          <div className="prob-row" key={label}>
            <span>{label}</span>
            <div className="bar-track">
              <div className={`bar-fill ${label}`} style={{ width: `${Math.round(value * 100)}%` }} />
            </div>
            <strong>{Math.round(value * 100)}%</strong>
          </div>
        ))}
      </div>
      <div className="explanation">
        <h3>Influential words</h3>
        <div className="token-list">
          {(result.explanation || []).length ? (
            result.explanation.map((item) => (
              <span key={`${item.token}-${item.weight}`}>{item.token}</span>
            ))
          ) : (
            <span>No strong token contribution found</span>
          )}
        </div>
      </div>
      <div className="metadata-grid">
        <Metric label="Latency" value={`${result.latency_ms.toFixed(1)} ms`} />
        <Metric label="Model" value={result.model_version} />
        <Metric label="MLflow run" value={result.mlflow_run_id.slice(0, 10)} />
      </div>
      <div className="feedback-box">
        <p>Was the prediction correct?</p>
        <div className="feedback-actions">
          {['positive', 'neutral', 'negative'].map((label) => (
            <button key={label} onClick={() => onFeedback(label)}>{label}</button>
          ))}
        </div>
        {feedbackStatus && <small>{feedbackStatus}</small>}
      </div>
    </div>
  );
}

function OpsScreen() {
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState('');

  async function loadSummary() {
    setError('');
    try {
      const response = await fetch(`${API_BASE_URL}/metrics-summary`);
      if (!response.ok) throw new Error(`API returned ${response.status}`);
      setSummary(await response.json());
    } catch (err) {
      setError('Cannot load MLOps summary. Start the API to view live status.');
    }
  }

  useEffect(() => {
    loadSummary();
  }, []);

  const links = summary?.links || {
    airflow: 'http://localhost:8080',
    mlflow: 'http://localhost:5000',
    prometheus: 'http://localhost:9090',
    grafana: 'http://localhost:3000',
  };
  const metrics = summary?.evaluation?.metrics || summary?.evaluation?.test || {};
  const drift = summary?.drift || {};
  const model = summary?.model?.metadata || {};
  const ingestion = summary?.ingestion || {};
  const preprocessing = summary?.preprocessing || {};
  const comparison = summary?.model_comparison || {};
  const performance = summary?.pipeline_performance || {};
  const pipelineSummary = summary?.pipeline_summary || {};
  const stages = [
    ['ingest', 'ingest_data'],
    ['validate', 'validate_raw_data'],
    ['eda', 'run_eda'],
    ['preprocess', 'preprocess_data'],
    ['baseline', 'compute_drift_baseline'],
    ['compare', 'train_and_compare_models'],
    ['evaluate', 'evaluate_selected_model'],
    ['drift', 'run_batch_drift_check'],
    ['publish', 'publish_pipeline_report'],
  ];

  return (
    <main className="workspace ops-layout">
      <section className="status-strip">
        <StatusTile icon={<Server />} label="API" value={summary?.api?.healthy ? 'Healthy' : 'Unknown'} tone="positive" />
        <StatusTile icon={<CheckCircle2 />} label="Model" value={summary?.model?.model_loaded ? 'Loaded' : 'Fallback'} tone={summary?.model?.model_loaded ? 'positive' : 'warning'} />
        <StatusTile icon={<BarChart3 />} label="Macro F1" value={formatMetric(metrics.macro_f1)} tone="neutral" />
        <StatusTile icon={<AlertTriangle />} label="Drift" value={drift.drift_detected ? 'Detected' : 'Normal'} tone={drift.drift_detected ? 'warning' : 'positive'} />
      </section>

      <section className="status-strip">
        <StatusTile icon={<BarChart3 />} label="Raw rows" value={formatInteger(ingestion.rows || pipelineSummary.raw_rows)} tone="neutral" />
        <StatusTile icon={<CheckCircle2 />} label="Processed" value={formatInteger(preprocessing.final_rows || pipelineSummary.processed_rows)} tone="positive" />
        <StatusTile icon={<AlertTriangle />} label="Rejected" value={formatInteger(preprocessing.rejected_rows || pipelineSummary.rejected_rows)} tone={(preprocessing.rejected_rows || 0) > 0 ? 'warning' : 'positive'} />
        <StatusTile icon={<Gauge />} label="Pipeline" value={formatSeconds(performance.total_duration_seconds || pipelineSummary.total_duration_seconds)} tone="neutral" />
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Pipeline Console</p>
            <h2>Lifecycle visibility</h2>
          </div>
          <button className="secondary-button" onClick={loadSummary}>
            <RefreshCw size={18} />
            Refresh
          </button>
        </div>
        {error && <p className="error-text">{error}</p>}
        <div className="timeline">
          {stages.map(([stage, performanceKey], index) => (
            <div className="timeline-item" key={stage}>
              <span>{index + 1}</span>
              <strong>{stage}</strong>
              <small>{formatStage(performance?.stages?.[performanceKey])}</small>
            </div>
          ))}
        </div>
      </section>

      <section className="ops-grid">
        <div className="panel">
          <h2>Model metadata</h2>
          <dl className="info-list">
            <dt>Name</dt><dd>{model.model_name || 'not trained yet'}</dd>
            <dt>Version</dt><dd>{model.model_version || 'fallback'}</dd>
            <dt>MLflow run</dt><dd>{model.mlflow_run_id || 'unavailable'}</dd>
            <dt>Git commit</dt><dd>{model.git_commit || 'unavailable'}</dd>
            <dt>DVC version</dt><dd>{shortText(model.data_version, 32)}</dd>
          </dl>
        </div>
        <div className="panel">
          <h2>Model comparison</h2>
          <dl className="info-list">
            <dt>Selected</dt><dd>{comparison.selected_candidate || model.model_name || 'not available'}</dd>
            <dt>Candidates</dt><dd>{formatInteger((comparison.candidates || []).length || pipelineSummary.candidate_count)}</dd>
            <dt>Accepted</dt><dd>{formatInteger((comparison.accepted_candidates || []).length)}</dd>
            <dt>Test F1</dt><dd>{formatMetric(metrics.macro_f1)}</dd>
          </dl>
        </div>
      </section>

      <section className="ops-grid">
        <div className="panel">
          <h2>Dataset run</h2>
          <dl className="info-list">
            <dt>Dataset</dt><dd>{ingestion.dataset_name || pipelineSummary.dataset_name || 'not available'}</dd>
            <dt>Fallback</dt><dd>{String(ingestion.fallback_used ?? false)}</dd>
            <dt>EDA rows</dt><dd>{formatInteger(summary?.eda?.rows)}</dd>
            <dt>Warnings</dt><dd>{formatInteger((summary?.validation?.warnings || []).length)}</dd>
          </dl>
        </div>
        <div className="panel">
          <h2>MLOps tools</h2>
          <div className="tool-links">
            {Object.entries(links).map(([label, href]) => (
              <a href={href} key={label} target="_blank" rel="noreferrer">
                <GitBranch size={18} />
                {label}
              </a>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}

function StatusTile({ icon, label, value, tone }) {
  return (
    <div className={`status-tile ${tone}`}>
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function formatMetric(value) {
  if (typeof value !== 'number') return 'n/a';
  return value.toFixed(3);
}

function formatInteger(value) {
  if (typeof value !== 'number') return 'n/a';
  return new Intl.NumberFormat().format(value);
}

function formatSeconds(value) {
  if (typeof value !== 'number') return 'n/a';
  if (value < 1) return `${Math.round(value * 1000)} ms`;
  return `${value.toFixed(1)} s`;
}

function formatStage(stage) {
  if (!stage) return 'pending';
  const throughput = stage.throughput_rows_per_second;
  if (typeof throughput === 'number') return `${formatSeconds(stage.duration_seconds)} · ${Math.round(throughput)}/s`;
  return formatSeconds(stage.duration_seconds);
}

function shortText(value, length) {
  if (!value) return 'unavailable';
  return value.length > length ? `${value.slice(0, length)}...` : value;
}

createRoot(document.getElementById('root')).render(<App />);
