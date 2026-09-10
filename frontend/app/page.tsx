"use client";

import { useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Transaction = Record<string, number>;

type SampleResponse = {
  transaction: Transaction;
  true_label: "fraud" | "legit";
};

type PredictResponse = {
  fraud_probability: number;
  is_fraud: boolean;
  model_name: string;
};

export default function Home() {
  const [sample, setSample] = useState<SampleResponse | null>(null);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function fetchSample(label: "fraud" | "legit") {
    setError(null);
    setPrediction(null);
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/sample?label=${label}`);
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data: SampleResponse = await res.json();
      setSample(data);
    } catch (e) {
      setError(
        e instanceof Error
          ? `${e.message} — is the API running at ${API_URL}?`
          : "Unknown error"
      );
    } finally {
      setLoading(false);
    }
  }

  async function runPrediction() {
    if (!sample) return;
    setError(null);
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(sample.transaction),
      });
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data: PredictResponse = await res.json();
      setPrediction(data);
    } catch (e) {
      setError(
        e instanceof Error
          ? `${e.message} — is the API running at ${API_URL}?`
          : "Unknown error"
      );
    } finally {
      setLoading(false);
    }
  }

  const correct =
    prediction && sample
      ? (prediction.is_fraud ? "fraud" : "legit") === sample.true_label
      : null;

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center px-6 py-16">
      <div className="w-full max-w-2xl">
        <h1 className="text-3xl font-semibold mb-2">
          Credit Card Fraud Detection — Live Demo
        </h1>
        <p className="text-slate-400 mb-10">
          Pull a real transaction from the held-out test set and see whether
          the trained model catches it.
        </p>

        <div className="flex gap-3 mb-8">
          <button
            onClick={() => fetchSample("legit")}
            disabled={loading}
            className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition disabled:opacity-50"
          >
            Get a legit transaction
          </button>
          <button
            onClick={() => fetchSample("fraud")}
            disabled={loading}
            className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition disabled:opacity-50"
          >
            Get a fraud transaction
          </button>
        </div>

        {error && (
          <div className="mb-6 p-4 rounded-lg bg-red-950 border border-red-800 text-red-300 text-sm">
            {error}
          </div>
        )}

        {sample && (
          <div className="mb-6 p-5 rounded-xl bg-slate-900 border border-slate-800">
            <div className="flex items-center justify-between mb-3">
              <h2 className="font-medium text-slate-200">Transaction</h2>
              <span className="text-xs uppercase tracking-wide px-2 py-1 rounded bg-slate-800 text-slate-400">
                true label: {sample.true_label}
              </span>
            </div>
            <div className="grid grid-cols-3 gap-2 text-xs text-slate-400 font-mono">
              <div>Time: {sample.transaction.Time.toFixed(3)}</div>
              <div>Amount: {sample.transaction.Amount.toFixed(3)}</div>
              <div>V1: {sample.transaction.V1.toFixed(3)}</div>
              <div>V2: {sample.transaction.V2.toFixed(3)}</div>
              <div>V3: {sample.transaction.V3.toFixed(3)}</div>
              <div>… 25 more PCA features</div>
            </div>

            <button
              onClick={runPrediction}
              disabled={loading}
              className="mt-4 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 transition disabled:opacity-50 text-sm font-medium"
            >
              {loading ? "Scoring…" : "Run prediction"}
            </button>
          </div>
        )}

        {prediction && (
          <div
            className={`p-5 rounded-xl border ${
              correct
                ? "bg-emerald-950 border-emerald-800"
                : "bg-amber-950 border-amber-800"
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <h2 className="font-medium">
                Prediction: {prediction.is_fraud ? "Fraud" : "Legit"}
              </h2>
              <span className="text-xs px-2 py-1 rounded bg-black/20">
                {prediction.model_name}
              </span>
            </div>
            <p className="text-sm text-slate-300">
              Fraud probability:{" "}
              <span className="font-mono">
                {(prediction.fraud_probability * 100).toFixed(1)}%
              </span>
            </p>
            <p className="text-sm mt-2">
              {correct ? "✓ Matches the true label" : "✗ Does not match the true label"}
            </p>
          </div>
        )}
      </div>
    </main>
  );
}
