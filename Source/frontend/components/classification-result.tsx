"use client";

import { useMemo } from "react";
import { ClassificationResultProps } from "@/lib/types";

function getOverlayColor(probability: number): {
  bg: string;
  border: string;
  label: string;
  textColor: string;
} {
  if (probability >= 0.4 && probability <= 0.6) {
    return {
      bg: "rgba(251, 146, 60, 0.25)",
      border: "rgb(251, 146, 60)",
      label: "Uncertain",
      textColor: "text-orange-500",
    };
  }
  if (probability >= 0.5) {
    return {
      bg: "rgba(239, 68, 68, 0.25)",
      border: "rgb(239, 68, 68)",
      label: "Bleached",
      textColor: "text-red-500",
    };
  }
  return {
    bg: "rgba(34, 197, 94, 0.25)",
    border: "rgb(34, 197, 94)",
    label: "Healthy",
    textColor: "text-green-500",
  };
}

export function ClassificationResult({ file, result }: ClassificationResultProps) {
  const imageUrl = useMemo(() => URL.createObjectURL(file), [file]);
  const isChunkedResult = Boolean(result.result_image);
  const probability = result.probability ?? 0;
  const overlay = getOverlayColor(probability);
  const percentage = (probability * 100).toFixed(1);

  if (isChunkedResult) {
    return (
      <div className="w-full overflow-hidden rounded-xl border border-gray-200 bg-white dark:border-gray-800 dark:bg-zinc-900">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={result.result_image}
          alt={`Chunked classification result for ${file.name}`}
          className="block h-auto w-full"
        />
        <div className="px-4 py-3">
          <p className="text-sm font-medium text-gray-900 dark:text-white">
            Chunked classification complete
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            {result.progress ?? `${result.completed_chunks ?? 0}/${result.total_chunks ?? 0}`} chunks analysed
          </p>
          {result.summary && (
            <div className="mt-3 grid grid-cols-3 gap-2 text-center text-xs">
              <div className="rounded-md bg-red-50 px-2 py-2 text-red-700 dark:bg-red-950/30 dark:text-red-300">
                <span className="block text-lg font-bold">{result.summary.bleached_chunks}</span>
                bleached
              </div>
              <div className="rounded-md bg-green-50 px-2 py-2 text-green-700 dark:bg-green-950/30 dark:text-green-300">
                <span className="block text-lg font-bold">{result.summary.healthy_chunks}</span>
                healthy
              </div>
              <div className="rounded-md bg-gray-50 px-2 py-2 text-gray-700 dark:bg-zinc-800 dark:text-gray-300">
                <span className="block text-lg font-bold">{result.summary.total_chunks}</span>
                total
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="w-full rounded-xl overflow-hidden border border-gray-200 dark:border-gray-800 bg-white dark:bg-zinc-900">
      <div
        className="relative w-full"
        style={{ outline: `3px solid ${overlay.border}` }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={imageUrl}
          alt={file.name}
          className="w-full h-auto block"
        />
        <div
          className="absolute inset-0 pointer-events-none"
          style={{ backgroundColor: overlay.bg }}
        />
        <span
          className="absolute top-3 left-3 rounded-md px-2.5 py-1 text-xs font-bold uppercase tracking-wide text-white"
          style={{ backgroundColor: overlay.border }}
        >
          {overlay.label}
        </span>
      </div>

      <div className="px-4 py-3 flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-900 dark:text-white">
            {result.predicted_label}
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-400 truncate max-w-[200px]">
            {file.name}
          </p>
        </div>
        <span className={`text-lg font-bold tabular-nums ${overlay.textColor}`}>
          {percentage}%
        </span>
      </div>
    </div>
  );
}
