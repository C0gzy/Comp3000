"use client";

import { useMemo } from "react";

export interface ClassificationData {
  predicted_label: string;
  probability: number;
}

interface ClassificationResultProps {
  file: File;
  result: ClassificationData;
}

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
  const overlay = getOverlayColor(result.probability);
  const pct = (result.probability * 100).toFixed(1);

  return (
    <div className="w-full rounded-xl overflow-hidden border border-gray-200 dark:border-gray-800 bg-white dark:bg-zinc-900">
      <div
        className="relative w-full"
        style={{ outline: `3px solid ${overlay.border}` }}
      >
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
          {pct}%
        </span>
      </div>
    </div>
  );
}
