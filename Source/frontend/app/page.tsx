"use client";

import type { ChangeEvent, DragEvent } from "react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { AppBackground } from "@/components/app-background";
import { ClassificationResult } from "@/components/classification-result";
import { DatasetFooter } from "@/components/dataset-footer";
import { DocsLink } from "@/components/docs-link";
import { SelectedFiles } from "@/components/selected-files";
import { UploadDropzone } from "@/components/upload-dropzone";
import { ClassificationData } from "@/lib/types";

const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"];
const API_BASE = process.env.Production_API_URL || "http://localhost:5001";
const POLL_INTERVAL_MS = 750;

type JobStatus = ClassificationData & {
  status: "queued" | "processing" | "completed" | "error";
  completed_chunks: number;
  total_chunks: number;
  remaining_chunks: number;
  progress: string;
  error?: string;
};

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export default function Home() {
  const [isDragging, setIsDragging] = useState(false);
  const [files, setFiles] = useState<File[]>([]);
  const [results, setResults] = useState<{ file: File; data: ClassificationData }[]>([]);
  const [isClassifying, setIsClassifying] = useState(false);
  const [classificationProgress, setClassificationProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleClassify = useCallback(async () => {
    if (!files.length) return;
    setIsClassifying(true);
    setClassificationProgress(null);
    setError(null);
    setResults([]);
    try {
      for (const [index, file] of files.entries()) {
        const imageNumber = `${index + 1}/${files.length}`;
        setClassificationProgress(`${imageNumber}: queueing`);

        const formData = new FormData();
        formData.append("image", file);
        const response = await fetch(`${API_BASE}/V1/api/classify`, {
          method: "POST",
          body: formData,
        });
        if (!response.ok) {
          throw new Error(`${file.name}: classification failed (${response.statusText})`);
        }
        const { job_id: jobId }: { job_id: number } = await response.json();

        while (true) {
          const statusResponse = await fetch(`${API_BASE}/V1/api/status/${jobId}`);
          if (!statusResponse.ok) {
            throw new Error(`${file.name}: status check failed (${statusResponse.statusText})`);
          }

          const status: JobStatus = await statusResponse.json();
          setClassificationProgress(`${imageNumber}: ${status.progress}`);

          if (status.status === "error") {
            throw new Error(`${file.name}: ${status.error ?? "classification job failed"}`);
          }

          if (status.status === "completed") {
            setResults((currentResults) => [...currentResults, { file, data: status }]);
            break;
          }

          await wait(POLL_INTERVAL_MS);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to classify image. Is the server running?");
    } finally {
      setIsClassifying(false);
      setClassificationProgress(null);
    }
  }, [files]);

  const handleDragOver = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
    const dropped = Array.from(event.dataTransfer.files).filter((f) =>
      ACCEPTED_TYPES.includes(f.type)
    );
    if (dropped.length) {
      setFiles((prev) => [...prev, ...dropped]);
      setResults([]);
      setError(null);
    }
  }, []);

  const previewUrls = useMemo(
    () => files.map((f) => URL.createObjectURL(f)),
    [files]
  );

  useEffect(() => {
    return () => previewUrls.forEach((url) => URL.revokeObjectURL(url));
  }, [previewUrls]);

  const handleFileSelect = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const selected = Array.from(event.target.files ?? []).filter((f) =>
      ACCEPTED_TYPES.includes(f.type)
    );
    if (selected.length) {
      setFiles((prev) => [...prev, ...selected]);
      setResults([]);
      setError(null);
    }
    event.target.value = "";
  }, []);

  const handleClear = useCallback(() => {
    setFiles([]);
    setResults([]);
    setError(null);
    setClassificationProgress(null);
  }, []);

  return (
    <div className="relative flex min-h-screen flex-col font-sans dark:bg-black">
      <AppBackground />
      <DocsLink />
      <div className="flex flex-1 flex-col items-center justify-center px-4 py-10">
        <main className="relative flex w-full max-w-3xl flex-col items-center rounded-xl bg-white/90 p-6 shadow-xl backdrop-blur-md dark:bg-zinc-900/90 dark:backdrop-blur-md sm:items-start">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white">
            Coral Reef Health Monitor
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Upload an image of coral to classify its health status.
          </p>

          <UploadDropzone
            isDragging={isDragging}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onFileSelect={handleFileSelect}
          />

          {error && (
            <div className="mt-6 w-full rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-950/30 dark:text-red-400">
              {error}
            </div>
          )}

          <SelectedFiles
            files={files}
            previewUrls={previewUrls}
            isClassifying={isClassifying}
            classificationProgress={classificationProgress}
            onClassify={handleClassify}
            onClear={handleClear}
          />

          {results.length > 0 && (
            <div className="mt-8 w-full space-y-4">
              {results.map((result, index) => (
                <ClassificationResult
                  key={`${result.file.name}-${index}`}
                  file={result.file}
                  result={result.data}
                />
              ))}
            </div>
          )}
        </main>
      </div>

      <DatasetFooter />
    </div>
  );
}
