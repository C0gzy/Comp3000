"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { ClassificationResult, type ClassificationData } from "@/components/classification-result";
import Image from "next/image";
import { ExternalLink, Loader2, Waves } from "lucide-react";
import Link from "next/link";
import { CORAL_DATASET_URL } from "@/lib/dataset";

const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"];

export default function Home() {
  const [isDragging, setIsDragging] = useState(false);
  const [files, setFiles] = useState<File[]>([]);
  const [result, setResult] = useState<{ file: File; data: ClassificationData } | null>(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleClassify = useCallback(async () => {
    if (!files.length) return;
    setIsClassifying(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("image", files[0]);
      const response = await fetch("http://localhost:5001/classify", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(`Classification failed: ${response.statusText}`);
      }
      const data: ClassificationData = await response.json();
      setResult({ file: files[0], data });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to classify image. Is the server running?");
    } finally {
      setIsClassifying(false);
    }
  }, [files]);

  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
    const dropped = Array.from(event.dataTransfer.files).filter((f) =>
      ACCEPTED_TYPES.includes(f.type)
    );
    if (dropped.length) {
      setFiles((prev) => [...prev, ...dropped]);
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

  const handleFileSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const selected = Array.from(event.target.files ?? []).filter((f) =>
      ACCEPTED_TYPES.includes(f.type)
    );
    if (selected.length) {
      setFiles((prev) => [...prev, ...selected]);
      setError(null);
    }
    event.target.value = "";
  }, []);

  return (
    <div className="relative flex min-h-screen flex-col font-sans dark:bg-black">
      <div className="absolute inset-0 -z-10">
        <Image
          src="/background.jpg"
          fill
          sizes="100vw"
          alt="Underwater coral reef"
          className="object-cover"
        />
      </div>
      <Button
        variant="outline"
        size="sm"
        className="fixed top-4 right-4 z-20 shadow-md backdrop-blur-sm"
        asChild
      >
        <Link href="/docs" className="no-underline">
          documentation
        </Link>
      </Button>
      <div className="flex flex-1 flex-col items-center justify-center px-4 py-10">
        <main className="relative flex w-full max-w-3xl flex-col items-center rounded-xl bg-white/90 p-6 shadow-xl backdrop-blur-md dark:bg-zinc-900/90 dark:backdrop-blur-md sm:items-start">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white">
            Coral Reef Health Monitor
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Upload an image of coral to classify its health status.
          </p>

          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`mt-8 w-full rounded-xl border-2 border-dashed transition-colors ${
              isDragging
                ? "border-sky-400 bg-sky-50 dark:border-sky-500 dark:bg-sky-950/30"
                : "border-sky-200 bg-sky-50/50 dark:border-sky-800 dark:bg-sky-950/20"
            }`}
          >
            <label className="flex cursor-pointer flex-col items-center justify-center gap-4 py-16 px-8">
              <div className="rounded-full bg-sky-100 p-4 dark:bg-sky-900/50">
                <Waves className="size-10 text-sky-500 dark:text-sky-400" />
              </div>
              <div className="text-center">
                <span className="text-sky-600 dark:text-sky-400 font-medium">
                  Drop images here
                </span>
                <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                  or click to browse — PNG, JPG, GIF, WebP
                </p>
              </div>
              <input
                type="file"
                accept="image/png,image/jpeg,image/jpg,image/gif,image/webp"
                multiple
                onChange={handleFileSelect}
                className="hidden"
              />
            </label>
          </div>

          {error && (
            <div className="mt-6 w-full rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-950/30 dark:text-red-400">
              {error}
            </div>
          )}

          {files.length > 0 && (
            <div className="mt-6 w-full space-y-2">
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                {files.length} file{files.length !== 1 ? "s" : ""} selected
              </p>
              <div className="flex flex-wrap gap-3">
                {files.map((f, i) => (
                  <div
                    key={`${f.name}-${i}`}
                    className="flex items-center gap-2 rounded-lg border border-gray-200 bg-gray-50 p-2 dark:border-gray-700 dark:bg-gray-800"
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={previewUrls[i]}
                      alt={f.name}
                      className="size-12 rounded object-cover"
                    />
                    <span className="max-w-[120px] truncate text-sm text-gray-700 dark:text-gray-300">
                      {f.name}
                    </span>
                  </div>
                ))}
              </div>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  className="mt-2"
                  onClick={() => { setFiles([]); setResult(null); setError(null); }}
                >
                  Clear
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="mt-2"
                  onClick={handleClassify}
                  disabled={isClassifying}
                >
                  {isClassifying ? (
                    <>
                      <Loader2 className="mr-2 size-4 animate-spin" />
                      Classifying…
                    </>
                  ) : (
                    "Classify"
                  )}
                </Button>
              </div>
            </div>
          )}

          {result && (
            <div className="mt-8">
              <ClassificationResult file={result.file} result={result.data} />
            </div>
          )}
        </main>
      </div>

      <footer className="relative z-10 px-4 pb-8">
        <div className="mx-auto flex max-w-2xl flex-col items-center justify-center gap-3 text-center sm:flex-row sm:text-left">
          <Button variant="outline" asChild size="sm">
            <a
              href={CORAL_DATASET_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="no-underline"
            >
               Training data from the NOAA PIFSC ESD coral bleaching dataset
              <ExternalLink className="size-4" aria-hidden />
            </a>
          </Button>
        </div>
      </footer>
    </div>
  );
}
