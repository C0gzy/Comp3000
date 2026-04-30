"use client";

import type { ChangeEvent, DragEvent } from "react";
import { Waves } from "lucide-react";

interface UploadDropzoneProps {
  isDragging: boolean;
  onDragOver: (event: DragEvent<HTMLDivElement>) => void;
  onDragLeave: (event: DragEvent<HTMLDivElement>) => void;
  onDrop: (event: DragEvent<HTMLDivElement>) => void;
  onFileSelect: (event: ChangeEvent<HTMLInputElement>) => void;
}

export function UploadDropzone({
  isDragging,
  onDragOver,
  onDragLeave,
  onDrop,
  onFileSelect,
}: UploadDropzoneProps) {
  return (
    <div
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      className={`mt-8 w-full rounded-xl border-2 border-dashed transition-colors ${
        isDragging
          ? "border-sky-400 bg-sky-50 dark:border-sky-500 dark:bg-sky-950/30"
          : "border-sky-200 bg-sky-50/50 dark:border-sky-800 dark:bg-sky-950/20"
      }`}
    >
      <label className="flex cursor-pointer flex-col items-center justify-center gap-4 px-8 py-16">
        <div className="rounded-full bg-sky-100 p-4 dark:bg-sky-900/50">
          <Waves className="size-10 text-sky-500 dark:text-sky-400" />
        </div>
        <div className="text-center">
          <span className="font-medium text-sky-600 dark:text-sky-400">
            Drop images here
          </span>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            or click to browse - PNG, JPG, GIF, WebP
          </p>
        </div>
        <input
          type="file"
          accept="image/png,image/jpeg,image/jpg,image/gif,image/webp"
          multiple
          onChange={onFileSelect}
          className="hidden"
        />
      </label>
    </div>
  );
}
