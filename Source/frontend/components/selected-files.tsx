"use client";

import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";

interface SelectedFilesProps {
  files: File[];
  previewUrls: string[];
  isClassifying: boolean;
  classificationProgress: string | null;
  onClassify: () => void;
  onClear: () => void;
}

export function SelectedFiles({
  files,
  previewUrls,
  isClassifying,
  classificationProgress,
  onClassify,
  onClear,
}: SelectedFilesProps) {
  if (!files.length) {
    return null;
  }

  return (
    <div className="mt-6 w-full space-y-2">
      <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
        {files.length} file{files.length !== 1 ? "s" : ""} selected
      </p>
      <div className="flex flex-wrap gap-3">
        {files.map((file, index) => (
          <div
            key={`${file.name}-${index}`}
            className="flex items-center gap-2 rounded-lg border border-gray-200 bg-gray-50 p-2 dark:border-gray-700 dark:bg-gray-800"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={previewUrls[index]}
              alt={file.name}
              className="size-12 rounded object-cover"
            />
            <span className="max-w-[120px] truncate text-sm text-gray-700 dark:text-gray-300">
              {file.name}
            </span>
          </div>
        ))}
      </div>
      <div className="flex gap-2">
        <Button
          variant="outline"
          size="sm"
          className="mt-2"
          onClick={onClassify}
          disabled={isClassifying}
        >
          {isClassifying ? (
            <>
              <Loader2 className="mr-2 size-4 animate-spin" />
              {classificationProgress
                ? `Classifying ${classificationProgress}`
                : "Queueing..."}
            </>
          ) : (
            "Classify"
          )}
        </Button>
        <Button
          variant="outline"
          size="sm"
          className="mt-2"
          onClick={onClear}
        >
          Clear
        </Button>
      </div>
    </div>
  );
}
