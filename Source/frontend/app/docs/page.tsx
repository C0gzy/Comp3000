import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ArrowLeft, BookOpen, ExternalLink } from "lucide-react";
import type { Metadata } from "next";
import { CORAL_DATASET_URL } from "@/lib/dataset";

const API_BASE = "http://localhost:5001";

export const metadata: Metadata = {
  title: "API documentation | Coral Reef Health Monitor",
  description:
    "REST endpoints for health checks, async classification jobs, and job status for the coral reef classifier.",
};

const endpoints = [
  {
    method: "GET",
    path: "/V1/api/health",
    description:
      "Returns whether the service is running and responding. Use this for readiness or uptime checks.",
    response: "200 — JSON: { status: 'success' }",
  },
  {
    method: "GET",
    path: "/V1/api/status/{JOB_ID}",
    description:
      "Returns the progress of the classification job identified by JOB_ID (the value returned when you started a job). Progress is expressed as a percentage string.",
    response: "200 — JSON: { progress: '20%' } (example)",
  },
  {
    method: "POST",
    path: "/V1/api/classify",
    description:
      "Upload an image (multipart form field name: image). The server queues classification and responds immediately with a job ID. Poll GET /V1/api/status/{JOB_ID} until progress reaches 100%.",
    response: "200 — JSON: { job_id: 1052 } (example job ID)",
  },
  {
    method: "DELETE",
    path: "/V1/api/classify/{JOB_ID}",
    description:
      "Cancels and removes the job for JOB_ID. Use this if the client no longer needs the result or is abandoning the upload flow.",
    response: "200 — JSON: { status: 'success' }",
  },
] as const;

export default function DocsPage() {
  return (
    <div className="relative min-h-screen font-sans dark:bg-black">
      <Button
        variant="outline"
        size="sm"
        className="fixed top-4 right-4 z-20 border-gray-200 bg-white/95 shadow-sm backdrop-blur-sm dark:border-zinc-700 dark:bg-zinc-900/95"
        asChild
      >
        <Link href="/" className="no-underline">
          <ArrowLeft className="size-4" />
          Back to app
        </Link>
      </Button>
      <div className="mx-auto max-w-4xl px-4 py-10 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">
            Coral classifier API
          </h1>
          <p className="mt-2 max-w-2xl text-gray-600 dark:text-gray-400">
            Base URL for local development:{" "}
            <code className="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-sm dark:bg-zinc-800">
              {API_BASE}
            </code>
            . All paths below are appended to that base (for example,{" "}
            <code className="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-sm dark:bg-zinc-800">
              {API_BASE}/V1/api/health
            </code>
            ).
          </p>
        </div>

        <div className="overflow-hidden rounded-xl border border-gray-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <table className="w-full border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-gray-200 bg-gray-50 dark:border-zinc-800 dark:bg-zinc-950/80">
                <th className="px-4 py-3 font-semibold text-gray-900 dark:text-white">
                  Method
                </th>
                <th className="px-4 py-3 font-semibold text-gray-900 dark:text-white">
                  Route
                </th>
                <th className="hidden px-4 py-3 font-semibold text-gray-900 md:table-cell dark:text-white">
                  Definition and use
                </th>
                <th className="px-4 py-3 font-semibold text-gray-900 dark:text-white">
                  Expected response
                </th>
              </tr>
            </thead>
            <tbody>
              {endpoints.map((row) => (
                <tr
                  key={row.path + row.method}
                  className="border-b border-gray-100 last:border-0 dark:border-zinc-800"
                >
                  <td className="align-top px-4 py-4">
                    <span className="inline-flex rounded-md bg-gray-100 px-2 py-0.5 font-mono text-xs font-semibold text-gray-800 dark:bg-zinc-800 dark:text-gray-200">
                      {row.method}
                    </span>
                  </td>
                  <td className="align-top px-4 py-4 font-mono text-xs text-gray-800 dark:text-gray-200">
                    {row.path}
                  </td>
                  <td className="hidden align-top px-4 py-4 text-gray-600 md:table-cell dark:text-gray-400">
                    {row.description}
                  </td>
                  <td className="align-top px-4 py-4 text-gray-800 dark:text-gray-200">
                    {row.response}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-4 space-y-3 md:hidden">
          <p className="text-xs font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-500">
            Definitions (mobile)
          </p>
          {endpoints.map((row) => (
            <div
              key={`m-${row.path}`}
              className="rounded-lg border border-gray-200 bg-white/80 p-4 text-sm text-gray-600 dark:border-zinc-800 dark:bg-zinc-900/80 dark:text-gray-400"
            >
              <p className="font-mono text-xs text-gray-800 dark:text-gray-200">
                {row.method} {row.path}
              </p>
              <p className="mt-2">{row.description}</p>
            </div>
          ))}
        </div>

        <section className="mt-12 border-t border-gray-200 pt-10 dark:border-zinc-800">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            Training data
          </h2>
          <p className="mt-2 max-w-2xl text-sm text-gray-600 dark:text-gray-400">
            The model is trained on imagery from the NOAA PIFSC ESD coral bleaching
            dataset.
          </p>
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <Button asChild>
              <a
                href={CORAL_DATASET_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="no-underline"
              >
                dataset
                <ExternalLink className="size-4" aria-hidden />
              </a>
            </Button>
          </div>
          <p className="mt-6 text-sm text-gray-600 dark:text-gray-400">
            <span className="font-medium text-gray-800 dark:text-gray-200">
              Reference:
            </span>{" "}
            NMFS-OSI. NOAA PIFSC ESD Coral Bleaching Dataset. Hugging Face,{" "}
            <a
              href={CORAL_DATASET_URL}
              className="text-sky-700 underline underline-offset-2 hover:text-sky-600 dark:text-sky-400 dark:hover:text-sky-300"
              target="_blank"
              rel="noopener noreferrer"
            >
              {CORAL_DATASET_URL}
            </a>
            .
          </p>
        </section>
      </div>
    </div>
  );
}
