import { ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { CORAL_DATASET_URL } from "@/lib/dataset";

export function DatasetFooter() {
  return (
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
  );
}
