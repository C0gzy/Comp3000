import Link from "next/link";
import { Button } from "@/components/ui/button";

export function DocsLink() {
  return (
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
  );
}
