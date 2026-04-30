import Image from "next/image";

export function AppBackground() {
  return (
    <div className="absolute inset-0 -z-10">
      <Image
        src="/background.jpg"
        fill
        sizes="100vw"
        alt="Underwater coral reef"
        className="object-cover"
      />
    </div>
  );
}
