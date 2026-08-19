import { useId, type ReactNode } from "react";

/**
 * A plate: one sheet pinned to the drafting table.
 *
 * The figure number is passed in rather than handed out by a counter. An
 * auto-incrementing context looked tidier but numbered by mount order, so a
 * plate that appeared later in the page could carry a lower number than one
 * above it — the first build printed fig. 1 followed by fig. 3. A caption
 * that cannot be trusted to be in order is worse than no caption.
 *
 * The caption text itself is always generated from the data: what the figure
 * shows, and how much of it had to be left out.
 */
export function Plate({
  figure,
  title,
  eyebrow,
  caption,
  actions,
  children,
  className = "",
}: {
  figure: number;
  title: string;
  eyebrow?: string;
  caption?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  const headingId = useId();
  return (
    <section className={`plate ${className}`} aria-labelledby={headingId}>
      <header className="plate-head">
        <div className="min-w-0">
          {eyebrow ? <div className="eyebrow">{eyebrow}</div> : null}
          <h2 id={headingId} className="plate-title truncate">
            {title}
          </h2>
        </div>
        {actions ? (
          <div className="flex items-center gap-2">{actions}</div>
        ) : null}
      </header>
      <div className="pt-3">{children}</div>
      {caption ? (
        <p className="plate-caption pt-3">
          fig. {figure} — {caption}
        </p>
      ) : null}
    </section>
  );
}
