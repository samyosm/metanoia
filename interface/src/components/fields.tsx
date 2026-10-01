import type { ComponentProps, ReactNode } from "react";

/** Carbon-style form primitives: a label above, a filled field with a bottom rule. */

const field =
  "w-full bg-field border-b border-border-strong px-4 text-text-primary placeholder:text-text-helper disabled:opacity-50";

export function Label({
  htmlFor,
  children,
}: {
  htmlFor: string;
  children: ReactNode;
}) {
  return (
    <label
      htmlFor={htmlFor}
      className="mb-2 block text-xs tracking-[0.32px] text-text-secondary"
    >
      {children}
    </label>
  );
}

export function TextInput(props: ComponentProps<"input">) {
  return <input type="text" {...props} className={`${field} h-10`} />;
}

export function TextArea(props: ComponentProps<"textarea">) {
  return (
    <textarea {...props} className={`${field} min-h-32 resize-y py-2.75`} />
  );
}

export function Select(props: ComponentProps<"select">) {
  return <select {...props} className={`${field} h-10 cursor-pointer`} />;
}

export function PrimaryButton(props: ComponentProps<"button">) {
  return (
    <button
      {...props}
      className="h-12 min-w-40 bg-interactive px-4 text-left text-text-on-color hover:bg-interactive-hover disabled:cursor-not-allowed disabled:bg-border-strong"
    />
  );
}

export function GhostButton(props: ComponentProps<"button">) {
  return (
    <button
      type="button"
      {...props}
      className="h-10 px-4 text-interactive hover:bg-layer disabled:text-text-helper dark:text-[#78a9ff]"
    />
  );
}

export function Heading({ children }: { children: ReactNode }) {
  return (
    <h2 className="mb-4 text-base font-normal text-text-primary">{children}</h2>
  );
}
