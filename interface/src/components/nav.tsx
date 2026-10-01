"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const TABS = [
  { href: "/", label: "Try it" },
  { href: "/trials", label: "What we tried" },
];

export function Nav() {
  const pathname = usePathname();
  return (
    <nav aria-label="Sections" className="ml-8 flex h-full">
      {TABS.map((tab) => {
        const current = pathname === tab.href;
        return (
          <Link
            key={tab.href}
            href={tab.href}
            aria-current={current ? "page" : undefined}
            className={`flex h-full items-center border-b-2 px-4 ${
              current
                ? "border-white text-white"
                : "border-transparent text-[#c6c6c6] hover:bg-[#262626]"
            }`}
          >
            {tab.label}
          </Link>
        );
      })}
    </nav>
  );
}
