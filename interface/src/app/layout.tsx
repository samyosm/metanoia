import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import { Nav } from "@/components/nav";
import "./globals.css";

const plexSans = IBM_Plex_Sans({
  variable: "--font-plex-sans",
  subsets: ["latin"],
  weight: ["300", "400", "600"],
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400"],
});

export const metadata: Metadata = {
  title: "Metanoia",
  description: "Candidate probabilities from a local Metanoia predictor.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${plexSans.variable} ${plexMono.variable} antialiased`}
    >
      <body className="min-h-dvh font-sans text-sm">
        <header className="flex h-12 items-center border-b border-[#393939] bg-[#161616] px-4 text-[#f4f4f4]">
          <span className="font-semibold">Metanoia</span>
          <span className="ml-2 text-[#c6c6c6]">Decision workbench</span>
          <Nav />
        </header>
        {children}
      </body>
    </html>
  );
}
