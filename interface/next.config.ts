import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // A self-contained server (.next/standalone) for the Docker image.
  output: "standalone",
  allowedDevOrigins: ["*.trycloudflare.com"],
};

export default nextConfig;
