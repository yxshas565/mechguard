import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MechGuard — AI Safety, From the Inside",
  description:
    "Mechanistic AI safety monitoring across training and deployment.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
