import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Credit Card Fraud Detection — Demo",
  description: "Live demo of the trained fraud detection model.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
