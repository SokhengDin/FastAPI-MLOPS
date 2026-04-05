import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata = {
  title: {
    default  : "Irrigation MLOps Dashboard- SokhengDin",
    template : "%s | Irrigation MLOps",
  },
  description  : "ML-powered irrigation prediction dashboard. Get real-time irrigation need forecasts — High, Medium, or Low — using LightGBM, XGBoost, and CatBoost models trained on soil, crop, and weather features.",
  keywords     : ["irrigation prediction", "MLOps", "machine learning", "smart farming", "precision agriculture", "soil moisture", "crop management", "LightGBM", "XGBoost", "CatBoost"],
  authors      : [{ name: "SokhengDin", url: "https://github.com/SokhengDin" }],
  creator      : "SokhengDin",
  source       : "https://github.com/SokhengDin/FastAPI-MLOPS/tree/main",
  robots       : { index: true, follow: true },
  openGraph: {
    type        : "website",
    title       : "Irrigation MLOps Dashboard — SokhengDin",
    description : "Real-time ML-powered irrigation need predictions by SokhengDin. Configure soil, crop, and environment parameters to get instant forecasts.",
    siteName    : "Irrigation MLOps by SokhengDin",
    locale      : "en_US",
  },
  twitter: {
    card        : "summary_large_image",
    title       : "Irrigation MLOps Dashboard — SokhengDin",
    description : "Real-time ML-powered irrigation need predictions for smart farming by SokhengDin.",
    creator     : "@SokhengDin",
  },
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable}`}>
      <head>
        <link rel="source" href="https://github.com/SokhengDin/FastAPI-MLOPS/tree/main" />
      </head>
      <body className="min-h-screen bg-[var(--background)] text-[var(--text-primary)] antialiased">
        {children}
      </body>
    </html>
  );
}
