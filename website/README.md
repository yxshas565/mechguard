# MechGuard — Website Frontend & Prototype API

This directory contains the Next.js web application for MechGuard, implementing the product interface and prototype API route.

---

## 1. Stack & Architecture

- **Framework**: Next.js 16 (App Router)
- **Library**: React 19, TypeScript
- **Styling**: Tailwind CSS v4, Custom CSS (`src/app/globals.css`)
- **Icons**: `lucide-react`
- **Build System**: Next Build / Turbopack

---

## 2. Quickstart — Running Locally

1. **Install Dependencies**:
   ```bash
   cd website
   npm install
   ```

2. **Start Development Server**:
   ```bash
   npm run dev
   ```
   Open **[http://localhost:3000](http://localhost:3000)** in your browser.

3. **Verify Production Build**:
   ```bash
   npm run build
   ```

---

## 3. Directory Structure

```text
website/
├── src/
│   └── app/
│       ├── page.tsx             ← Primary product UI & interactive workspace
│       ├── globals.css          ← Entire visual design system & animations
│       ├── layout.tsx           ← Next.js root layout & metadata
│       └── api/
│           └── analyze/
│               └── route.ts     ← Prototype API endpoint (POST /api/analyze)
├── public/
│   └── evidence/                ← Persisted research CSV & JSON artifacts
├── package.json                 ← Node dependencies & scripts
├── next.config.ts               ← Next.js configuration
├── postcss.config.mjs           ← PostCSS configuration
├── tsconfig.json                ← TypeScript configuration
└── README.md
```

---

## 4. Prototype API Mechanics (`/api/analyze`)

The interactive workspace communicates with the Next.js backend API at `POST /api/analyze`.

### Request Payload:
```json
{
  "stage": "attest" | "watch" | "review",
  "input": "User payload text or artifact contents"
}
```

### Response Behavior:
- **`attest`**: Reads `public/evidence/a001_geometry_timeseries.csv` and `a001_monitor_results.json`, returning A001 geometry measurements (45 checkpoints, +735.9% top singular value growth) with `live: false`.
- **`watch`**: Returns B001 and B003 multi-agent representation probing evidence with `live: false`.
- **`review`**: Aggregates recorded signals into an engineering review payload, explicitly noting that no automated safety verdict is generated.

> [!NOTE]
> The current API endpoint returns **persisted research evidence**. It does not perform live foundation model GPU inference in the browser.
