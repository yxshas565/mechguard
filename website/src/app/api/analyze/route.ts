
import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST(request: Request) {
  const backend = process.env.MECHGUARD_API_URL;

  if (!backend) {
    return NextResponse.json(
      {
        ok: false,
        code: "BACKEND_NOT_CONFIGURED",
        error:
          "MechGuard analysis backend is not connected. Configure MECHGUARD_API_URL before using live analysis.",
      },
      { status: 503 }
    );
  }

  try {
    const contentType = request.headers.get("content-type") || "";
    const target = `${backend.replace(/\/$/, "")}/analyze`;

    let response: Response;

    if (contentType.includes("multipart/form-data")) {
      const form = await request.formData();
      const forwarded = new FormData();

      for (const [key, value] of form.entries()) {
        forwarded.append(key, value);
      }

      response = await fetch(target, {
        method: "POST",
        body: forwarded,
        cache: "no-store",
      });
    } else {
      const body = await request.text();

      response = await fetch(target, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body,
        cache: "no-store",
      });
    }

    const text = await response.text();

    let data: unknown;
    try {
      data = JSON.parse(text);
    } catch {
      data = { raw: text };
    }

    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        code: "BACKEND_UNREACHABLE",
        error:
          error instanceof Error
            ? error.message
            : "Unable to reach the MechGuard analysis backend.",
      },
      { status: 502 }
    );
  }
}
