export const maxDuration = 120;

export async function POST(request: Request) {
  const apiUrl = process.env.RAG_AGENT_API_URL;
  if (!apiUrl) return Response.json({ error: "Сервис не настроен" }, { status: 503 });

  // Stream-read with a byte bound: content-length is optional and untrusted.
  const reader = request.body?.getReader();
  if (!reader) return Response.json({ error: "Пустой запрос" }, { status: 400 });
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > 128_000) {
        await reader.cancel();
        return Response.json({ error: "Запрос слишком большой" }, { status: 413 });
      }
      chunks.push(value);
    }
  } catch {
    return Response.json({ error: "Не удалось прочитать запрос" }, { status: 400 });
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  let body;
  try { body = JSON.parse(new TextDecoder().decode(bytes)); }
  catch { return Response.json({ error: "Некорректный JSON" }, { status: 400 }); }

  try {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (process.env.RAG_AGENT_API_KEY) headers["X-API-Key"] = process.env.RAG_AGENT_API_KEY;
    const upstream = await fetch(`${apiUrl}/agent/ask`, {
      method: "POST", headers, body: JSON.stringify(body),
      signal: AbortSignal.any([request.signal, AbortSignal.timeout(110_000)]),
      cache: "no-store",
    });
    if (!upstream.ok) {
      const status = [422, 429, 503].includes(upstream.status) ? upstream.status : 502;
      return Response.json({ error: status === 429 ? "Слишком много запросов. Попробуйте позже." : "Сервис временно недоступен или запрос некорректен." },
        { status, headers: { "Retry-After": upstream.headers.get("Retry-After") ?? "10" } });
    }
    return Response.json(await upstream.json());
  } catch {
    return Response.json({ error: "Сервис не успел ответить. Попробуйте позже." }, { status: 504 });
  }
}
