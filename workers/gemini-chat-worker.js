const GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models";
const DEFAULT_MODEL = "gemini-3.5-flash";

function jsonResponse(data, status, headers) {
    return new Response(JSON.stringify(data), {
        status,
        headers: {
            ...headers,
            "Content-Type": "application/json; charset=UTF-8"
        }
    });
}

function getAllowedOrigins(env) {
    return String(env.ALLOWED_ORIGIN || "")
        .split(",")
        .map((origin) => origin.trim())
        .filter(Boolean);
}

function getCorsHeaders(request, env) {
    const requestOrigin = request.headers.get("Origin") || "";
    const allowedOrigins = getAllowedOrigins(env);
    const allowAll = allowedOrigins.length === 0 || allowedOrigins.includes("*");
    const isAllowed = allowAll || allowedOrigins.includes(requestOrigin);

    return {
        isAllowed,
        headers: {
            "Access-Control-Allow-Origin": allowAll ? "*" : requestOrigin,
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "86400",
            "Vary": "Origin"
        }
    };
}

function readGeminiText(data) {
    const parts = data?.candidates?.[0]?.content?.parts;
    if (!Array.isArray(parts)) return "";

    return parts
        .map((part) => typeof part.text === "string" ? part.text : "")
        .join("")
        .trim();
}

function normalizeHistory(history) {
    if (!Array.isArray(history)) return "";

    return history
        .slice(-8)
        .map((item) => {
            const role = item?.role === "user" ? "Customer" : "Assistant";
            const content = String(item?.content || "").slice(0, 1200);
            return content ? `${role}: ${content}` : "";
        })
        .filter(Boolean)
        .join("\n");
}

function buildPrompt(body) {
    const context = String(body.context || "").slice(0, 20000);
    const history = normalizeHistory(body.history);
    const message = String(body.message || "").slice(0, 3000);
    const contact = body.contact || {};

    return [
        `CONTEXT.MD:\n${context}`,
        history ? `RECENT CHAT HISTORY:\n${history}` : "",
        `CURRENT CUSTOMER QUESTION:\n${message}`,
        `CONTACT:\nHotline: ${contact.hotlineText || contact.hotline || ""}\nZalo: ${contact.zaloText || contact.zalo || ""}`,
        "Answer the customer directly in Vietnamese. Do not invent facts outside CONTEXT.MD. If the customer needs urgent rescue, prioritize asking them to call the hotline or send their location via Zalo."
    ].filter(Boolean).join("\n\n");
}

function getGeminiUrl(env) {
    const model = String(env.GEMINI_MODEL || DEFAULT_MODEL).replace(/^models\//, "").trim();
    return `${GEMINI_API_BASE}/${encodeURIComponent(model)}:generateContent`;
}

export default {
    async fetch(request, env) {
        const cors = getCorsHeaders(request, env);

        if (request.method === "OPTIONS") {
            return new Response(null, { status: 204, headers: cors.headers });
        }

        if (!cors.isAllowed) {
            return jsonResponse({ error: "Origin is not allowed" }, 403, cors.headers);
        }

        if (request.method !== "POST") {
            return jsonResponse({ error: "Method not allowed" }, 405, cors.headers);
        }

        if (!env.GEMINI_API_KEY) {
            return jsonResponse({ error: "Gemini API key is not configured" }, 500, cors.headers);
        }

        let body;
        try {
            body = await request.json();
        } catch {
            return jsonResponse({ error: "Invalid JSON body" }, 400, cors.headers);
        }

        const message = String(body.message || "").trim();
        if (!message) {
            return jsonResponse({ error: "Message is required" }, 400, cors.headers);
        }

        const systemInstruction = String(body.instructions || [
            "You are the chat assistant for a 24/7 road rescue service in Ninh Binh, Vietnam.",
            "Answer in Vietnamese, briefly and clearly.",
            "Use only the provided context. If unsure, ask the customer to call the hotline."
        ].join("\n"));

        const geminiResponse = await fetch(getGeminiUrl(env), {
            method: "POST",
            headers: {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "x-goog-api-key": env.GEMINI_API_KEY
            },
            body: JSON.stringify({
                system_instruction: {
                    parts: [{ text: systemInstruction }]
                },
                contents: [
                    {
                        role: "user",
                        parts: [{ text: buildPrompt(body) }]
                    }
                ],
                generationConfig: {
                    temperature: 0.35,
                    maxOutputTokens: 700
                }
            })
        });

        if (!geminiResponse.ok) {
            return jsonResponse({
                error: "Gemini request failed",
                status: geminiResponse.status
            }, 502, cors.headers);
        }

        const geminiData = await geminiResponse.json();
        const answer = readGeminiText(geminiData);

        if (!answer) {
            return jsonResponse({ error: "Gemini returned no answer" }, 502, cors.headers);
        }

        return jsonResponse({ answer }, 200, cors.headers);
    }
};
