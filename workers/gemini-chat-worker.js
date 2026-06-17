const GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models";
const DEFAULT_MODEL = "gemini-3.5-flash";
const DEFAULT_FALLBACK_MODELS = ["gemini-2.5-flash"];

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
        .map(normalizeOrigin)
        .filter(Boolean);
}

function normalizeOrigin(origin) {
    return String(origin || "").trim().replace(/\/+$/, "");
}

function getCorsHeaders(request, env) {
    const requestOrigin = normalizeOrigin(request.headers.get("Origin"));
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

function normalizeModel(model) {
    return String(model || "").replace(/^models\//, "").trim();
}

function getGeminiModels(env) {
    const primaryModel = normalizeModel(env.GEMINI_MODEL || DEFAULT_MODEL);
    const fallbackModels = String(env.GEMINI_FALLBACK_MODELS || DEFAULT_FALLBACK_MODELS.join(","))
        .split(",")
        .map(normalizeModel)
        .filter(Boolean);

    return [...new Set([primaryModel, ...fallbackModels])].filter(Boolean);
}

function getGeminiUrl(model) {
    return `${GEMINI_API_BASE}/${encodeURIComponent(model)}:generateContent`;
}

function buildGeminiPayload(body, systemInstruction) {
    return {
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
    };
}

async function readErrorDetail(response) {
    const text = await response.text();
    if (!text) return "";

    try {
        const data = JSON.parse(text);
        return data?.error?.message || data?.message || text.slice(0, 500);
    } catch {
        return text.slice(0, 500);
    }
}

function canTryNextModel(status) {
    return [429, 500, 502, 503, 504].includes(status);
}

async function requestGemini(env, body, systemInstruction) {
    const payload = buildGeminiPayload(body, systemInstruction);
    let lastError = null;

    for (const model of getGeminiModels(env)) {
        let response;

        try {
            response = await fetch(getGeminiUrl(model), {
                method: "POST",
                headers: {
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "x-goog-api-key": env.GEMINI_API_KEY
                },
                body: JSON.stringify(payload)
            });
        } catch (error) {
            lastError = {
                error: "Gemini fetch failed",
                model,
                detail: error instanceof Error ? error.message : String(error)
            };
            continue;
        }

        if (response.ok) {
            const data = await response.json();
            return { data, model };
        }

        lastError = {
            error: "Gemini request failed",
            model,
            status: response.status,
            detail: await readErrorDetail(response)
        };

        if (!canTryNextModel(response.status)) {
            break;
        }
    }

    throw lastError || { error: "Gemini request failed" };
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

        let geminiResult;
        try {
            geminiResult = await requestGemini(env, body, systemInstruction);
        } catch (error) {
            return jsonResponse(error, 502, cors.headers);
        }

        const geminiData = geminiResult.data;
        const answer = readGeminiText(geminiData);

        if (!answer) {
            return jsonResponse({
                error: "Gemini returned no answer",
                model: geminiResult.model
            }, 502, cors.headers);
        }

        return jsonResponse({ answer, model: geminiResult.model }, 200, cors.headers);
    }
};
