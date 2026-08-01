import { env } from "cloudflare:workers";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type ModelResponse = {
  choices?: Array<{ message?: { content?: string } }>;
  error?: { message?: string };
};

const SYSTEM_PROMPT = `你是“珞珈行旅”，一个专业、可靠的企业差旅规划助手。你的目标是像资深差旅顾问一样，把用户的自然语言需求转化为详细、可执行的方案。

当前能力边界：
- 可以基于用户提供的信息和常识完成行程规划、交通与住宿选择建议、预算拆分、差旅准备清单和风险提醒。
- 可以在当前对话中记住用户刚刚表达的偏好，并在后续回答中主动应用。
- 无法实时查询票价、余票、天气、酒店库存或企业内部制度原文。涉及实时信息时必须明确标注“建议预订前核验”，不得编造具体班次、价格或公司制度。

回答规则：
1. 用户要求规划行程时，即使信息不完整，也先基于合理假设给出有价值的完整方案，并在结尾列出待确认信息，不要只反问。
2. 详细行程必须按“行程概览、逐日安排、交通建议、住宿建议、预算参考、出行清单、风险与待确认事项”组织。
3. 每天至少给出上午、午间、下午、晚间四个时间段；商务行程要包含通勤缓冲、会议准备和复盘时间。
4. 结合用户明确偏好，如高铁、安静酒店、靠窗座位、预算等；不要擅自声称已经完成真实预订或永久保存偏好。
5. 追问必须结合此前对话继续细化，避免重复上一轮的概括。
6. 采用清晰的中文纯文本与分点格式，不输出 JSON，不使用 Markdown 表格。
7. 回答简洁但信息密度高，常规详细规划控制在 800—1400 字。

今天是 ${new Date().toLocaleDateString("zh-CN", { timeZone: "Asia/Shanghai" })}。`;

const requestLog = new Map<string, number[]>();

function isRateLimited(request: Request) {
  const key =
    request.headers.get("cf-connecting-ip") ||
    request.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ||
    "anonymous";
  const now = Date.now();
  const windowStart = now - 10 * 60 * 1000;
  const recent = (requestLog.get(key) || []).filter(
    (timestamp) => timestamp > windowStart,
  );
  if (recent.length >= 8) return true;
  recent.push(now);
  requestLog.set(key, recent);
  return false;
}

function cleanMessages(value: unknown): ChatMessage[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter(
      (item): item is ChatMessage =>
        Boolean(
          item &&
            typeof item === "object" &&
            (item.role === "user" || item.role === "assistant") &&
            typeof item.content === "string",
        ),
    )
    .slice(-12)
    .map((item) => ({
      role: item.role,
      content: item.content.trim().slice(0, 3000),
    }))
    .filter((item) => item.content.length > 0);
}

async function callModel(
  url: string,
  apiKey: string,
  model: string,
  messages: ChatMessage[],
) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 55_000);

  try {
    return await fetch(`${url.replace(/\/$/, "")}/chat/completions`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model,
        messages: [{ role: "system", content: SYSTEM_PROMPT }, ...messages],
        temperature: 0.65,
        max_tokens: 3200,
        thinking: { type: "disabled" },
      }),
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timeout);
  }
}

export async function POST(request: Request) {
  try {
    if (isRateLimited(request)) {
      return Response.json(
        { error: "体验请求较多，请十分钟后再试。" },
        { status: 429 },
      );
    }

    const runtimeEnv = env as unknown as Record<string, string | undefined>;
    const apiKey = runtimeEnv.ZHIPUAI_API_KEY;
    const model = runtimeEnv.ZHIPUAI_MODEL || "glm-4.7";
    const baseUrl =
      runtimeEnv.ZHIPUAI_BASE_URL || "https://open.bigmodel.cn/api/paas/v4";

    if (!apiKey) {
      return Response.json(
        { error: "模型服务尚未配置，请联系项目维护者。" },
        { status: 503 },
      );
    }

    const payload = (await request.json()) as { messages?: unknown };
    const messages = cleanMessages(payload.messages);
    if (!messages.length || messages[messages.length - 1].role !== "user") {
      return Response.json({ error: "请输入有效的差旅问题。" }, { status: 400 });
    }

    let response = await callModel(baseUrl, apiKey, model, messages);
    if (response.status === 429 || response.status >= 500) {
      await new Promise((resolve) => setTimeout(resolve, 800));
      response = await callModel(baseUrl, apiKey, model, messages);
    }

    const data = (await response.json()) as ModelResponse;
    if (!response.ok) {
      throw new Error(data.error?.message || `模型服务返回 ${response.status}`);
    }

    const answer = data.choices?.[0]?.message?.content?.trim();
    if (!answer) throw new Error("模型未返回有效内容");

    return Response.json({
      answer,
      model,
      detail: `智谱 ${model} 实时回答 · 已结合当前对话上下文`,
    });
  } catch (error) {
    const timedOut = error instanceof Error && error.name === "AbortError";
    return Response.json(
      {
        error: timedOut
          ? "本次规划用时较长，请稍后重试。"
          : "模型暂时没有成功响应，请稍后再试。",
      },
      { status: timedOut ? 504 : 502 },
    );
  }
}
