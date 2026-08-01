"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

type Message = {
  role: "assistant" | "user";
  content: string;
  detail?: string;
};

const samples = [
  "我明天从武汉去北京出差3天，偏好高铁和安静酒店",
  "公司出差住宿标准是多少？",
  "记住我以后优先选靠窗座位",
];

const capabilityItems = [
  {
    icon: "⌁",
    title: "智能行程规划",
    text: "从自然语言中提取城市、日期、时长与偏好，生成可执行的差旅行程。",
    tag: "Itinerary Agent",
  },
  {
    icon: "◎",
    title: "企业制度问答",
    text: "通过本地 BGE 向量检索与 RAG 回答差旅制度，并返回可追溯来源。",
    tag: "RAG Agent",
  },
  {
    icon: "◌",
    title: "偏好长期记忆",
    text: "区分本次约束与长期偏好，支持新增、追加、覆盖和历史查询。",
    tag: "Memory Agent",
  },
  {
    icon: "↗",
    title: "多 Agent 编排",
    text: "按优先级分批执行，同优先级任务并行，复杂请求一次完成。",
    tag: "Orchestration",
  },
];

const metrics = [
  { value: "99.68%", label: "意图识别 Macro F1", note: "60 条冻结集 × 3 轮" },
  { value: "100%", label: "RAG Recall@5", note: "60 条正式评测集" },
  { value: "100%", label: "正确拒答率", note: "人工逐条复核" },
  { value: "45.8%", label: "Agent P50 提速", note: "40 组配对实验" },
];

const flow = [
  ["01", "理解需求", "IntentionAgent 识别多意图与关键实体"],
  ["02", "拆解任务", "生成 Agent 调度计划与优先级"],
  ["03", "并行执行", "制度、偏好、信息与事件 Agent 同批运行"],
  ["04", "生成方案", "Itinerary Agent 汇总上下文输出行程"],
];

export default function Home() {
  const [input, setInput] = useState("");
  const [qrOpen, setQrOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "你好，我是珞珈行旅。告诉我出发地、目的地和时间，我可以帮你规划差旅，也可以查询企业制度或记住你的偏好。",
      detail: "GLM-4.7 实时服务 · 支持多轮上下文",
    },
  ]);
  const [working, setWorking] = useState(false);

  const lastMessage = useMemo(
    () => messages[messages.length - 1],
    [messages],
  );

  useEffect(() => {
    if (!qrOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setQrOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [qrOpen]);

  async function submitQuery(query: string) {
    const clean = query.trim();
    if (!clean || working) return;
    const nextMessages: Message[] = [
      ...messages,
      { role: "user", content: clean },
    ];
    setInput("");
    setWorking(true);
    setMessages(nextMessages);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: nextMessages.map(({ role, content }) => ({ role, content })),
        }),
      });
      const data = (await response.json()) as {
        answer?: string;
        detail?: string;
        error?: string;
      };

      if (!response.ok || !data.answer) {
        throw new Error(data.error || "模型暂时没有成功响应，请稍后再试。");
      }

      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: data.answer!,
          detail: data.detail || "模型实时回答",
        },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content:
            error instanceof Error
              ? error.message
              : "模型暂时没有成功响应，请稍后再试。",
          detail: "本次请求未完成 · 可重新发送",
        },
      ]);
    } finally {
      setWorking(false);
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    submitQuery(input);
  }

  return (
    <main>
      <nav className="nav" aria-label="主导航">
        <a className="brand" href="#top" aria-label="珞珈行旅首页">
          <span className="logo-shell">
            <img src="/whu-logo.png" alt="武汉大学校徽" />
          </span>
          <span>
            <strong>珞珈行旅</strong>
            <small>智能差旅助手</small>
          </span>
        </a>
        <div className="nav-links">
          <a href="#experience">产品体验</a>
          <a href="#capabilities">核心能力</a>
          <a href="#evidence">评测证据</a>
          <a href="#architecture">技术架构</a>
        </div>
        <div className="nav-actions">
          <button className="qr-trigger" type="button" onClick={() => setQrOpen(true)}>
            扫码体验
          </button>
          <a className="nav-cta" href="#experience">
            立即体验 <span>↗</span>
          </a>
        </div>
      </nav>

      <section className="hero" id="top">
        <div className="route route-one" aria-hidden="true" />
        <div className="route route-two" aria-hidden="true" />
        <div className="hero-copy">
          <div className="eyebrow">
            <span className="eyebrow-dot" />
            武汉大学实验室 Agent 项目
          </div>
          <h1>
            让每一次出差
            <br />
            <em>从容抵达</em>
          </h1>
          <p className="hero-lead">
            一个会理解、会协作、会记忆的多智能体差旅助手。
            <br />
            从一句需求到一份合规、个性化的完整行程。
          </p>
          <div className="hero-actions">
            <a className="primary-button" href="#experience">
              体验智能规划 <span>→</span>
            </a>
            <button className="text-button qr-text-button" type="button" onClick={() => setQrOpen(true)}>
              手机扫码体验 <span>⌗</span>
            </button>
          </div>
          <div className="trust-row">
            <div>
              <strong>6</strong>
              <span>业务 Agent</span>
            </div>
            <i />
            <div>
              <strong>2</strong>
              <span>层记忆系统</span>
            </div>
            <i />
            <div>
              <strong>3</strong>
              <span>套冻结评测集</span>
            </div>
          </div>
        </div>

        <div className="hero-visual" aria-label="差旅规划示意">
          <div className="stamp">WUH → PEK</div>
          <div className="plane">✈</div>
          <div className="map-card">
            <div className="map-head">
              <div>
                <small>TRIP PLAN · 03 DAYS</small>
                <strong>武汉 → 北京</strong>
              </div>
              <span>已规划</span>
            </div>
            <div className="map-line">
              <span className="city city-a">武汉</span>
              <div className="rail">
                <i />
                <b>G66 · 3h 58m</b>
                <i />
              </div>
              <span className="city city-b">北京</span>
            </div>
            <div className="trip-grid">
              <div>
                <span>8月 04日</span>
                <strong>09:00</strong>
                <small>武汉站出发</small>
              </div>
              <div>
                <span>8月 06日</span>
                <strong>18:30</strong>
                <small>北京南返程</small>
              </div>
            </div>
            <div className="preference">
              <span>☼</span>
              <p>
                <small>已应用个人偏好</small>
                高铁优先 · 安静酒店 · 靠窗座位
              </p>
              <b>✓</b>
            </div>
          </div>
          <div className="float-card float-memory">
            <span>◉</span>
            <div>
              <small>记忆已生效</small>
              <strong>偏好自动匹配</strong>
            </div>
          </div>
          <div className="float-card float-policy">
            <span>⌕</span>
            <div>
              <small>企业制度</small>
              <strong>合规校验通过</strong>
            </div>
          </div>
        </div>
      </section>

      <section className="experience section-pad" id="experience">
        <div className="section-heading">
          <div>
            <span className="section-index">01 / 产品体验</span>
            <h2>把需求交给协作中的 Agent</h2>
          </div>
          <p>
            输入一句自然语言，观察系统如何识别意图、
            <br />
            调度多个 Agent 并组织最终答复。
          </p>
        </div>

        <div className="demo-frame">
          <div className="demo-side">
            <div className="assistant-id">
              <span>旅</span>
              <div>
                <strong>珞珈行旅</strong>
                <small>
                  <i /> GLM 模型在线
                </small>
              </div>
            </div>
            <div className="side-label">试试这样问</div>
            <div className="sample-list">
              {samples.map((sample, index) => (
                <button
                  key={sample}
                  type="button"
                  onClick={() => submitQuery(sample)}
                >
                  <span>0{index + 1}</span>
                  {sample}
                </button>
              ))}
            </div>
            <div className="agent-stack">
              <small>当前可用 Agent</small>
              <div className="agent-icons">
                {["意", "知", "忆", "偏", "事", "程"].map((item) => (
                  <span key={item}>{item}</span>
                ))}
                <b>6</b>
              </div>
            </div>
          </div>

          <div className="chat-panel" aria-live="polite">
            <div className="chat-top">
              <div>
                <span />
                <span />
                <span />
              </div>
              <small>智能差旅对话 · LIVE</small>
              <button
                type="button"
                onClick={() =>
                  setMessages([
                    {
                      role: "assistant",
                      content:
                        "你好，我是珞珈行旅。告诉我出发地、目的地和时间，我可以帮你规划差旅，也可以查询企业制度或记住你的偏好。",
                      detail: "GLM-4.7 实时服务 · 支持多轮上下文",
                    },
                  ])
                }
              >
                清空
              </button>
            </div>
            <div className="chat-scroll">
              {messages.map((message, index) => (
                <div
                  className={`message ${message.role}`}
                  key={`${message.role}-${index}`}
                >
                  {message.role === "assistant" && (
                    <span className="message-avatar">旅</span>
                  )}
                  <div>
                    <p>{message.content}</p>
                    {message.detail && <small>{message.detail}</small>}
                  </div>
                </div>
              ))}
              {working && (
                <div className="message assistant">
                  <span className="message-avatar">旅</span>
                  <div className="thinking">
                    <span />
                    <span />
                    <span />
                    正在理解需求并生成详细方案…
                  </div>
                </div>
              )}
            </div>
            <form className="chat-input" onSubmit={handleSubmit}>
              <input
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder="输入你的差旅需求…"
                aria-label="输入差旅问题"
              />
              <button
                type="submit"
                aria-label="发送问题"
                disabled={!input.trim() || working}
              >
                ↑
              </button>
            </form>
            <div className="chat-foot">
              <span>↵ Enter 发送</span>
              <span>
                最后响应：{lastMessage.role === "assistant" ? "已完成" : "处理中"}
              </span>
            </div>
          </div>
        </div>
      </section>

      <section className="capabilities section-pad" id="capabilities">
        <div className="section-heading inverse">
          <div>
            <span className="section-index">02 / 核心能力</span>
            <h2>不是一个模型，<br />而是一支会协作的团队</h2>
          </div>
          <p>
            每个 Agent 专注一个领域，由编排中枢统一理解、
            <br />
            分工与汇总，降低复杂任务的处理成本。
          </p>
        </div>
        <div className="capability-grid">
          {capabilityItems.map((item, index) => (
            <article key={item.title}>
              <div className="cap-num">0{index + 1}</div>
              <span className="cap-icon">{item.icon}</span>
              <h3>{item.title}</h3>
              <p>{item.text}</p>
              <small>{item.tag}</small>
            </article>
          ))}
        </div>
      </section>

      <section className="evidence section-pad" id="evidence">
        <div className="section-heading">
          <div>
            <span className="section-index">03 / 评测证据</span>
            <h2>用冻结数据集回答“做得怎么样”</h2>
          </div>
          <p>
            正式结果来自独立冻结测试集、重复运行与人工逐条复核，
            <br />
            不以单次演示替代产品可靠性证据。
          </p>
        </div>
        <div className="metrics-grid">
          {metrics.map((metric, index) => (
            <article key={metric.label}>
              <span>0{index + 1}</span>
              <strong>{metric.value}</strong>
              <h3>{metric.label}</h3>
              <p>{metric.note}</p>
            </article>
          ))}
        </div>
        <div className="evidence-note">
          <span>✓</span>
          <p>
            <strong>14 / 14 离线回归测试通过</strong>
            覆盖核心编排、意图规则、RAG 序列化、重试与错误透传
          </p>
          <small>评测口径可追溯</small>
        </div>
      </section>

      <section className="architecture section-pad" id="architecture">
        <div className="arch-intro">
          <span className="section-index">04 / 技术架构</span>
          <h2>从一句话，到一份可执行行程</h2>
          <p>
            核心设计不是让大模型“一次猜完”，而是将复杂需求拆成有顺序、
            有依赖、可追溯的 Agent 工作流。
          </p>
        </div>
        <div className="flow-grid">
          {flow.map((item, index) => (
            <article key={item[0]}>
              <div className="flow-line">
                <span>{item[0]}</span>
                {index < flow.length - 1 && <i />}
              </div>
              <h3>{item[1]}</h3>
              <p>{item[2]}</p>
            </article>
          ))}
        </div>
        <div className="tech-row" aria-label="技术栈">
          <span>智谱 GLM</span>
          <i>×</i>
          <span>AgentScope</span>
          <i>×</i>
          <span>BGE Embedding</span>
          <i>×</i>
          <span>Python</span>
          <i>×</i>
          <span>RAG</span>
        </div>
      </section>

      <footer>
        <div className="footer-brand">
          <img src="/whu-logo.png" alt="" />
          <div>
            <strong>珞珈行旅</strong>
            <small>让每一次出差，从容抵达</small>
          </div>
        </div>
        <p>武汉大学实验室多智能体项目 · 个人作品集展示</p>
        <a href="#top">返回顶部 ↑</a>
      </footer>

      {qrOpen && (
        <div className="qr-overlay" role="presentation" onMouseDown={() => setQrOpen(false)}>
          <section
            className="qr-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="qr-title"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <button
              className="qr-close"
              type="button"
              aria-label="关闭二维码"
              onClick={() => setQrOpen(false)}
            >
              ×
            </button>
            <div className="qr-dialog-copy">
              <span className="section-index">INTERVIEW EXPERIENCE</span>
              <h2 id="qr-title">扫码，体验珞珈行旅</h2>
              <p>无需安装、无需登录。用手机扫码后，即可直接向 GLM 差旅助手提问。</p>
              <div className="qr-suggestion">
                <small>推荐体验问题</small>
                <strong>我明天从武汉去北京出差3天，偏好高铁和安静酒店，请给出详细规划。</strong>
              </div>
              <div className="qr-steps">
                <span><b>01</b> 手机扫码</span>
                <span><b>02</b> 输入需求</span>
                <span><b>03</b> 继续追问</span>
              </div>
            </div>
            <div className="qr-code-panel">
              <div className="qr-image-shell">
                <img src="/experience-qr.png" alt="珞珈行旅扫码体验二维码" />
              </div>
              <strong>珞珈行旅 · LIVE</strong>
              <small>真实 GLM API · 支持多轮对话</small>
              <a href="/experience-qr.png" download="珞珈行旅-扫码体验.png">
                下载二维码 ↓
              </a>
            </div>
          </section>
        </div>
      )}
    </main>
  );
}
