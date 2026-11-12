/** Pure codecs between System One questions and provider wire formats. */
import type { Prose, Question } from "./model.ts";

export interface SystemOneBody {
  readonly model: string;
  readonly state: unknown;
  readonly questions: Readonly<Record<string, Question>>;
}

const isRecord = (x: unknown): x is Readonly<Record<string, unknown>> =>
  typeof x === "object" && x !== null && !Array.isArray(x);

export const serializeState = (state: unknown): string => (typeof state === "string" ? state : JSON.stringify(state));

/** Keeps the context before a structured question, so its backtick references still resolve. */
export const proseToString = (prose: Prose): string => {
  if (typeof prose === "string") return prose;
  if (
    isRecord(prose) &&
    (Object.getPrototypeOf(prose) === Object.prototype || Object.getPrototypeOf(prose) === null) &&
    typeof prose["question"] === "string"
  ) {
    const { question, ...rest } = prose;
    return Object.keys(rest).length === 0 ? question : `${JSON.stringify(rest)}\n\n${question}`;
  }
  return JSON.stringify(prose);
};

export const noulInstructions = (q: Extract<Question, { type: "noul" }>): string => {
  const lines = [proseToString(q.instructions)];
  if (q.criteria?.true !== undefined) lines.push(`True when: ${proseToString(q.criteria.true)}`);
  if (q.criteria?.false !== undefined) lines.push(`False when: ${proseToString(q.criteria.false)}`);
  return lines.join("\n");
};

/** Maps question or option ids to unique names accepted by both providers. */
export const buildKeyMap = (
  ids: readonly string[],
): { readonly toWire: ReadonlyMap<string, string>; readonly fromWire: ReadonlyMap<string, string> } => {
  const toWire = new Map<string, string>();
  const fromWire = new Map<string, string>();
  const used = new Set<string>();
  for (const id of ids) {
    let base = id.replace(/[^A-Za-z0-9_-]+/g, "_");
    if (!/^[A-Za-z]/.test(base)) base = `q_${base}`;
    base = base.slice(0, 56);
    let candidate = base;
    for (let suffix = 2; used.has(candidate); suffix++) candidate = `${base}_${suffix}`;
    used.add(candidate);
    toWire.set(id, candidate);
    fromWire.set(candidate, id);
  }
  return { toWire, fromWire };
};

const maxProbability = (probabilities: Readonly<Record<string, unknown>>): number =>
  Math.max(0, ...Object.values(probabilities).filter((p): p is number => typeof p === "number"));

/** Restores choice option ids; unknown options remain for the answer validator to reject. */
const unwrapChoice = (raw: Readonly<Record<string, unknown>>, q: Extract<Question, { type: "choice" }>) => {
  if (!isRecord(raw["probabilities"])) return null;
  const optMap = buildKeyMap(Object.keys(q.criteria));
  const selected = typeof raw["choice"] === "string" ? raw["choice"] : raw["selected"];
  const choice = typeof selected === "string" ? (optMap.fromWire.get(selected) ?? selected) : selected;
  const probabilities = Object.fromEntries(
    Object.entries(raw["probabilities"]).map(([k, p]) => [optMap.fromWire.get(k) ?? k, p]),
  );
  const confidence = typeof raw["confidence"] === "number" ? raw["confidence"] : maxProbability(probabilities);
  return { type: "choice", choice, probabilities, confidence };
};

const scoreProbabilities = (raw: unknown): Readonly<Record<string, unknown>> | null =>
  Array.isArray(raw) ? Object.fromEntries(raw.map((p, i) => [String(i), p])) : isRecord(raw) ? raw : null;

type GeminiQuestion =
  | { readonly type: "noul"; readonly instructions: string }
  | {
      readonly type: "choice" | "score";
      readonly instructions: string;
      readonly criteria: Readonly<Record<string, string>>;
    };

export const encodeOpenAIRequest = (body: SystemOneBody) => {
  const keyMap = buildKeyMap(Object.keys(body.questions));
  return {
    model: body.model,
    input: serializeState(body.state),
    questions: Object.entries(body.questions).map(([id, q]) => {
      const name = keyMap.toWire.get(id)!;
      if (q.type === "noul") return { type: "predicate", name, instructions: noulInstructions(q) };
      if (q.type === "choice") {
        const optMap = buildKeyMap(Object.keys(q.criteria));
        return {
          type: "choice",
          name,
          instructions: proseToString(q.instructions),
          choices: Object.entries(q.criteria).map(([k, desc]) =>
            desc === null
              ? { value: optMap.toWire.get(k)! }
              : { value: optMap.toWire.get(k)!, description: proseToString(desc) },
          ),
        };
      }
      return {
        type: "score",
        name,
        instructions: proseToString(q.instructions),
        levels: q.criteria.map((desc, i) => ({ label: String(i), description: proseToString(desc) })),
      };
    }),
  };
};

export const unwrapOpenAIResponse = (json: unknown, body: SystemOneBody): unknown => {
  if (!isRecord(json) || !Array.isArray(json["answers"])) return null;
  const keyMap = buildKeyMap(Object.keys(body.questions));
  const answers: [string, unknown][] = [];
  for (const item of json["answers"]) {
    if (!isRecord(item) || typeof item["name"] !== "string") return null;
    const origId = keyMap.fromWire.get(item["name"]);
    if (origId === undefined) return null;
    const q = body.questions[origId];
    if (item["type"] === "predicate" || item["type"] === "noul") {
      const noul = typeof item["probability"] === "number" ? item["probability"] : item["noul"];
      answers.push([origId, { type: "noul", noul }]);
    } else if (item["type"] === "choice" && q?.type === "choice") {
      const answer = unwrapChoice(item, q);
      if (answer === null) return null;
      answers.push([origId, answer]);
    } else if (item["type"] === "score" && q?.type === "score") {
      const raw = scoreProbabilities(item["probabilities"]);
      if (raw === null) return null;
      const labels = q.criteria.map(proseToString);
      const indices = new Set(q.criteria.map((_, i) => String(i)));
      const probabilities = Object.fromEntries(
        Object.entries(raw).map(([k, p]) => {
          const index = labels.indexOf(k);
          const unique = index !== -1 && labels.lastIndexOf(k) === index;
          const label = indices.has(k) ? k : unique ? String(index) : k;
          return [label, p];
        }),
      );
      const confidence = typeof item["confidence"] === "number" ? item["confidence"] : maxProbability(probabilities);
      answers.push([origId, { type: "score", score: item["score"], probabilities, confidence }]);
    } else return null;
  }
  const usage = isRecord(json["usage"]) ? json["usage"]["input_tokens"] : undefined;
  const model = typeof json["model"] === "string" ? json["model"] : body.model;
  return { model, answers: Object.fromEntries(answers), usage: { input_tokens: usage } };
};

export const encodeGeminiRequest = (body: SystemOneBody) => {
  const keyMap = buildKeyMap(Object.keys(body.questions));
  return {
    model: body.model,
    input: serializeState(body.state),
    questions: Object.fromEntries(
      Object.entries(body.questions).map(([id, q]): [string, GeminiQuestion] => {
        const wireId = keyMap.toWire.get(id)!;
        if (q.type === "noul") return [wireId, { type: "noul", instructions: noulInstructions(q) }];
        if (q.type === "choice") {
          const optMap = buildKeyMap(Object.keys(q.criteria));
          return [
            wireId,
            {
              type: "choice",
              instructions: proseToString(q.instructions),
              criteria: Object.fromEntries(
                Object.entries(q.criteria).map(([k, desc]) => {
                  const text = desc === null ? "" : proseToString(desc);
                  return [optMap.toWire.get(k)!, text === "" ? k : text];
                }),
              ),
            },
          ];
        }
        return [
          wireId,
          {
            type: "score",
            instructions: proseToString(q.instructions),
            criteria: Object.fromEntries(
              q.criteria.map((desc, i) => {
                const text = proseToString(desc);
                return [String(i), text === "" ? String(i) : text];
              }),
            ),
          },
        ];
      }),
    ),
  };
};

export const unwrapGeminiResponse = (json: unknown, body: SystemOneBody): unknown => {
  if (!isRecord(json) || !isRecord(json["answers"])) return null;
  const keyMap = buildKeyMap(Object.keys(body.questions));
  const answers: [string, unknown][] = [];
  for (const [wireId, raw] of Object.entries(json["answers"])) {
    const origId = keyMap.fromWire.get(wireId);
    if (origId === undefined || !isRecord(raw)) return null;
    const q = body.questions[origId];
    if (raw["type"] === "noul") answers.push([origId, { type: "noul", noul: raw["noul"] }]);
    else if (raw["type"] === "choice" && q?.type === "choice") {
      const answer = unwrapChoice(raw, q);
      if (answer === null) return null;
      answers.push([origId, answer]);
    } else if (raw["type"] === "score" && q?.type === "score") {
      const probabilities = scoreProbabilities(raw["probabilities"]);
      if (probabilities === null) return null;
      const confidence = typeof raw["confidence"] === "number" ? raw["confidence"] : maxProbability(probabilities);
      answers.push([origId, { type: "score", score: raw["score"], probabilities, confidence }]);
    } else return null;
  }
  const model =
    typeof json["model"] === "string"
      ? json["model"]
      : typeof json["modelVersion"] === "string"
        ? json["modelVersion"]
        : body.model;
  const usage = isRecord(json["usage"]) ? json["usage"] : {};
  const metadata = isRecord(json["usageMetadata"]) ? json["usageMetadata"] : {};
  const inputTokens = usage["total_input_tokens"] ?? usage["input_tokens"] ?? metadata["promptTokenCount"];
  return { model, answers: Object.fromEntries(answers), usage: { input_tokens: inputTokens } };
};
