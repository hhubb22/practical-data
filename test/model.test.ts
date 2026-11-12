import { test } from "node:test";
import assert from "node:assert/strict";
import {
  DecisionError,
  gemini,
  hashJson,
  hosted,
  modelFromConfig,
  openai,
  validateAnswers,
  type AnswerCache,
  type DecisionResponse,
  type Question,
} from "../src/decide/model.ts";

const questions: Record<string, Question> = {
  role: { type: "choice", instructions: "?", criteria: { measure: null, identifier: null } },
  log: { type: "noul", instructions: "?" },
};
const ok = {
  model: "jev",
  usage: { input_tokens: 1 },
  answers: {
    role: {
      type: "choice" as const,
      choice: "measure",
      probabilities: { measure: 0.9, identifier: 0.1 },
      confidence: 0.8,
    },
    log: { type: "noul" as const, noul: 0.3 },
  },
};

/** The DecisionError that `promise` rejects with. Fails the test if it resolves or rejects with anything else. */
const rejection = (promise: Promise<unknown>): Promise<DecisionError> =>
  promise.then(
    () => assert.fail("expected a DecisionError"),
    (e: unknown) => {
      assert.ok(e instanceof DecisionError, String(e));
      return e;
    },
  );

test("accepts answers that match the questions asked", () => {
  assert.equal(validateAnswers(questions, ok), null);
});

test("rejects missing answers, wrong types, unknown options and out-of-range probabilities", () => {
  assert.equal(validateAnswers(questions, { ...ok, answers: { role: ok.answers.role } }), "missing answer for log");
  assert.equal(
    validateAnswers(questions, { ...ok, answers: { ...ok.answers, log: ok.answers.role } }),
    "log: expected noul, got choice",
  );
  assert.equal(
    validateAnswers(questions, {
      ...ok,
      answers: { ...ok.answers, role: { ...ok.answers.role, probabilities: { measure: 0.9, rank: 0.1 } } },
    }),
    "role: unexpected option rank",
  );
  assert.equal(
    validateAnswers(questions, { ...ok, answers: { ...ok.answers, log: { type: "noul", noul: 2 } } }),
    "log: noul 2 out of range",
  );
});

test("a spent hosted allowance fails at once as quota, without waiting or retrying", async () => {
  let calls = 0;
  const realFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    calls++;
    return new Response('{"error":"install_limit"}', {
      status: 429,
      headers: { "Retry-After": "40000", "X-Practical-Data-Quota": "install_limit" },
    });
  };
  try {
    const started = Date.now();
    const error = await rejection(hosted("id").evaluate({ state: {}, questions }));
    assert.equal(error.reason, "quota");
    assert.equal(calls, 1);
    assert.ok(Date.now() - started < 1000);
  } finally {
    globalThis.fetch = realFetch;
  }
});

test("a spent allowance for everyone is told apart from this install's own", async () => {
  const realFetch = globalThis.fetch;
  globalThis.fetch = async () =>
    new Response('{"error":"budget_spent"}', {
      status: 429,
      headers: {
        "X-Practical-Data-Quota": "budget_spent",
        "X-Practical-Data-Remaining": "250",
        "X-Practical-Data-Limit": "400",
      },
    });
  try {
    const error = await rejection(hosted("id").evaluate({ state: {}, questions: { q: questions["log"]! } }));
    assert.equal(error.reason, "quota_everyone");
  } finally {
    globalThis.fetch = realFetch;
  }
});

test("hosted responses report what's left of the allowance; cache hits report nothing", async () => {
  const heard: unknown[] = [];
  const realFetch = globalThis.fetch;
  let hit = false;
  globalThis.fetch = async () =>
    new Response(JSON.stringify(ok), {
      headers: hit ? { "X-Cache": "hit" } : { "X-Practical-Data-Remaining": "73", "X-Practical-Data-Limit": "400" },
    });
  try {
    const model = hosted("id", { onAllowance: (a) => heard.push(a) });
    await model.evaluate({ state: { n: 1 }, questions });
    hit = true;
    await model.evaluate({ state: { n: 2 }, questions });
    assert.deepEqual(heard, [{ remaining: 73, limit: 400 }]);
  } finally {
    globalThis.fetch = realFetch;
  }
});

/** Counts the fetch calls that one evaluate makes when every response is a 503. */
const callsOn503 = async (provider: "hosted" | "jev"): Promise<number> => {
  const { hosted, jev } = await import("../src/decide/model.ts");
  let calls = 0;
  const realFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    calls++;
    return new Response('{"error":"upstream"}', { status: 503 });
  };
  try {
    const model = provider === "hosted" ? hosted("id") : jev("key");
    const error = await rejection(model.evaluate({ state: {}, questions }));
    assert.equal(error.reason, "overloaded");
    return calls;
  } finally {
    globalThis.fetch = realFetch;
  }
};

test("the hosted endpoint retries a 503 only once, because each attempt spends from the allowance", async () => {
  assert.equal(await callsOn503("hosted"), 2);
});

test("Jev retries a 503 four times", async () => {
  assert.equal(await callsOn503("jev"), 5);
});

const memoryCache = () => {
  const entries = new Map<string, DecisionResponse>();
  const cache: AnswerCache = {
    get: async (key) => entries.get(key),
    set: async (key, value) => {
      entries.set(key, value);
    },
  };
  return { entries, cache };
};

test("OpenAI sends authenticated Decisions requests in 400-question chunks and caches normalized answers", async (t) => {
  const { entries, cache } = memoryCache();
  const sizes: number[] = [];
  const fetch = t.mock.method(globalThis, "fetch", async (url: string | URL | Request, init?: RequestInit) => {
    assert.equal(url, "https://api.openai.com/v1/decisions");
    assert.equal(init?.method, "POST");
    const headers = new Headers(init?.headers);
    assert.equal(headers.get("Authorization"), "Bearer sk-test");
    assert.equal(headers.get("Content-Type"), "application/json");
    assert.ok(init?.signal instanceof AbortSignal);
    const body = JSON.parse(String(init?.body)) as {
      model: string;
      input: string;
      questions: { type: string; name: string; choices?: { value: string; description?: string }[] }[];
    };
    assert.equal(body.model, "gpt-6-luna");
    assert.equal(body.input, '{"page":"test"}');
    assert.ok(body.questions.length <= 400);
    sizes.push(body.questions.length);
    return new Response(
      JSON.stringify({
        model: "gpt-6-luna-version",
        usage: { input_tokens: 13 },
        answers: body.questions.map((q) => {
          assert.doesNotMatch(q.name, /\./);
          if (q.type === "predicate") return { type: "predicate", name: q.name, probability: 0.8 };
          assert.deepEqual(q.choices, [{ value: "columns" }, { value: "rows", description: "By row" }]);
          return { type: "choice", name: q.name, choice: "rows", probabilities: { columns: 0.1, rows: 0.9 } };
        }),
      }),
    );
  });
  const questions: Record<string, Question> = {
    ...Object.fromEntries(
      Array.from({ length: 401 }, (_, i) => [`n${i}.log`, { type: "noul", instructions: "Use ratios?" }]),
    ),
    "table.orientation": {
      type: "choice",
      instructions: "Rows or columns?",
      criteria: { columns: null, rows: "By row" },
    },
  };
  const request = { state: { page: "test" }, questions };
  const model = openai("sk-test", { cache });
  const first = await model.evaluate(request);
  assert.deepEqual(
    sizes.sort((a, b) => a - b),
    [2, 400],
  );
  assert.equal(fetch.mock.callCount(), 2);
  assert.equal(first.model, "gpt-6-luna-version");
  assert.deepEqual(first.usage, { input_tokens: 26 });
  assert.equal(Object.keys(first.answers).length, 402);
  assert.deepEqual(first.answers["n400.log"], { type: "noul", noul: 0.8 });
  assert.deepEqual(first.answers["table.orientation"], {
    type: "choice",
    choice: "rows",
    probabilities: { columns: 0.1, rows: 0.9 },
    confidence: 0.9,
  });
  assert.equal(entries.size, 2);
  assert.ok(
    entries.has(
      await hashJson({
        model: "gpt-6-luna",
        state: request.state,
        questions: Object.fromEntries(Object.entries(questions).slice(0, 400)),
      }),
    ),
  );
  for (const response of entries.values())
    assert.equal(
      validateAnswers(Object.fromEntries(Object.keys(response.answers).map((id) => [id, questions[id]!])), response),
      null,
    );
  fetch.mock.resetCalls();
  assert.deepEqual(await model.evaluate(request), first);
  assert.equal(fetch.mock.callCount(), 0);
});

test("Gemini uses an API key header, 500-question chunks and caches normalized answers and token usage", async (t) => {
  const { entries, cache } = memoryCache();
  const sizes: number[] = [];
  const fetch = t.mock.method(globalThis, "fetch", async (url: string | URL | Request, init?: RequestInit) => {
    assert.equal(url, "https://generativelanguage.googleapis.com/v1beta/decisions");
    assert.doesNotMatch(String(url), /gm-test|\?/);
    assert.equal(init?.method, "POST");
    const headers = new Headers(init?.headers);
    assert.equal(headers.get("x-goog-api-key"), "gm-test");
    assert.equal(headers.get("Content-Type"), "application/json");
    assert.ok(init?.signal instanceof AbortSignal);
    const body = JSON.parse(String(init?.body)) as {
      model: string;
      input: string;
      questions: Record<string, { type: string }>;
    };
    assert.equal(body.model, "gemini-decision-1");
    assert.equal(body.input, "table text");
    const qs = Object.entries(body.questions);
    assert.ok(qs.length <= 500);
    sizes.push(qs.length);
    return new Response(
      JSON.stringify({
        modelVersion: "gemini-version",
        usage: { total_input_tokens: 17 },
        answers: Object.fromEntries(
          qs.map(([id, q]) => {
            assert.doesNotMatch(id, /\./);
            return [
              id,
              q.type === "noul"
                ? { type: "noul", noul: 0.7 }
                : { type: "choice", selected: "rows", probabilities: { columns: 0.2, rows: 0.8 } },
            ];
          }),
        ),
      }),
    );
  });
  const questions: Record<string, Question> = {
    ...Object.fromEntries(
      Array.from({ length: 501 }, (_, i) => [`row${i}.aggregate`, { type: "noul", instructions: "A total?" }]),
    ),
    "table.orientation": { type: "choice", instructions: "Rows or columns?", criteria: { columns: null, rows: null } },
  };
  const request = { state: "table text", questions };
  const model = gemini("gm-test", { cache });
  const first = await model.evaluate(request);
  assert.deepEqual(
    sizes.sort((a, b) => a - b),
    [2, 500],
  );
  assert.equal(fetch.mock.callCount(), 2);
  assert.equal(first.model, "gemini-version");
  assert.deepEqual(first.usage, { input_tokens: 34 });
  assert.equal(Object.keys(first.answers).length, 502);
  assert.deepEqual(first.answers["row500.aggregate"], { type: "noul", noul: 0.7 });
  assert.deepEqual(first.answers["table.orientation"], {
    type: "choice",
    choice: "rows",
    probabilities: { columns: 0.2, rows: 0.8 },
    confidence: 0.8,
  });
  assert.equal(entries.size, 2);
  assert.ok(
    entries.has(
      await hashJson({
        model: "gemini-decision-1",
        state: request.state,
        questions: Object.fromEntries(Object.entries(questions).slice(0, 500)),
      }),
    ),
  );
  fetch.mock.resetCalls();
  assert.deepEqual(await model.evaluate(request), first);
  assert.equal(fetch.mock.callCount(), 0);
});

test("modelFromConfig selects OpenAI and Gemini and forwards model overrides and caches", async (t) => {
  assert.equal(modelFromConfig({ provider: "openai", apiKey: "key" }).name, "gpt-6-luna");
  assert.equal(modelFromConfig({ provider: "gemini", apiKey: "key" }).name, "gemini-decision-1");
  const fetch = t.mock.method(globalThis, "fetch", async (_url: string | URL | Request, init?: RequestInit) => {
    const body = JSON.parse(String(init?.body)) as { model: string };
    return new Response(
      JSON.stringify({
        usage: { input_tokens: 1 },
        answers:
          body.model === "custom-openai"
            ? [{ type: "predicate", name: "n0_log", probability: 0.6 }]
            : { n0_log: { type: "noul", noul: 0.6 } },
      }),
    );
  });
  const { cache } = memoryCache();
  for (const provider of ["openai", "gemini"] as const) {
    const model = modelFromConfig({ provider, apiKey: "key", model: `custom-${provider}` }, cache);
    const request = { state: {}, questions: { "n0.log": questions["log"]! } };
    const response = await model.evaluate(request);
    assert.equal(response.model, `custom-${provider}`);
    assert.deepEqual(response.answers["n0.log"], { type: "noul", noul: 0.6 });
    assert.deepEqual(await model.evaluate(request), response);
  }
  assert.equal(fetch.mock.callCount(), 2);
});

test("models of one provider share the eight-request concurrency limit", async (t) => {
  let active = 0;
  let peak = 0;
  t.mock.method(globalThis, "fetch", async () => {
    active++;
    peak = Math.max(peak, active);
    await new Promise((resolve) => setTimeout(resolve, 5));
    active--;
    return new Response(
      JSON.stringify({ usage: { input_tokens: 1 }, answers: [{ type: "predicate", name: "log", probability: 0.5 }] }),
    );
  });
  await Promise.all(
    Array.from({ length: 12 }, (_, i) =>
      openai("key", { model: `model-${i}` }).evaluate({ state: {}, questions: { log: questions["log"]! } }),
    ),
  );
  assert.equal(peak, 8);
});

test("provider errors retain the selected model name", async (t) => {
  t.mock.method(globalThis, "fetch", async () => new Response("invalid", { status: 400 }));
  const error = await rejection(openai("key", { model: "custom-openai" }).evaluate({ state: {}, questions }));
  assert.equal(error.reason, "bad_request");
  assert.equal(error.message, "custom-openai 400: invalid");
});
