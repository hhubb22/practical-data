import { test } from "node:test";
import assert from "node:assert/strict";
import { profileTable } from "../src/core/profile.ts";
import { decodeDecisionResponse, validateAnswers, type DecisionResponse } from "../src/decide/model.ts";
import {
  aggregateQuestions,
  categoricalQuestions,
  numericQuestions,
  orientationQuestion,
  shareQuestion,
} from "../src/decide/questions.ts";
import {
  buildKeyMap,
  encodeGeminiRequest,
  encodeOpenAIRequest,
  noulInstructions,
  proseToString,
  serializeState,
  unwrapGeminiResponse,
  unwrapOpenAIResponse,
  type SystemOneBody,
} from "../src/decide/wire.ts";

test("state and prose preserve strings and serialize structured context before its question", () => {
  assert.equal(serializeState("already text"), "already text");
  assert.equal(serializeState({ a: 1 }), '{"a":1}');
  assert.equal(serializeState(null), "null");
  assert.equal(proseToString("Which is lower?"), "Which is lower?");
  assert.equal(
    proseToString({ a: "Low", question: "Which of `a` and `b` is lower?", b: "High" }),
    '{"a":"Low","b":"High"}\n\nWhich of `a` and `b` is lower?',
  );
  assert.equal(proseToString({ question: "Which is lower?" }), "Which is lower?");
  assert.equal(proseToString(["Low", { question: "High?" }]), '["Low",{"question":"High?"}]');
  assert.equal(proseToString({ question: 3, a: 1 }), '{"question":3,"a":1}');
});

test("Noul instructions include each defined criterion", () => {
  assert.equal(noulInstructions({ type: "noul", instructions: "Is this an amount?" }), "Is this an amount?");
  assert.equal(
    noulInstructions({
      type: "noul",
      instructions: "Amount?",
      criteria: { true: { question: "A quantity." }, false: "" },
    }),
    "Amount?\nTrue when: A quantity.\nFalse when: ",
  );
  assert.equal(
    noulInstructions({ type: "noul", instructions: "Amount?", criteria: { false: "An identifier." } }),
    "Amount?\nFalse when: An identifier.",
  );
});

test("key maps restore dotted ids and disambiguate collisions in input order", () => {
  const ids = ["n0.role", "table.orientation", "c1.lt0_1", "a.b", "a_b", "a_b_2", "a b"];
  const map = buildKeyMap(ids);
  assert.deepEqual(
    [...map.toWire.values()],
    ["n0_role", "table_orientation", "c1_lt0_1", "a_b", "a_b_2", "a_b_2_2", "a_b_3"],
  );
  for (const id of ids) assert.equal(map.fromWire.get(map.toWire.get(id)!), id);
  assert.deepEqual(buildKeyMap(ids), map);
  assert.equal(
    buildKeyMap(["measure", "higher_is_better", "a", "b"]).toWire.get("higher_is_better"),
    "higher_is_better",
  );
});

test("key maps prefix non-letter starts and keep truncated names unique and wire-safe", () => {
  const long = "a".repeat(80);
  const ids = ["", "0", "_a", "-a", "é", "汉字", "a...b", long, `${long}b`];
  const map = buildKeyMap(ids);
  assert.equal(map.toWire.get(""), "q_");
  assert.equal(map.toWire.get("0"), "q_0");
  assert.equal(map.toWire.get("_a"), "q__a");
  assert.equal(map.toWire.get(long), "a".repeat(56));
  assert.equal(map.toWire.get(`${long}b`), `${"a".repeat(56)}_2`);
  for (const [id, wire] of map.toWire) {
    assert.match(wire, /^[A-Za-z][A-Za-z0-9_-]{0,63}$/);
    assert.equal(map.fromWire.get(wire), id);
  }
  assert.equal(new Set(map.toWire.values()).size, ids.length);
});

const body: SystemOneBody = {
  model: "test-decision",
  state: { table: "Values" },
  questions: {
    "n0.log": {
      type: "noul",
      instructions: "Use ratios?",
      criteria: { true: "Ratios matter.", false: "Differences matter." },
    },
    "n0.role": {
      type: "choice",
      instructions: { column: "Value", question: "What does `column` mean?" },
      criteria: { "a.b": "An amount", a_b: null, other: "" },
    },
    "n0.score": { type: "score", instructions: "How much?", criteria: ["Low", "High", ""] },
  },
};

const openAIReply = {
  model: "openai-version",
  usage: { input_tokens: 27 },
  answers: [
    { type: "predicate", name: "n0_log", probability: 0.8 },
    { type: "choice", name: "n0_role", selected: "a_b_2", probabilities: { a_b: 0.2, a_b_2: 0.7, other: 0.1 } },
    { type: "score", name: "n0_score", score: 1, probabilities: [0.1, 0.8, 0.1] },
  ],
};

const geminiReply = {
  modelVersion: "gemini-version",
  usage: { total_input_tokens: 27, input_tokens: 90 },
  usageMetadata: { promptTokenCount: 100 },
  answers: {
    n0_log: { type: "noul", noul: 0.8 },
    n0_role: { type: "choice", selected: "a_b_2", probabilities: { a_b: 0.2, a_b_2: 0.7, other: 0.1 } },
    n0_score: { type: "score", score: 1, probabilities: { "0": 0.1, "1": 0.8, "2": 0.1 } },
  },
};

const expectedAnswers: DecisionResponse["answers"] = {
  "n0.log": { type: "noul", noul: 0.8 },
  "n0.role": { type: "choice", choice: "a_b", probabilities: { "a.b": 0.2, a_b: 0.7, other: 0.1 }, confidence: 0.7 },
  "n0.score": { type: "score", score: 1, probabilities: { "0": 0.1, "1": 0.8, "2": 0.1 }, confidence: 0.8 },
};

const checked = (json: unknown, request = body): DecisionResponse => {
  const response = decodeDecisionResponse(json);
  assert.ok(response);
  assert.equal(validateAnswers(request.questions, response), null);
  return response;
};

test("OpenAI encodes predicates, choices and score levels and restores normalized answers", () => {
  assert.deepEqual(encodeOpenAIRequest(body), {
    model: "test-decision",
    input: '{"table":"Values"}',
    questions: [
      {
        type: "predicate",
        name: "n0_log",
        instructions: "Use ratios?\nTrue when: Ratios matter.\nFalse when: Differences matter.",
      },
      {
        type: "choice",
        name: "n0_role",
        instructions: '{"column":"Value"}\n\nWhat does `column` mean?',
        choices: [{ value: "a_b", description: "An amount" }, { value: "a_b_2" }, { value: "other", description: "" }],
      },
      {
        type: "score",
        name: "n0_score",
        instructions: "How much?",
        levels: [
          { label: "0", description: "Low" },
          { label: "1", description: "High" },
          { label: "2", description: "" },
        ],
      },
    ],
  });
  assert.deepEqual(checked(unwrapOpenAIResponse(openAIReply, body)), {
    model: "openai-version",
    answers: expectedAnswers,
    usage: { input_tokens: 27 },
  });
});

test("OpenAI accepts Noul aliases, explicit choice and confidence, and unique score descriptions", () => {
  const response = checked(
    unwrapOpenAIResponse(
      {
        usage: { input_tokens: 1 },
        answers: [
          { type: "noul", name: "n0_log", noul: 0.4 },
          {
            type: "choice",
            name: "n0_role",
            choice: "a_b",
            selected: "other",
            probabilities: { a_b: 1 },
            confidence: 0.9,
          },
          {
            type: "score",
            name: "n0_score",
            score: 0,
            probabilities: { Low: 0.75, High: 0.25, "2": 0 },
            confidence: 0.6,
          },
        ],
      },
      body,
    ),
  );
  assert.equal(response.model, body.model);
  assert.deepEqual(response.answers["n0.log"], { type: "noul", noul: 0.4 });
  assert.deepEqual(response.answers["n0.role"], {
    type: "choice",
    choice: "a.b",
    probabilities: { "a.b": 1 },
    confidence: 0.9,
  });
  assert.deepEqual(response.answers["n0.score"], {
    type: "score",
    score: 0,
    probabilities: { "0": 0.75, "1": 0.25, "2": 0 },
    confidence: 0.6,
  });
});

test("OpenAI leaves ambiguous score descriptions for validation", () => {
  const request: SystemOneBody = {
    ...body,
    questions: { score: { type: "score", instructions: "?", criteria: ["same", "same"] } },
  };
  const response = decodeDecisionResponse(
    unwrapOpenAIResponse(
      { usage: { input_tokens: 1 }, answers: [{ type: "score", name: "score", score: 0, probabilities: { same: 1 } }] },
      request,
    ),
  );
  assert.ok(response);
  assert.equal(validateAnswers(request.questions, response), "score: unexpected option same");
});

test("Gemini encodes criteria, substitutes empty descriptions and restores normalized answers", () => {
  assert.deepEqual(encodeGeminiRequest(body), {
    model: "test-decision",
    input: '{"table":"Values"}',
    questions: {
      n0_log: { type: "noul", instructions: "Use ratios?\nTrue when: Ratios matter.\nFalse when: Differences matter." },
      n0_role: {
        type: "choice",
        instructions: '{"column":"Value"}\n\nWhat does `column` mean?',
        criteria: { a_b: "An amount", a_b_2: "a_b", other: "other" },
      },
      n0_score: { type: "score", instructions: "How much?", criteria: { "0": "Low", "1": "High", "2": "2" } },
    },
  });
  assert.deepEqual(checked(unwrapGeminiResponse(geminiReply, body)), {
    model: "gemini-version",
    answers: expectedAnswers,
    usage: { input_tokens: 27 },
  });
});

test("Gemini accepts array score probabilities, explicit confidence and metadata fallbacks", () => {
  const response = checked(
    unwrapGeminiResponse(
      {
        model: "preferred-model",
        modelVersion: "other-version",
        usageMetadata: { promptTokenCount: 15 },
        answers: {
          ...geminiReply.answers,
          n0_role: { type: "choice", choice: "a_b", probabilities: { a_b: 1 }, confidence: 0.9 },
          n0_score: { type: "score", score: 1, probabilities: [0.1, 0.9, 0], confidence: 0.6 },
        },
      },
      body,
    ),
  );
  assert.equal(response.model, "preferred-model");
  assert.deepEqual(response.usage, { input_tokens: 15 });
  assert.deepEqual(response.answers["n0.role"], {
    type: "choice",
    choice: "a.b",
    probabilities: { "a.b": 1 },
    confidence: 0.9,
  });
  assert.deepEqual(response.answers["n0.score"], {
    type: "score",
    score: 1,
    probabilities: { "0": 0.1, "1": 0.9, "2": 0 },
    confidence: 0.6,
  });
  const fallback = checked(
    unwrapGeminiResponse(
      { answers: geminiReply.answers, usage: { input_tokens: 0 }, usageMetadata: { promptTokenCount: 15 } },
      body,
    ),
  );
  assert.equal(fallback.model, body.model);
  assert.deepEqual(fallback.usage, { input_tokens: 0 });
});

test("wire codecs reject unknown answers, malformed records and invalid probability containers", () => {
  for (const json of [
    null,
    [],
    {},
    { answers: {} },
    { answers: [null] },
    { answers: [{ name: "unknown", type: "predicate", probability: 1 }] },
    { answers: [{ name: "n0_log", type: "other" }] },
    { answers: [{ name: "n0_role", type: "choice", probabilities: [] }] },
    { answers: [{ name: "n0_score", type: "score", probabilities: null }] },
  ]) {
    assert.equal(unwrapOpenAIResponse(json, body), null);
  }
  for (const json of [
    null,
    [],
    {},
    { answers: [] },
    { answers: { n0_log: null } },
    { answers: { unknown: { type: "noul", noul: 1 } } },
    { answers: { n0_log: { type: "predicate", probability: 1 } } },
    { answers: { n0_role: { type: "choice", probabilities: [] } } },
    { answers: { n0_score: { type: "score", probabilities: null } } },
  ]) {
    assert.equal(unwrapGeminiResponse(json, body), null);
  }
  assert.equal(decodeDecisionResponse(unwrapOpenAIResponse({ ...openAIReply, usage: {} }, body)), null);
  assert.equal(decodeDecisionResponse(unwrapGeminiResponse({ answers: geminiReply.answers }, body)), null);
});

test("real table questions retain their context and round-trip through both providers", () => {
  const table = {
    id: "t0",
    title: "Values",
    headers: ["Amount", "Grade"],
    rows: [
      ["1", "Low"],
      ["10", "Medium"],
      ["20", "High"],
      ["30", "Low"],
      ["100", "Medium"],
    ],
  };
  const profiles = profileTable(table);
  const n = profiles[0]!;
  const c = profiles[1]!;
  assert.equal(n.kind, "numeric");
  assert.equal(c.kind, "categorical");
  if (n.kind !== "numeric" || c.kind !== "categorical") assert.fail("expected numeric and categorical profiles");
  const numeric = numericQuestions(0, n);
  const categorical = categoricalQuestions(c);
  const orientation = orientationQuestion({ header: "Amount", examples: ["1", "10"] });
  const aggregates = aggregateQuestions(table.headers, table.rows);
  const request: SystemOneBody = {
    model: "test",
    state: "table context",
    questions: {
      ...numeric.questions,
      ...orientation.questions,
      ...categorical.questions,
      ...categorical.pairQuestions,
      ...aggregates.questions,
      "g0.share": shareQuestion("quantitative", [
        { table: "First", header: "Amount", unit: "$", examples: ["1", "10"] },
        { table: "Second", header: "Amount", unit: "$", examples: ["20", "30"] },
      ]),
    },
  };
  const open = encodeOpenAIRequest(request);
  const gemini = encodeGeminiRequest(request);
  assert.equal(open.input, "table context");
  assert.equal(gemini.input, "table context");
  const instructions = Object.fromEntries(open.questions.map((q) => [q.name, q.instructions]));
  assert.match(instructions["n0_role"]!, /`columns\[0\]`/);
  assert.match(instructions["table_orientation"]!, /^\{"first_column":.*\n\n.*`first_column`/);
  assert.match(instructions["c1_lt0_1"]!, /^\{"a":"Low","b":"Medium"\}\n\n.*`a`.*`b`/);
  assert.match(instructions["row0_aggregate"]!, /^\{"row":.*\n\n.*`row`/);
  assert.match(instructions["g0_share"]!, /^\{"columns":.*\n\n.*`columns`/);
  const openResponse = checked(
    unwrapOpenAIResponse(
      {
        usage: { input_tokens: 12 },
        answers: open.questions.map((q) =>
          q.type === "predicate"
            ? { type: "predicate", name: q.name, probability: 0.8 }
            : {
                type: "choice",
                name: q.name,
                choice: q.choices![0]!.value,
                probabilities: Object.fromEntries(q.choices!.map((o, i) => [o.value, i === 0 ? 1 : 0])),
              },
        ),
      },
      request,
    ),
    request,
  );
  const geminiResponse = checked(
    unwrapGeminiResponse(
      {
        usage: { total_input_tokens: 12 },
        answers: Object.fromEntries(
          Object.entries(gemini.questions).map(([id, q]) => [
            id,
            q.type === "noul"
              ? { type: "noul", noul: 0.8 }
              : {
                  type: "choice",
                  selected: Object.keys(q.criteria!)[0],
                  probabilities: Object.fromEntries(Object.keys(q.criteria!).map((key, i) => [key, i === 0 ? 1 : 0])),
                },
          ]),
        ),
      },
      request,
    ),
    request,
  );
  assert.deepEqual(openResponse, geminiResponse);
  assert.deepEqual(Object.keys(openResponse.answers), Object.keys(request.questions));
  assert.equal(numeric.read(openResponse.answers).measure, 1);
  assert.equal(categorical.read(openResponse.answers).lower(0, 1), 0.5);
  assert.equal(orientation.read(openResponse.answers), 0);
  assert.deepEqual(aggregates.read(openResponse.answers, 0.5), [0, 1, 3, 4]);
});
