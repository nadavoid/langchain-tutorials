/**
 * Usage:
 * yarn tsx local-rap-battle.ts
 */
import { ChatOllama } from "@langchain/ollama";
// The current default model for ChatOllama is llama3.
// Install it with `ollama run llama3`.
// View other installed models with `ollama list`.
const ollamaLlm = new ChatOllama({
  model: 'mistral',
  // model: 'llama3.2',
});
const response = await ollamaLlm.invoke(
  "Imagine a simulated rap battle between Stephen Colber and John Oliver, but keep the full text to yourself. \
  I only want to know who the winner is, and why they won, in one sentence."
);
console.log({rapResponse: response.content});
