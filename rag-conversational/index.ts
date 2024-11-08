/**
 * Usage:
 * yarn tsx index.ts
 */
import { ChatOpenAI } from "@langchain/openai";
import dotenv from "dotenv";
dotenv.config({
  path: '../.env',
});

const llm = new ChatOpenAI({
  model: "gpt-40-mini",
  temperature: 0,
});
