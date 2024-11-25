/**
 * Usage:
 * yarn tsx agent.ts
 */
import dotenv from "dotenv";
dotenv.config({
  path: '../.env',
});

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { createRetrieverTool } from "langchain/tools/retriever";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

// Basic setup.
const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
});
const loader = new CheerioWebBaseLoader(
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  {
    selector: ".post-content, .post-title, .post-header",
  }
);
const docs = await loader.load();
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const splits = await textSplitter.splitDocuments(docs);
const vectorstore = await MemoryVectorStore.fromDocuments(
  splits,
  new OpenAIEmbeddings()
);
const retriever = vectorstore.asRetriever();

// New for agents and tools.
const blogPostRetrieverTool = createRetrieverTool(retriever, {
  name: "blog_post_retriever",
  description: "Searches and returns exceprts from the Autonomous Agents blog post.",
});

const tools = [blogPostRetrieverTool];

import { createReactAgent } from "@langchain/langgraph/prebuilt"
import { HumanMessage } from "@langchain/core/messages";

const agentExecutor = createReactAgent({ llm, tools });
let query = "What is Task Decomposition?";
// Triggers data retrieval.
console.log("Starting complex question");
console.log("----");
for await (const s of await agentExecutor.stream({
  messages: [new HumanMessage(query)],
})) {
  console.log(s);
  console.log("----");
}

// Set up for managing state.
import { MemorySaver } from "@langchain/langgraph";
const memory = new MemorySaver();
const agentExecutorWithMemory = createReactAgent({
  llm,
  tools,
  checkpointSaver: memory,
});
const config = { configurable: { thread_id: "thread1" } };
query = "Hi, I'm Bob";
// Does not trigger data retrieval, because of how it interprets the query.
console.log("Starting simple greeting.");
console.log("----");
for await (const s of await agentExecutorWithMemory.stream({
  messages: [new HumanMessage(query)]},
  config
)) {
  console.log(s);
  console.log("----");
}

query = 'What is the main idea of the post? Please use no more than 2 concise sentences, and address me by name.';
console.log('Followup question 1');
console.log(query);
console.log("----");
for await (const s of await agentExecutorWithMemory.stream({
  messages: [new HumanMessage(query)]},
  config
)) {
  console.log(s);
  console.log("----");
}

query = 'Could you go into more detail about that?';
console.log('Followup question 2');
console.log(query);
console.log("----");
for await (const s of await agentExecutorWithMemory.stream({
  messages: [new HumanMessage(query)]},
  config
)) {
  console.log(s);
  console.log("----");
}

query = 'According to the post, what are common ways of doing it?';
console.log('Followup question 3');
console.log(query);
console.log("----");
for await (const s of await agentExecutorWithMemory.stream({
  messages: [new HumanMessage(query)]},
  config
)) {
  console.log(s);
  console.log("----");
}
