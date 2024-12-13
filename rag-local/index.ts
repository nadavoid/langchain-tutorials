/**
 * Usage:
 * yarn tsx index.ts
 */
import dotenv from "dotenv";
dotenv.config({
  path: '../.env',
});

import "cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";

const loader = new CheerioWebBaseLoader(
  "https://lilianweng.github.io/posts/2023-06-23-agent/"
);
const sourceDocs = await loader.load();

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 50,
});
const allSplits = await textSplitter.splitDocuments(sourceDocs);
console.log({totalSplits: allSplits.length});

import { OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

// Using Ollama Embeddings in the vector store requires a running instance
// of Ollama locally.
// See `OllamaEmbeddingsParams` for options including server address.
// See https://github.com/ollama/ollama#ollama for installation instructions.
// Install needed model with `ollama run mxbai-embed-large`.
const embeddings = new OllamaEmbeddings();
const vectorStore = await MemoryVectorStore.fromDocuments(
  allSplits,
  embeddings
);
const question = "What are the approaches to Task Decomposition?";
const contextDocs = await vectorStore.similaritySearch(question);
console.log({docsFromVectorStore: contextDocs.length});

// Add chat.
import { ChatOllama } from "@langchain/ollama";
// The current default model for ChatOllama is llama3.
// Install it with `ollama run llama3`.
// View other installed models with `ollama list`.
const ollamaLlm = new ChatOllama({
  // model: 'mistral',
  model: 'llama3.2',
});

// Use Local LLM in a chain.
import { PromptTemplate } from "@langchain/core/prompts";
const prompt = PromptTemplate.fromTemplate(
  "Summarize the main themes in these retrieved docs: {context}"
);
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { StringOutputParser } from "@langchain/core/output_parsers";
const chain = await createStuffDocumentsChain({
  llm: ollamaLlm,
  outputParser: new StringOutputParser(),
  prompt: prompt,
});
const answer = await chain.invoke({
  context: contextDocs,
});
console.log({answer})
