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
const docs = await loader.load();

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 50,
});
const allSplits = await textSplitter.splitDocuments(docs);
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
const docs2 = await vectorStore.similaritySearch(question);
console.log({docsFromVectorStore: docs2.length});
