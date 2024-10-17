// Expected usage:
// node rag.js
import "cheerio";
import dotenv from "dotenv";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { pull } from "langchain/hub";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";

dotenv.config()

const selector = "article";
const loader = new CheerioWebBaseLoader(
  "https://www.phase2technology.com/blog/phase2-elevates-to-sitecore-platinum-partner-status-and-named-winner-for-2024-customer-success",
  {
    selector: selector
  }
);

const docs = await loader.load();

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const splits = await textSplitter.splitDocuments(docs);
const vectorStore = await MemoryVectorStore.fromDocuments(
  splits,
  new OpenAIEmbeddings()
);

// Retrieve and generate using the relevant snippets of the blog.
const retriever = vectorStore.asRetriever();
const prompt = await pull("rlm/rag-prompt");
const llm = new ChatOpenAI({ model: "gpt-3.5-turbo", temperature: 0 });

const ragChain = await createStuffDocumentsChain({
  llm,
  prompt,
  outputParser: new StringOutputParser(),
});

const retrievedDocs = await retriever.invoke("what was the award about");
console.log(retrievedDocs);
const answer = await ragChain.invoke({
  question: "what was the award about?",
  context: retrievedDocs,
});
console.log('answer', answer);
