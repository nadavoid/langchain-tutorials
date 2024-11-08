// Expected usage:
// yarn build && node rag.js
import "cheerio";
import dotenv from "dotenv";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { pull } from "langchain/hub";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
dotenv.config();
const selector = "article";
const loader = new CheerioWebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/", {
    selector: selector,
});
const docs = await loader.load();
const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 50,
});
const splits = await textSplitter.splitDocuments(docs);
console.log('number of splits', splits.length);
const vectorStore = await MemoryVectorStore.fromDocuments(splits, new OpenAIEmbeddings());
// Retrieve and generate using the relevant snippets of the blog.
const retriever = vectorStore.asRetriever();
const prompt = await pull("rlm/rag-prompt");
const llm = new ChatOpenAI({ model: "gpt-3.5-turbo", temperature: 0 });
const ragChain = await createStuffDocumentsChain({
    llm,
    prompt,
    outputParser: new StringOutputParser(),
});
const question = "what are the types of memory";
const retrievedDocs = await retriever.invoke(question);
console.log('retrievedDocs', retrievedDocs);
const answer = await ragChain.invoke({
    question: question,
    context: retrievedDocs,
});
console.log('answer', answer);
