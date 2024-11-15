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
  model: "gpt-4o-mini",
  temperature: 0,
});

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { BaseMessage, HumanMessage, AIMessage } from "@langchain/core/messages";

///////////////////////////////////
// Version without chat history. //
///////////////////////////////////
// 1. Load, chunk and index the contents of the blog to create a retriever.
// Used by both versions, mainly the `retriever`.
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

// 2. Incorporate the retriever into a question-answering chain.
const systemPrompt =
  "You are an assistant for question-answering tasks. " +
  "Use the following pieces of retrieved context to answer " +
  "the question. If you don't know the answer, say that you " +
  "don't know. Use three sentences maximum and keep the " +
  "answer concise." +
  "\n\n" +
  "{context}";

const prompt = ChatPromptTemplate.fromMessages([
  ["system", systemPrompt],
  ["human", "{input}"],
]);

const questionAnswerChain = await createStuffDocumentsChain({
  llm,
  prompt,
});

const ragChain = await createRetrievalChain({
  retriever,
  combineDocsChain: questionAnswerChain,
});

const response = await ragChain.invoke({
  input: "What is the most complex idea in this article?"
});

////////////////////////////////
// Version with chat history. //
////////////////////////////////
// Text of system prompt to contextualize the query.
const contextualizeQuerySystemPrompt =
  "Given a chat history and the latest user question " +
  "which might reference context in the chat history, " +
  "formulate a standalone question which can be understood " +
  "without the chat history. Do NOT answer the question, " +
  "just reformulate it if needed and otherwise return it as is.";

// Prompt template combining a system prompt, chat history, and human input.
const contextualizeQueryPrompt = ChatPromptTemplate.fromMessages([
  ["system", contextualizeQuerySystemPrompt],
  new MessagesPlaceholder("chat_history"),
  ["human", "{input}"],
]);

// Document retriever that uses an LLM, a regular retriever,
// and a query contextualizer.
const historyAwareRetriever = await createHistoryAwareRetriever({
  llm,
  retriever,
  rephrasePrompt: contextualizeQueryPrompt,
});

// Prompt for chat, to retrieve a conversational response from an LLM.
// This combines a system prompt, chat history, and human input, very
// similar to the contextualizeQueryPrompt.
const chatPrompt = ChatPromptTemplate.fromMessages([
  ["system", systemPrompt],
  new MessagesPlaceholder("chat_history"),
  ["human", "{input}"],
]);

// Chain that is given to the retriever.
const docsChain = await createStuffDocumentsChain({
  llm,
  prompt: chatPrompt,
});

// Chain that retrieves documents by combining
// the history aware retriever and the docs chain.
const chatChain = await createRetrievalChain({
  retriever: historyAwareRetriever,
  combineDocsChain: docsChain,
});

// Chat history that will be updated as the conversation proceeds.
let chatHistory: BaseMessage[] = [];
// Initial question. Variable will be reused for new questions.
let question = "What is Task Decomposition?";
// Initial response. Variable is reused for subsequent responses.
let aiResponse = await chatChain.invoke({
  input: question,
  chat_history: chatHistory,
});
// Store the question and response in the chat history.
chatHistory = chatHistory.concat([
  new HumanMessage(question),
  new AIMessage(aiResponse.answer),
]);

// Repeat question, response, and update history pattern.
question = "What are common ways of doing it?";
aiResponse = await chatChain.invoke({
  input: question,
  chat_history: chatHistory,
});
chatHistory = chatHistory.concat([
  new HumanMessage(question),
  new AIMessage(aiResponse.answer),
]);

// Repeat question, response, and update history pattern again.
question = "Of those, pick one and go into more detail";
aiResponse = await chatChain.invoke({
  input: question,
  chat_history: chatHistory,
});
chatHistory = chatHistory.concat([
  new HumanMessage(question),
  new AIMessage(aiResponse.answer),
]);

console.log(aiResponse.answer);
