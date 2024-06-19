#!/usr/bin/env zx
import 'zx/globals';
import { config } from 'dotenv';
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { TavilySearchResults } from '@langchain/community/tools/tavily_search';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import {
  ChatPromptTemplate,
  MessagesPlaceholder
} from '@langchain/core/prompts';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { AgentExecutor, createOpenAIFunctionsAgent } from 'langchain/agents';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { createRetrieverTool } from 'langchain/tools/retriever';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';

config({ path: '../../.env' });

//*------------------------------ START Copied from last lesson (part 2) ------------------------------*//

// load data and create vector store:

// get all content from the web page
const loader = new CheerioWebBaseLoader(
  'https://python.langchain.com/v0.1/docs/expression_language/'
);
const docs = await loader.load();

/*--------- split content up into smaller chunks using text splitter ---------*/

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 200,
  chunkOverlap: 20
});
const splitDocs = await splitter.splitDocuments(docs);

/*--------- create embedding and store in vector DB ---------*/

// create embeddings out of the smaller chunks and store them in a local in-memory vector database
const embeddings = new OpenAIEmbeddings();
const vectorStore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings
);

/*--------- create vector search retrieval chain ---------*/

// query the store to get the most similar chunks to the input prompt
const retriever = vectorStore.asRetriever({
  k: 3 // optional: return the top 3 matches
});

//*------------------------------ /END COPIED ------------------------------*//

// Instantiate the model
const model = new ChatOpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: 'gpt-3.5-turbo',
  temperature: 0.7,
  maxTokens: 1_000
});

// Prompt template
const prompt = ChatPromptTemplate.fromMessages([
  ['system', 'You are a helpful assistant called Max.'],
  new MessagesPlaceholder('chat_history'),
  ['human', '{input}'],
  new MessagesPlaceholder('agent_scratchpad')
]);

//*------------------------------ TOOLS ------------------------------*//

const searchTool = new TavilySearchResults();

const retrieverTool = createRetrieverTool(retriever, {
  name: 'lcel_search',
  description:
    'Use this tool when searching for information about Langchain Expression Language (LCEL)'
});

const tools = [searchTool, retrieverTool];

//*------------------------------ /END TOOLS ------------------------------*//

// Create agent
const agent = await createOpenAIFunctionsAgent({
  llm: model,
  prompt,
  tools
});

// Create agent executor
const agentExecutor = new AgentExecutor({
  agent,
  tools
});

// Keep track of chat history
const chatHistory = [];

// Create terminal chat UI
async function askQuestion() {
  try {
    // console.log({ chatHistory }); // peek memory

    // Get user response
    const input = await question('User: ');

    // Call agent
    const response = await agentExecutor.invoke({
      input,
      chat_history: chatHistory
    });

    // Get AI response
    console.log('Agent: ', chalk.green(response.output));

    // Update chat history
    chatHistory.push(new HumanMessage(input));
    chatHistory.push(new AIMessage(response.output));

    askQuestion(); // continue convo recursively
  } catch (error) {
    console.error(error);
    process.exit(1);
  }
}

// Start the conversation off...
askQuestion();
