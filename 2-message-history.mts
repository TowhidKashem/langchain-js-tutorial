import { config } from 'dotenv';
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import {
  ChatPromptTemplate,
  MessagesPlaceholder
} from '@langchain/core/prompts';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { createHistoryAwareRetriever } from 'langchain/chains/history_aware_retriever';
import { createRetrievalChain } from 'langchain/chains/retrieval';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';

config({ path: '../../../.env' });

//*------------------------------ ADD CHAT HISTORY ------------------------------*//

// load data and create vector store
const createVectorStore = async () => {
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

  return vectorStore;
};

// create retrieval chain
const createChain = async (vectorStore: MemoryVectorStore) => {
  // Instantiate Model
  const model = new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: 'gpt-3.5-turbo',
    temperature: 0.7, // creativity
    maxTokens: 1_000 // output length
    // streaming: true,
    // verbose: true
  });

  // const prompt = ChatPromptTemplate.fromTemplate(`
  //   Answer the user's question.
  //   Context: {context}
  //   Chat History: {chat_history}
  //   Question: {input}
  // `);

  // questions are answered based on the model's inherent knowledge and 2 more things:
  // - the chat history
  // - context retrieved from the vector store
  const prompt = ChatPromptTemplate.fromMessages([
    [
      'system',
      "Answer the user's questions based on the following context: {context}."
    ],
    new MessagesPlaceholder('chat_history'), // the messages placeholder converts the array of message objects into a string
    ['user', '{input}']
  ]);

  // create chain
  const chain = await createStuffDocumentsChain({
    llm: model,
    prompt
  });

  /*--------- create vector search retrieval chain ---------*/

  // query the store to get the most similar chunks to the input prompt
  const retriever = vectorStore.asRetriever({
    k: 3 // optional: return the top 3 matches
  });

  const rephrasePrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder('chat_history'),
    ['user', '{input}'],
    [
      'user',
      'Given the above conversation, generate a search query to look up in order to get information relevant to the conversation'
    ]
  ]);

  const historyAwareRetriever = await createHistoryAwareRetriever({
    llm: model,
    retriever,
    rephrasePrompt
  });

  const conversationChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever: historyAwareRetriever
  });

  return conversationChain;
};

//*------------------------------ USAGE ------------------------------*//

const vectorStore = await createVectorStore();
const chain = await createChain(vectorStore);

// hard-coded chat history
const chatHistory = [
  new HumanMessage('hello'),
  new AIMessage("oh hi there, what's your name?"),
  new HumanMessage('my name is TK'),
  new AIMessage('Hi TK, how can I help you?'),
  new HumanMessage('What is LCEL?'),
  new AIMessage('LCEL stands for Langchain Expression Language.')
];

const response = await chain.invoke({
  // input: 'What is my name?', // without the chat history the AI wouldn't know the user's name
  input: 'what is it?', // not only does the AI know what "it" refers to, but it also has info about LCEL retreved from the vector store
  chat_history: chatHistory
});

console.log(response); //  'Your name is TK.'
