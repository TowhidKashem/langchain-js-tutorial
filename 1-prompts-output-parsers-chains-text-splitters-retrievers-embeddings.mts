//
// tutorial: https://www.youtube.com/watch?v=MaSynwSIty4&list=PL4HikwTaYE0EG379sViZZ6QsFMjJ5Lfwj&index=1
//
import { config } from 'dotenv';
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { Document } from '@langchain/core/documents';
import {
  CommaSeparatedListOutputParser,
  StringOutputParser
} from '@langchain/core/output_parsers';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { createRetrievalChain } from 'langchain/chains/retrieval';
import { StructuredOutputParser } from 'langchain/output_parsers';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { z } from 'zod';

config({ path: '../../../.env' });

const model = new ChatOpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
  model: 'gpt-3.5-turbo',
  temperature: 0.7, // creativity
  maxTokens: 1_000
  // streaming: true,
  // verbose: true
});

//*------------------------------ BASIC USAGE ------------------------------*//

// const response = await model.invoke('tell me a poem about AI'); // regular
// const response = await model.stream('tell me a poem about AI'); // streaming

// console.log(response); // regular

// for await (const chunk of response) {// streaming
//   console.log(chunk);
// }

//*------------------------------ FORMATTING THE OUTPUT USING OUTPUT PARSERS ------------------------------*//

async function callStringOutputParser() {
  // make prompt template
  const prompt = ChatPromptTemplate.fromMessages([
    [
      'system',
      'You are a comedian, tell me a joke based on a word provided by the user.'
    ],
    ['human', '{input}']
  ]);
  // console.log(await prompt.format({ input: 'chickens' }));
  const parser = new StringOutputParser(); // create parser to format output
  const chain = prompt.pipe(model).pipe(parser); // create chain
  const response = await chain.invoke({ input: 'dogs' });
  return response;
}

async function callListOutputParser() {
  const prompt = ChatPromptTemplate.fromTemplate(
    'Provide 5 synonyms, separated by commas, for the following word {word}'
  );
  const outputParser = new CommaSeparatedListOutputParser();
  const chain = prompt.pipe(model).pipe(outputParser);
  const response = await chain.invoke({ word: 'animals' });
  return response;
}

async function callStructuredOutputParser() {
  const prompt = ChatPromptTemplate.fromTemplate(`
    Extract information from the following phrase.
    Formatting Instructions: {format_instructions}
    Phrase: {phrase}
  `);
  const outputParser = StructuredOutputParser.fromNamesAndDescriptions({
    name: 'the name of the person',
    age: 'the age of the person'
  });
  const chain = prompt.pipe(model).pipe(outputParser);
  const response = await chain.invoke({
    phrase: 'Max is 40 years old',
    // phrase: 'Max is a man', // will show 'unknown' for age property
    format_instructions: outputParser.getFormatInstructions()
  });
  return response;
}

async function callZodOutputParser() {
  const prompt = ChatPromptTemplate.fromTemplate(`
    Extract information from the following phrase.
    Formatting Instructions: {format_instructions}
    Phrase: {phrase}
  `);
  const outputParser = StructuredOutputParser.fromZodSchema(
    z.object({
      recipe: z.string().describe('the name of the recipe'),
      ingredients: z.array(z.string()).describe('ingredients')
    })
  );
  const chain = prompt.pipe(model).pipe(outputParser);
  const response = await chain.invoke({
    // phrase: 'Max is 40 years old',
    phrase:
      'The recipe is called "Pasta Carbonara" and the ingredients are eggs, bacon, and cheese.',
    format_instructions: outputParser.getFormatInstructions()
  });
  return response;
}

// console.log(await callStringOutputParser());
// console.log(await callListOutputParser());
// console.log(await callStructuredOutputParser());
// console.log(await callZodOutputParser());

//*------------------------------ CONNECTING TO EXTERNAL DATA SOURCES ------------------------------*//

const prompt = ChatPromptTemplate.fromTemplate(`
  Answer the user's question.
  Context: {context}
  Question: {input}
`);

// trivial example - hard coded context
async function hardCodedContext() {
  const chain = prompt.pipe(model);
  const response = await chain.invoke({
    input: 'What is LCEL?',
    context: 'LCEL stands for Langchain Chatbot Engine Language.'
  });
  return response;
}

// trivial example - manually constructed Document context
// a Document contains data and optional metadata
// metadata can hold bits of info needed later like the source of the data, etc
async function manuallyConstructedDocumentContext() {
  const documentA = new Document({
    metadata: {
      source: 'https://python.langchain.com/v0.1/docs/expression_language/'
    },
    pageContent: 'LCEL stands for Langchain Chatbot Engine Language.'
  });
  const documentB = new Document({
    pageContent: 'The password is 1234.'
  });
  const chain = await createStuffDocumentsChain({
    llm: model,
    prompt
  });
  const response = await chain.invoke({
    input: 'What is LCEL?',
    context: [documentA, documentB]
    // input: 'What is the password?'
  });
  return response;
}

// realistic example - dynamically load context scraped from a web page
async function scrapedWebsiteContext() {
  const loader = new CheerioWebBaseLoader(
    'https://python.langchain.com/v0.1/docs/expression_language/'
  );
  const docs = await loader.load();
  const chain = prompt.pipe(model);

  console.log({
    docs, // contains unnecessary parts of the page like javascript, new line symbols etc
    size: docs[0].pageContent.length // is "4179" chars long (too much!)
  });

  const response = await chain.invoke({
    input: 'What is LCEL?',
    // @ts-ignore
    context: docs
  });
  return response;
}

// to avoid the large context size (all content on page) we can grab just the relevant part using a "text splitter"
async function scrapedWebsiteContextWithTextSplitter() {
  // create chain
  const chain = await createStuffDocumentsChain({
    llm: model,
    prompt
  });

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

  // console.log(splitDocs); // prints webpage contents in small chunks in separate docs

  /*--------- create embedding and store in vector DB ---------*/

  // create embeddings out of the smaller chunks and store them in a local in-memory vector database
  const embeddings = new OpenAIEmbeddings();
  const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );

  /*--------- perform vector search ---------*/

  // query the store to get the most similar chunks to the input prompt
  const retriever = vectorStore.asRetriever({
    k: 2 // optional: return the top 2 matches
  });
  const retrievalChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever
  });

  /*--------- execute ---------*/

  const response = await retrievalChain.invoke({
    input: 'What is LCEL?'
    // note: no need to add context, the retriever chain will handle it automatically
  });
  return response;
}

// console.log(await hardCodedContext());
// console.log(await manuallyConstructedDocumentContext());
// console.log(await scrapedWebsiteContext());
console.log(await scrapedWebsiteContextWithTextSplitter());
