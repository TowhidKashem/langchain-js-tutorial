#!/usr/bin/env zx
import 'zx/globals';
import { config } from 'dotenv';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { RunnableSequence } from '@langchain/core/runnables';
import { ChatOpenAI } from '@langchain/openai';
import { RedisChatMessageHistory } from '@langchain/redis';
import { ConversationChain } from 'langchain/chains';
import { BufferMemory } from 'langchain/memory';

config({ path: '../../.env' });

// Instantiate the model
const model = new ChatOpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: 'gpt-3.5-turbo',
  temperature: 0.7
});

// Prompt template
const prompt = ChatPromptTemplate.fromTemplate(`
  You are an AI assistant.
  History: {history}
  {input}
`);

const upstashChatHistory = new RedisChatMessageHistory({
  sessionId: 'chat1',
  config: {
    url: `rediss://default:${process.env.UPSTASH_PASSWORD}@${process.env.UPSTASH_REDIS_URL}:6379`
  }
});

// Memory
const memory1 = new BufferMemory({
  memoryKey: 'history',
  chatHistory: upstashChatHistory
});

//*------------------------------ Method 1: using memory via Chain class ------------------------------*//

// this chain will insert the memory into the model's context
const chain1 = new ConversationChain({
  llm: model,
  prompt,
  memory: memory1
});

// Test that memory is working (it will also be saved to the redis session)
console.log('Memory: ', await memory1.loadMemoryVariables({}));
const response1 = await chain1.invoke({
  input: 'The bird is yellow and green'
});
console.log(response1);

console.log('Updated Memory: ', await memory1.loadMemoryVariables({}));
const response2 = await chain1.invoke({
  // input: 'What did I tell you earlier about the bird?'
  input: 'how many time did I ask you that question about the bird?'
});
console.log(response2); // "you mentioned the bird is yellow and green" | "you asked me that question about the bird a total of three times"

//*------------------------------ Method 2: using memory via LCEL (NEWER APPROACH) ------------------------------*//

const memory2 = new BufferMemory({
  memoryKey: 'history',
  chatHistory: upstashChatHistory
});

// 2 or more "runnables" can be chained into "sequences"

const chain2 = RunnableSequence.from([
  {
    input: (initialInput) => initialInput.input, // these are "runnables", they receive the last input from the previous sequence as argument
    memory: () => memory2.loadMemoryVariables({}) // custom runnable (can be named anything), in this case it stores the current memory's state
  },
  {
    input: (previousOutput) => previousOutput.input,
    history: (previousOutput) => previousOutput.memory.history
  },
  prompt,
  model
]);

console.log('Updated Memory: ', await memory2.loadMemoryVariables({}));
const response3 = await chain2.invoke({
  input: 'What did I tell you earlier about the bird?'
});
console.log(response3); // "you mentioned the bird is yellow and green"
