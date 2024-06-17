import { config } from "dotenv";
import {
  CommaSeparatedListOutputParser,
  StringOutputParser,
} from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { StructuredOutputParser } from "langchain/output_parsers";
import { z } from "zod";

config();

const model = new ChatOpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
  model: "gpt-3.5-turbo",
  temperature: 0.7, // creativity
  maxTokens: 1_000,
  // streaming: true,
  // verbose: true
});

//*------------------------------ basic usage ------------------------------*//

// const response = await model.invoke('tell me a poem about AI'); // regular
// const response = await model.stream('tell me a poem about AI'); // streaming

// console.log(response); // regular

// for await (const chunk of response) {// streaming
//   console.log(chunk);
// }

//*------------------------------ formatting the output using output parsers ------------------------------*//

async function callStringOutputParser() {
  // make prompt template
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "You are a comedian, tell me a joke based on a word provided by the user.",
    ],
    ["human", "{input}"],
  ]);
  // console.log(await prompt.format({ input: 'chickens' }));
  const parser = new StringOutputParser(); // create parser to format output
  const chain = prompt.pipe(model).pipe(parser); // create chain
  const response = await chain.invoke({ input: "dogs" });
  return response;
}

async function callListOutputParser() {
  const prompt = ChatPromptTemplate.fromTemplate(
    "Provide 5 synonyms, separated by commas, for the following word {word}"
  );
  const outputParser = new CommaSeparatedListOutputParser();
  const chain = prompt.pipe(model).pipe(outputParser);
  const response = await chain.invoke({ word: "animals" });
  return response;
}

async function callStructuredOutputParser() {
  const prompt = ChatPromptTemplate.fromTemplate(`
    Extract information from the following phrase.
    Formatting Instructions: {format_instructions}
    Phrase: {phrase}
  `);
  const outputParser = StructuredOutputParser.fromNamesAndDescriptions({
    name: "the name of the person",
    age: "the age of the person",
  });
  const chain = prompt.pipe(model).pipe(outputParser);
  const response = await chain.invoke({
    phrase: "Max is 40 years old",
    // phrase: 'Max is a man', // will show 'unknown' for age property
    format_instructions: outputParser.getFormatInstructions(),
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
      recipe: z.string().describe("the name of the recipe"),
      ingredients: z.array(z.string()).describe("ingredients"),
    })
  );
  const chain = prompt.pipe(model).pipe(outputParser);
  const response = await chain.invoke({
    // phrase: 'Max is 40 years old',
    phrase:
      'The recipe is called "Pasta Carbonara" and the ingredients are eggs, bacon, and cheese.',
    format_instructions: outputParser.getFormatInstructions(),
  });
  return response;
}

// console.log(await callStringOutputParser());
// console.log(await callListOutputParser());
// console.log(await callStructuredOutputParser());
console.log(await callZodOutputParser());
