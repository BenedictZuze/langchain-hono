import {
  ChatCloudflareWorkersAI,
  CloudflareVectorizeStore,
  CloudflareWorkersAIEmbeddings,
} from "@langchain/cloudflare";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { Document } from "@langchain/core/documents";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Context, Hono } from "hono";

const app = new Hono();

app.get("/", async (c: Context) => {
  const loader = new CheerioWebBaseLoader(
    "https://docs.qgis.org/3.34/en/docs/user_manual/working_with_projections/working_with_projections.html"
  );
  const doc = await loader.load();
  const splitter = new RecursiveCharacterTextSplitter();
  const docs = await splitter.splitDocuments(doc);
  docs.forEach((doc) => delete doc.metadata.loc);

  const embeddings = new CloudflareWorkersAIEmbeddings({
    binding: c.env.AI,
    model: "@cf/baai/bge-small-en-v1.5",
  });
  const store = new CloudflareVectorizeStore(embeddings, {
    index: c.env.VECTORIZE,
  });
  // await store.addDocuments(docs);

  const retriever = store.asRetriever(4);

  const model = new ChatCloudflareWorkersAI({
    cloudflareAccountId: c.env.CLOUDFLARE_ACCOUNT_ID,
    cloudflareApiToken: c.env.CLOUDFLARE_API_TOKEN,
  });

  // Create a system & human prompt for the chat model
  const SYSTEM_TEMPLATE = `Use the following pieces of context to answer the question at the end.
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                ----------------
                {context}`;
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", SYSTEM_TEMPLATE],
    ["human", "{question}"],
  ]);

  const formatDocumentsAsString = (documents: Document[]) => {
    return documents.map((document) => document.pageContent).join("\n\n");
  };

  const chain = RunnableSequence.from([
    {
      context: retriever.pipe(formatDocumentsAsString),
      question: new RunnablePassthrough(),
    },
    prompt,
    model,
    new StringOutputParser(),
  ]);

  const { searchParams } = new URL(c.req.url);
  const question = searchParams.get("question") ?? "Where is Brooklyn located?";
  const answer = await chain.invoke(question);

  return c.text(answer);
});

export default app;
