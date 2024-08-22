import pinecone
import openai
from uuid import uuid4
from tqdm.auto import tqdm


class VectorDBChain:

    name: str = "Vector Search Tool"
    description: str = "A tool for finding information about a topic."

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        index_name,
        environment,
        pinecone_api_key
    ):
        pinecone.init(
            api_key=pinecone_api_key,
            environment=environment
        )

        for indexname in pinecone.list_indexes():
            pinecone.delete_index(indexname)

        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                metric="cosine",
                shards=1
            )

        self.index = pinecone.Index(index_name)

    def _embed(self, texts):
        res = openai.Embedding.create(
            input=texts,
            engine="text-embedding-ada-002"
        )
        embeds = [x["embedding"] for x in res["data"]]
        return embeds

    def query(self, text):
        xq = self._embed([text])[0]
        res = self.index.query(
            queries=xq,
            top_k=3,
            include_metadata=True
        )
        documents = [x["metadata"]["text"] for x in res["matches"]]
        return documents

    def build_index(self, documents, batch_size=100):
        for i in tqdm(range(0, len(documents), batch_size)):
            i_end = min(i + batch_size, len(documents))
            batch = documents[i:i_end]
            xd = self._embed(batch)
            metadata = [{"document": x} for x in batch]
            ids = [str(uuid4()) for _ in batch]
            self.index.upsert(
                vectors=zip(ids, xd, metadata)
            )
