# ROOT/EXAMPLES/SEARCHAGENT/SRC/SAMSARA/PROVIDERS/SEARCHPROVIDER.PY

from tavily import AsyncTavilyClient

class SearchProvider:
    def __init__(
            self,
            api_key: str
    ):
        self.client = AsyncTavilyClient(api_key=api_key)


    async def search(
            self,
            query: str
    ) -> dict:
        results = await self.client.search(query)
        return results

    async def extract(self, urls: list[str]) -> dict:
        """NEW: Actually extract raw content from the given URLs."""
        results = await self.client.extract(urls=urls, include_images=False)
        return results