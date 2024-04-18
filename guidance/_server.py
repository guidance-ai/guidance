import base64
import os

from typing import TYPE_CHECKING

try:
    import pydantic

    from fastapi import FastAPI, HTTPException, Security
    from fastapi.security import APIKeyHeader
    from fastapi.responses import StreamingResponse
except ImportError:
    if TYPE_CHECKING:
        raise

from .models._model import Model, Engine
from ._grammar import GrammarFunction


class GuidanceRequest(pydantic.BaseModel):
    parser: str = pydantic.Field(
        title="parser",
        description="The text generated so far by the guidance program",
    )
    grammar: str = pydantic.Field(
        title="grammar",
        description="Guidance grammar to constrain the next characters generated",
    )


class Server:

    def __init__(self, engine, api_key=None, ssl_certfile=None, ssl_keyfile=None):
        """This exposes an Engine object over the network."""

        if isinstance(engine, Model):
            engine = engine.engine
        elif not isinstance(engine, Engine):
            raise TypeError("engine must be an Engine object")
        self.engine = engine
        self.app = FastAPI()
        self.valid_api_keys = self._load_api_keys(api_key)
        if ssl_certfile is None:
            ssl_certfile = os.getenv("GUIDANCE_SSL_CERTFILE")
        if ssl_keyfile is None:
            ssl_keyfile = os.getenv("GUIDANCE_SSL_KEYFILE")
        self.ssl_certfile = ssl_certfile
        self.ssl_keyfile = ssl_keyfile

        api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

        # def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
        #     if api_key_header in self.valid_api_keys:
        #         return api_key_header
        #     raise HTTPException(
        #         status_code=status.HTTP_401_UNAUTHORIZED,
        #         detail="Invalid or missing API Key",
        #     )

        @self.app.post("/extend")
        async def extend_parser(
            guidance_request: "GuidanceRequest",
            x_api_key: str = Security(api_key_header),
        ):
            if x_api_key not in self.valid_api_keys:
                raise HTTPException(status_code=401, detail="Invalid API key")

            # data = await request.json()
            # parser = data.get("parser")
            grammar = GrammarFunction.deserialize(
                base64.b64decode(guidance_request.grammar)
            )

            return StreamingResponse(
                self.engine(guidance_request.parser, grammar),
                media_type="application/json",
            )

    def _load_api_keys(self, api_key):
        valid_api_keys = set()
        if api_key is None:
            api_key = os.getenv("GUIDANCE_API_KEY")
            if api_key:
                valid_api_keys.add(api_key)
        else:
            valid_api_keys.add(api_key)
        return valid_api_keys

    def run(self, host="localhost", port=8000):
        # Use uvicorn or another ASGI server to run
        import uvicorn

        uvicorn.run(
            self.app,
            host=host,
            port=port,
            ssl_certfile=self.ssl_certfile,
            ssl_keyfile=self.ssl_keyfile,
        )  # use host="0.0.0.0" for remote access
