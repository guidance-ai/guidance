from typing import Any, Dict, List, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator


class LlamaCppEmbeddings_Guidance(BaseModel, Embeddings):
    """llama.cpp embedding models using Guidance

    To use, you should have the llama-cpp-python and langchain library installed.
    LlamaCpp istance must have embedding = true.
    
    USAGE EXAMPLE (using Chroma database):
        llama2 = guidance.models.LlamaCpp(model=modelPath,n_gpu_layers=-1,n_ctx=4096,embedding = true)
        embeddings = GuidanceLlamaCppEmbeddings(client=llama2)
        vectordb = Chroma(persist_directory={path_to_chromadb}, embedding_function=embeddings)
    """
    model: Any
    client: Optional[Any]

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        
        """Validate that llama-cpp-python library is installed."""
        try:
            if values["model"].engine.model_obj:
                values["client"] = values["model"].engine.model_obj
                return values

            if values["model"].model_obj:
                values["client"] = values["model"].model_obj
                return values
            
            raise ModuleNotFoundError("Could not import llama-cpp-python library or incompatible version.")
        except ImportError:
            raise ModuleNotFoundError(
                "Could not import llama-cpp-python library. "
                "Please install the llama-cpp-python library to "
                "use this embedding model: pip install llama-cpp-python"
            )
        except Exception as e:
            raise ValueError(
                f"Received error {e}"
            )
        return values


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the Llama model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = [self.client.embed(text) for text in texts]
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the Llama model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embedding = self.client.embed(text)
        return list(map(float, embedding))
