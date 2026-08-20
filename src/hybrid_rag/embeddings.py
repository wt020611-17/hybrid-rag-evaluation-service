from pathlib import Path
from typing import Iterable, List, Optional, Sequence


class BGEEmbedder:
    """Local BGE encoder using the model card's normalized CLS pooling."""

    QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        model_path: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "BGE dependencies are missing; install the production extra"
            ) from exc
        self._torch = torch
        self.model_name = model_name
        self.model_source = str(Path(model_path).resolve()) if model_path else model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_source)
        self.model = AutoModel.from_pretrained(self.model_source)
        self.model.to(device)
        self.model.eval()
        self.dimension = int(self.model.config.hidden_size)

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        return self._encode(list(texts), query=False)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text], query=True)[0]

    def _encode(self, texts: Sequence[str], query: bool) -> List[List[float]]:
        if not texts:
            return []
        prepared = [self.QUERY_INSTRUCTION + text if query else text for text in texts]
        vectors: List[List[float]] = []
        for start in range(0, len(prepared), self.batch_size):
            batch = prepared[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self._torch.no_grad():
                output = self.model(**encoded)
            pooled = output.last_hidden_state[:, 0]
            normalized = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(normalized.cpu().tolist())
        return vectors
