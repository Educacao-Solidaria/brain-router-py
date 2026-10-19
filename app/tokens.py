"""Contagem local de tokens com o tiktoken, na codificação do modelo que vai receber o texto.

    count = token_counter("openai/text-embedding-3-small")
    count("A matrícula começa em março.")
    chunk_text(texto, doc_id, count_tokens=count)  # chunk_size passa a ser token de verdade

- O nome do modelo pode vir como na OpenRouter: o prefixo do provedor (`openai/`) e a
  variante (`:free`, `:nitro`) saem antes de consultar o mapa do tiktoken.
- Modelo que o tiktoken não conhece (Claude, Llama, Gemini...) usa `FALLBACK_ENCODING`
  (cl100k_base), com um aviso no log por nome de modelo. É estimativa: em pt-BR o cl100k
  conta mais tokens que o o200k, o que erra para o lado seguro num orçamento de contexto,
  mas o tokenizador real do modelo pode contar ainda mais. Deixe folga.
- O texto conta como texto: `<|endoftext|>` vindo de um documento são 7 tokens comuns, não o
  token especial (o `encode` padrão do tiktoken levantaria erro nesse caso).
- Rede: no primeiro uso de cada codificação o tiktoken baixa o arquivo dela (~1,7 MB o
  cl100k) de openaipublic.blob.core.windows.net e o guarda em `TIKTOKEN_CACHE_DIR` (sem a
  variável, numa pasta do diretório temporário). Ele confere o SHA-256 do arquivo, baixado ou
  em cache. Em produção sem acesso a esse host, deixe o arquivo pronto na imagem e aponte a
  variável. `token_counter` carrega a codificação na hora: o erro de rede aparece na montagem
  do serviço, não no meio da ingestão.
"""

from functools import lru_cache

import tiktoken

from app.chunking import TokenCounter
from app.log import get_logger

log = get_logger(__name__)

DEFAULT_MODEL = "openai/text-embedding-3-small"
FALLBACK_ENCODING = "cl100k_base"


@lru_cache(maxsize=256)
def encoding_name_for(model: str) -> str:
    """Codificação do tiktoken para `model`, aceitando o formato `provedor/modelo:variante`."""
    name = model.strip().rsplit("/", 1)[-1]
    if not name:
        raise ValueError(f"nome de modelo vazio: {model!r}")
    # O nome inteiro antes: em "ft:gpt-4o:org::id" (fine-tune) o ":" não é variante.
    for candidate in (name, name.split(":", 1)[0]):
        try:
            return tiktoken.encoding_name_for_model(candidate)
        except KeyError:
            continue
    # Uma vez por nome de modelo (lru_cache): o aviso não se repete a cada contagem.
    log.warning("tokens.unknown_model", model=model, encoding=FALLBACK_ENCODING)
    return FALLBACK_ENCODING


@lru_cache(maxsize=8)
def _encoding(name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


def token_counter(model: str = DEFAULT_MODEL) -> TokenCounter:
    """Função `texto -> nº de tokens` de `model`, no formato do `count_tokens` do chunker."""
    encoding = _encoding(encoding_name_for(model))

    def count(text: str) -> int:
        return len(encoding.encode_ordinary(text))

    return count


def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    """Número de tokens de `text` na codificação de `model`."""
    return token_counter(model)(text)
