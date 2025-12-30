import array
import ctypes
import time

import llama_cpp
import numpy as np
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler


def get_image_embedding(llm: Llama, llava: Llava15ChatHandler, path: str) -> np.array:
    # Important, otherwise embeddings seem to leak across different
    # invocations of get_image_embedding
    llm.reset()
    llm._ctx.kv_cache_clear()

    image_bytes = llava.load_image(path)

    data_array = array.array("B", image_bytes)
    c_ubyte_ptr = (ctypes.c_ubyte * len(data_array)).from_buffer(data_array)

    t_start = time.time_ns()

    """embed = llava._llava_cpp.llava_image_embed_make_with_bytes(
        ctx_clip=llava.clip_ctx,
        image_bytes=c_ubyte_ptr,
        n_threads=6,
        image_bytes_length=len(image_bytes),
    )"""

    embed = llava._llava_cpp.llava_image_embed_make_with_bytes(
        llava.clip_ctx, 6, c_ubyte_ptr, len(image_bytes)
    )

    t_embed_llava = time.time_ns()

    n_past = ctypes.c_int(llm.n_tokens)
    n_past_p = ctypes.pointer(n_past)

    # Write the image represented by embed into the llama context
    """llava._llava_cpp.llava_eval_image_embed(
        ctx_llama=llm.ctx,
        embed=embed,
        n_batch=llm.n_batch,
        n_past=n_past_p,
    )"""

    llava._llava_cpp.llava_eval_image_embed(llm.ctx, embed, llm.n_batch, n_past_p)
    """    ctx_llama=llm.ctx,
        embed=embed,
        n_batch=llm.n_batch,
        n_past=n_past_p,
    )"""

    t_eval = time.time_ns()

    print(n_past.value)

    assert llm.n_ctx() >= n_past.value
    llm.n_tokens = n_past.value
    llava._llava_cpp.llava_image_embed_free(embed)

    # Get the embedding out of the LLM
    embedding = np.array(
        llama_cpp.llama_get_embeddings(llm._ctx.ctx)[: llama_cpp.llama_n_embd(llm._model.model)]
    )

    t_embed_llm = time.time_ns()

    print(
        f"Total: {float(t_embed_llm - t_start) / 1e6:.2f}ms, LLaVA embed: {float(t_embed_llava - t_start) / 1e6:.2f}ms, LLaMA eval: {float(t_eval - t_embed_llava) / 1e6:.2f}ms, LLaMA embed: {float(t_embed_llm - t_eval) / 1e6:.2f}ms"
    )

    llm.reset()
    return embedding


def get_similarity_matrix(left: np.array, right: np.array):
    return np.dot(left, right.T) / (
        np.linalg.norm(left, axis=1)[:, np.newaxis] * np.linalg.norm(right, axis=1)[np.newaxis, :]
    )


def get_text_embedding(llm: Llama, text: str) -> np.array:
    embed = np.array(llm.embed(text))
    llm.reset()
    return embed


chat_handler = Llava15ChatHandler(clip_model_path="mmproj-model-f16.gguf", verbose=False)
llm = Llama(
    model_path="ggml-model-q5_k.gguf",
    chat_handler=chat_handler,
    n_ctx=2048,
    n_batch=1024,
    logits_all=False,
    n_threads=6,
    offload_kqv=True,
    n_gpu_layers=64,
    embedding=True,
    verbose=False,
)


class LLaVautils:
    def __init__(
        self,
        model_path: str,
        clip_model_path: str,
        n_ctx=2048,
        n_batch=1024,
        n_threads=6,
        n_gpu_layers=64,
    ):
        self.chat_handler = Llava15ChatHandler(clip_model_path=clip_model_path, verbose=False)
        self.llm = Llama(
            model_path=model_path,
            chat_handler=self.chat_handler,
            n_ctx=n_ctx,
            n_batch=n_batch,
            logits_all=False,
            n_threads=n_threads,
            offload_kqv=True,
            n_gpu_layers=n_gpu_layers,
            embedding=True,
            verbose=False,
        )

    def get_image_embedding(self, path: str) -> np.array:
        """Encodes an image and returns its embedding vector."""
        self.llm.reset()
        self.llm._ctx.kv_cache_clear()

        image_bytes = self.chat_handler.load_image(path)
        data_array = array.array("B", image_bytes)
        c_ubyte_ptr = (ctypes.c_ubyte * len(data_array)).from_buffer(data_array)

        embed = self.chat_handler._llava_cpp.llava_image_embed_make_with_bytes(
            self.chat_handler.clip_ctx, 6, c_ubyte_ptr, len(image_bytes)
        )

        n_past = ctypes.c_int(self.llm.n_tokens)
        n_past_p = ctypes.pointer(n_past)

        self.chat_handler._llava_cpp.llava_eval_image_embed(
            self.llm.ctx, embed, self.llm.n_batch, n_past_p
        )
        self.llm.n_tokens = n_past.value
        self.chat_handler._llava_cpp.llava_image_embed_free(embed)

        embedding = np.array(
            llama_cpp.llama_get_embeddings(self.llm._ctx.ctx)[
                : llama_cpp.llama_n_embd(self.llm._model.model)
            ]
        )

        self.llm.reset()
        return embedding[0]

    def get_text_embedding(self, text: str) -> np.array:
        """Encodes a text and returns its embedding vector."""
        embed = np.array(self.llm.embed(text))
        embed = np.mean(embed, axis=0)
        self.llm.reset()
        return embed
