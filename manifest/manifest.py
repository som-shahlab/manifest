"""Manifest class."""
<<<<<<< HEAD
import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast
=======
import asyncio
import copy
import logging
from typing import (
    Any,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)
>>>>>>> upstream/main

import numpy as np

from manifest.caches.noop import NoopCache
from manifest.caches.postgres import PostgresCache
from manifest.caches.redis import RedisCache
from manifest.caches.sqlite import SQLiteCache
<<<<<<< HEAD
from manifest.clients.ai21 import AI21Client
from manifest.clients.cohere import CohereClient
from manifest.clients.dummy import DummyClient
from manifest.clients.huggingface import HuggingFaceClient
from manifest.clients.huggingface_embedding import HuggingFaceEmbeddingClient
from manifest.clients.openai import OpenAIClient
from manifest.clients.openai_chat import OpenAIChatClient
from manifest.clients.openai_embedding import OpenAIEmbeddingClient
from manifest.clients.toma import TOMAClient
from manifest.request import Request
from manifest.response import Response
=======
from manifest.clients.client import Client
from manifest.clients.huggingface import HuggingFaceClient
from manifest.connections.client_pool import (
    CLIENT_CONSTRUCTORS,
    ClientConnection,
    ClientConnectionPool,
)
from manifest.request import LMChatRequest, LMScoreRequest, Request
from manifest.response import ModelChoices, Response, Usage, Usages
>>>>>>> upstream/main

logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

<<<<<<< HEAD
CLIENT_CONSTRUCTORS = {
    OpenAIClient.NAME: OpenAIClient,
    OpenAIChatClient.NAME: OpenAIChatClient,
    OpenAIEmbeddingClient.NAME: OpenAIEmbeddingClient,
    CohereClient.NAME: CohereClient,
    AI21Client.NAME: AI21Client,
    HuggingFaceClient.NAME: HuggingFaceClient,
    HuggingFaceEmbeddingClient.NAME: HuggingFaceEmbeddingClient,
    DummyClient.NAME: DummyClient,
    TOMAClient.NAME: TOMAClient,
}

# Diffusion
DIFFUSION_CLIENTS = ["diffuser", "tomadiffuser"]
try:
    from manifest.clients.diffuser import DiffuserClient
    from manifest.clients.toma_diffuser import TOMADiffuserClient

    CLIENT_CONSTRUCTORS[DiffuserClient.NAME] = DiffuserClient
    CLIENT_CONSTRUCTORS[TOMADiffuserClient.NAME] = TOMADiffuserClient
except Exception:
    logger.info("Diffusion not supported. Skipping import.")
    pass

=======
>>>>>>> upstream/main

CACHE_CONSTRUCTORS = {
    "redis": RedisCache,
    "sqlite": SQLiteCache,
    "noop": NoopCache,
    "postgres": PostgresCache,
}


class Manifest:
    """Manifest session object."""

    def __init__(
        self,
<<<<<<< HEAD
        client_name: str = "openai",
        client_connection: Optional[str] = None,
=======
        client_name: Optional[str] = None,
        client_connection: Optional[str] = None,
        client_pool: Optional[List[ClientConnection]] = None,
        client_pool_schedule: str = "round_robin",
>>>>>>> upstream/main
        cache_name: str = "noop",
        cache_connection: Optional[str] = None,
        stop_token: str = "",
        **kwargs: Any,
    ):
        """
        Initialize manifest.

        Args:
            client_name: name of client.
            client_connection: connection string for client.
<<<<<<< HEAD
=======
            client_pool: list of client connections for multi-client.
            client_pool_schedule: schedule for client pool.
>>>>>>> upstream/main
            cache_name: name of cache.
            cache_connection: connection string for cache.
            stop_token: stop token prompt generation.
                        Can be overridden in run

        Remaining kwargs sent to client and cache.
        """
<<<<<<< HEAD
        if client_name not in CLIENT_CONSTRUCTORS:
            if client_name in DIFFUSION_CLIENTS:
                raise ImportError(
                    f"Diffusion client {client_name} requires the proper install. "
                    "Make sure to run `pip install manifest-ml[diffusers]` "
                    "or install Pillow."
                )
            else:
                raise ValueError(
                    f"Unknown client name: {client_name}. "
                    f"Choices are {list(CLIENT_CONSTRUCTORS.keys())}"
                )
=======
        if not client_name and not client_pool:
            raise ValueError(
                "Must specify client_name or client_pool. "
                f"Choices are {list(CLIENT_CONSTRUCTORS.keys())}"
            )
        if client_name and client_pool:
            raise ValueError("Cannot specify both client_name and client_pool")
        if client_name:
            client_pool = [
                ClientConnection(
                    client_name=client_name,
                    client_connection=client_connection,
                    # Remove engine from kwargs
                    engine=kwargs.pop("engine", None),
                )
            ]
        self.client_pool = ClientConnectionPool(
            client_pool, client_pool_schedule, client_args=kwargs
        )
>>>>>>> upstream/main
        if cache_name not in CACHE_CONSTRUCTORS:
            raise ValueError(
                f"Unknown cache name: {cache_name}. "
                f"Choices are {list(CACHE_CONSTRUCTORS.keys())}"
            )
<<<<<<< HEAD
        self.client_name = client_name
        # Must pass kwargs as dict for client "pop" methods removed used arguments
        self.cache = CACHE_CONSTRUCTORS[cache_name](  # type: ignore
            cache_connection, self.client_name, cache_args=kwargs
        )
        self.client = CLIENT_CONSTRUCTORS[self.client_name](  # type: ignore
            client_connection, client_args=kwargs
=======
        # Must pass kwargs as dict for client "pop" methods removed used arguments
        self.cache = CACHE_CONSTRUCTORS[cache_name](  # type: ignore
            cache_connection, self.client_pool.request_type, cache_args=kwargs
>>>>>>> upstream/main
        )
        if len(kwargs) > 0:
            raise ValueError(f"{list(kwargs.items())} arguments are not recognized.")

        self.stop_token = stop_token

    def close(self) -> None:
        """Close the client and cache."""
<<<<<<< HEAD
        self.client.close()
        self.cache.close()

    def change_client(
        self,
        client_name: Optional[str] = None,
        client_connection: Optional[str] = None,
        stop_token: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Change manifest client.

        Args:
            client_name: name of client.
            client_connection: connection string for client.
            stop_token: stop token prompt generation.
                        Can be overridden in run

        Remaining kwargs sent to client.
        """
        if client_name:
            if client_name not in CLIENT_CONSTRUCTORS:
                raise ValueError(
                    f"Unknown client name: {client_name}. "
                    f"Choices are {list(CLIENT_CONSTRUCTORS.keys())}"
                )
            self.client_name = client_name
            self.client = CLIENT_CONSTRUCTORS[client_name](  # type: ignore
                client_connection, client_args=kwargs
            )
            if len(kwargs) > 0:
                raise ValueError(
                    f"{list(kwargs.items())} arguments are not recognized."
                )

        if stop_token is not None:
            self.stop_token = stop_token

=======
        self.client_pool.close()
        self.cache.close()

>>>>>>> upstream/main
    def _validate_kwargs(self, kwargs: Dict, request_params: Request) -> None:
        """Validate kwargs.

        Args:
            kwargs: kwargs to validate.
            request_params: request object to validate against.
        """
        # Check for invalid kwargs
        non_request_kwargs = [
            (k, v) for k, v in kwargs.items() if k not in request_params.__dict__
        ]
        if len(non_request_kwargs) > 0:
            raise ValueError(
                f"{list(non_request_kwargs)} arguments are not recognized."
            )

        # Warn for valid but unused kwargs
        request_unused_kwargs = [
            (k, v) for k, v in kwargs.items() if k not in non_request_kwargs
        ]
        if len(request_unused_kwargs) > 0:
            logger.warning(f"{list(request_unused_kwargs)} arguments are unused.")
        return

    def _split_cached_requests(
        self,
        request: Request,
<<<<<<< HEAD
=======
        client: Client,
>>>>>>> upstream/main
        overwrite_cache: bool,
    ) -> Tuple[Dict[int, Response], Request]:
        """Split a request into cached responses and Requests to run.

        Args:
            request: request object.
            overwrite_cache: whether to overwrite cache.

        Returns:
            cached_idx_to_response: dict of cached responses.
            new_request: request object with only prompts to run.
        """
        cached_idx_to_response: Dict[int, Response] = {}
        new_request = copy.deepcopy(request)
        if not overwrite_cache:
<<<<<<< HEAD
            if isinstance(new_request.prompt, list):
=======
            if isinstance(new_request.prompt, list) and not isinstance(
                request, LMChatRequest
            ):
>>>>>>> upstream/main
                new_request.prompt = []
                for idx, prompt_str in enumerate(request.prompt):
                    single_request = copy.deepcopy(request)
                    single_request.prompt = prompt_str
                    possible_response = self.cache.get(
<<<<<<< HEAD
                        self.client.get_cache_key(single_request)
=======
                        client.get_cache_key(single_request)
>>>>>>> upstream/main
                    )
                    if possible_response:
                        cached_idx_to_response[idx] = possible_response
                    else:
                        new_request.prompt.append(prompt_str)
<<<<<<< HEAD
            else:
                possible_response = self.cache.get(
                    self.client.get_cache_key(new_request)
                )
                if possible_response:
                    cached_idx_to_response[0] = possible_response
                    new_request.prompt = None
=======
            # Chat or single string requests are not broken down into
            # subprompts for caching.
            elif (isinstance(new_request.prompt, str)) or (
                isinstance(new_request.prompt, list)
                and isinstance(request, LMChatRequest)
            ):
                possible_response = self.cache.get(client.get_cache_key(new_request))
                if possible_response:
                    cached_idx_to_response[0] = possible_response
                    new_request.prompt = None
            else:
                raise ValueError(
                    f"Invalid prompt type: {type(new_request.prompt)}"
                    f" with request type: {type(request)}"
                )
>>>>>>> upstream/main
        return cached_idx_to_response, new_request

    def _stitch_responses_and_cache(
        self,
        request: Request,
<<<<<<< HEAD
=======
        client: Client,
>>>>>>> upstream/main
        response: Union[Response, None],
        cached_idx_to_response: Dict[int, Response],
    ) -> Response:
        """Stich together the cached and uncached responses."""
        # We stitch the responses (the choices) here from both the new request the
        # cached entries.
        all_model_choices = []
        all_usages = []
<<<<<<< HEAD
        all_input_prompts = []
        response_idx = 0
        number_prompts = len(cached_idx_to_response)
        single_output = False
        if response:
            if isinstance(response.get_request()["prompt"], str):
                single_output = True
                number_prompts += 1
            else:
                number_prompts += len(response.get_request()["prompt"])
        response_gen_key = None
        response_logits_key = None
        response_item_key = None
        for idx in range(number_prompts):
            if idx in cached_idx_to_response:
                cached_res = cached_idx_to_response[idx]
                response_gen_key = cached_res.generation_key
                response_logits_key = cached_res.logits_key
                response_item_key = cached_res.item_key
                response_usage_key = cached_res.usage_key
                all_input_prompts.append(cached_res.get_request()["prompt"])
                json_response = cached_res.get_json_response()
                if request.n == 1:
                    assert (
                        len(json_response[response_gen_key]) == 1
                    ), "cached response should have only one choice"
                all_model_choices.extend(json_response[response_gen_key])
                if response_usage_key:
                    all_usages.extend(json_response[response_usage_key])
            else:
                assert response is not None, "response should not be None"
                response = cast(Response, response)
                response_gen_key = response.generation_key
                response_logits_key = response.logits_key
                response_item_key = response.item_key
                response_usage_key = response.usage_key
                # the choices list in the response is a flat one.
                # length is request.n * num_prompts
                current_choices = response.get_json_response()[response_gen_key][
=======
        all_input_prompts: List[Union[str, List[str], List[Dict]]] = []
        response_idx = 0
        number_prompts = len(cached_idx_to_response)
        single_completion_output = False
        if response:
            if isinstance(response.get_request_obj().prompt, str):
                single_completion_output = True
                number_prompts += 1
            elif isinstance(response.get_request_obj().prompt, list) and not isinstance(
                request, LMChatRequest
            ):
                number_prompts += len(response.get_request_obj().prompt)
            elif isinstance(response.get_request_obj().prompt, list) and isinstance(
                request, LMChatRequest
            ):
                assert len(cached_idx_to_response) <= 1
                number_prompts += 1
            else:
                raise ValueError(
                    f"Invalid prompt type: {type(response.get_request_obj().prompt)}"
                    f" with request type: {type(request)}"
                )
        response_type = None
        request_type: Type[Request] = None
        for idx in range(number_prompts):
            if idx in cached_idx_to_response:
                cached_res = cached_idx_to_response[idx]
                response_type = cached_res._response_type
                request_type = cached_res._request_type
                all_input_prompts.append(cached_res.get_request_obj().prompt)
                if request.n == 1:
                    assert (
                        len(cached_res.get_response_obj().choices) == 1
                    ), "cached response should have only one choice"
                all_model_choices.extend(cached_res.get_response_obj().choices)
                if cached_res.get_usage_obj().usages:
                    all_usages.extend(cached_res.get_usage_obj().usages)
            else:
                assert response is not None, "response should not be None"
                response = cast(Response, response)
                response_type = response._response_type
                request_type = response._request_type
                # the choices list in the response is a flat one.
                # length is request.n * num_prompts
                current_choices = response.get_response_obj().choices[
>>>>>>> upstream/main
                    response_idx * request.n : (response_idx + 1) * request.n
                ]
                all_model_choices.extend(current_choices)

<<<<<<< HEAD
                if isinstance(response.get_request()["prompt"], list):
                    prompt = response.get_request()["prompt"][response_idx]
                else:
                    prompt = str(response.get_request()["prompt"])
                if response_usage_key:
                    usage = response.get_json_response()[response_usage_key][
                        response_idx * request.n : (response_idx + 1) * request.n
                    ]
                    all_usages.extend(usage)
                all_input_prompts.append(prompt)
                # set cache
                new_request = copy.deepcopy(request)
                new_request.prompt = prompt
                cache_key = self.client.get_cache_key(new_request)
                new_response_key = copy.deepcopy(response.get_json_response())
                new_response_key[response_gen_key] = current_choices
                if response_usage_key:
                    new_response_key[response_usage_key] = usage
                self.cache.set(cache_key, new_response_key)
=======
                if isinstance(
                    response.get_request_obj().prompt, list
                ) and not isinstance(request, LMChatRequest):
                    prompt: Union[
                        str, List[str], List[Dict]
                    ] = response.get_request_obj().prompt[response_idx]
                # Chat request
                elif isinstance(response.get_request_obj().prompt, list) and isinstance(
                    request, LMChatRequest
                ):
                    # We will only have response_idx == 0 here as we can only
                    # support single chat requests.
                    assert request.n == 1
                    assert number_prompts <= 1
                    prompt = response.get_request_obj().prompt
                else:
                    prompt = str(response.get_request_obj().prompt)

                usages: Optional[List[Usage]] = None
                if response.get_usage_obj().usages:
                    usages = response.get_usage_obj().usages[
                        response_idx * request.n : (response_idx + 1) * request.n
                    ]
                    all_usages.extend(usages)
                all_input_prompts.append(prompt)
                # set cache
                new_request = copy.deepcopy(request)
                new_request.prompt = prompt  # type: ignore
                cache_key = client.get_cache_key(new_request)
                new_response = copy.deepcopy(response)
                new_response._response.choices = current_choices
                new_response._usages = Usages(usages=(usages or []))
                self.cache.set(cache_key, new_response.to_dict(drop_request=True))
>>>>>>> upstream/main
                response_idx += 1

        new_request = copy.deepcopy(request)
        new_request.prompt = (
<<<<<<< HEAD
            all_input_prompts
            if len(all_input_prompts) > 1 or not single_output
            else all_input_prompts[0]
        )
        new_response = {response_gen_key: all_model_choices}
        if response_usage_key:
            new_response[response_usage_key] = all_usages
        response_obj = Response(
            new_response,
            cached=len(cached_idx_to_response) > 0,
            request_params=self.client.get_cache_key(new_request),
            generation_key=response_gen_key,
            logits_key=response_logits_key,
            item_key=response_item_key,
            usage_key=response_usage_key,
=======
            all_input_prompts  # type: ignore
            if len(all_input_prompts) > 1 or not single_completion_output
            else all_input_prompts[0]
        )
        response_obj = Response(
            response=ModelChoices(choices=all_model_choices),
            cached=len(cached_idx_to_response) > 0,
            request=new_request,
            usages=Usages(usages=all_usages),
            response_type=response_type,
            request_type=request_type,
>>>>>>> upstream/main
        )
        return response_obj

    def run(
        self,
<<<<<<< HEAD
        prompt: Union[str, List[str]],
=======
        prompt: Union[str, List[str], List[Dict[str, str]]],
        overwrite_cache: bool = False,
        stop_token: Optional[str] = None,
        return_response: bool = False,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[
        str,
        List[str],
        np.ndarray,
        List[np.ndarray],
        Response,
        Iterator[str],
        Iterator[Response],
    ]:
        """
        Run the prompt.

        Orchestrates between the standard run and chat run and batch run.

        Args:
            prompt: prompt(s) to run.
            overwrite_cache: whether to overwrite cache.
            stop_token: stop token for prompt generation.
                Default is self.stop_token.
                "" for no stop token.
            return_response: whether to return Response object.
            stream: whether to stream the prompt. Only supported
                for single string prompts and LMs.

        Returns:
            response from prompt.
        """
        if not isinstance(prompt, list) and not isinstance(prompt, str):
            raise ValueError(
                f"Invalid prompt type: {type(prompt)}. "
                "Prompt must be a string or list of strings "
                "or list of dicts."
            )
        if isinstance(prompt, list) and not prompt:
            raise ValueError("Prompt cannot be empty list")
        # Get the client to run
        client = self.client_pool.get_next_client()
        if stream:
            if not client.supports_streaming_inference():
                raise ValueError(
                    f"Client {client} does not support streaming inference."
                )
            if not isinstance(prompt, str):
                raise ValueError(
                    "Stream is only supported for single string prompts. "
                    "It will soon be supported for chat dictionary prompts, too."
                )
            return self._run_stream(
                prompt=cast(str, prompt),
                client=client,
                overwrite_cache=overwrite_cache,
                stop_token=stop_token,
                return_response=return_response,
                **kwargs,
            )
        if isinstance(prompt, list) and isinstance(prompt[0], dict):
            if not client.IS_CHAT:
                raise ValueError(
                    f"Client {client} does not support dict chat prompt. "
                    "Please use a chat model."
                )
            if stop_token:
                logger.warning(
                    "stop_token is not supported for chat prompt. "
                    "Ignoring stop_token."
                )
            return self._run_chat(
                prompt=cast(List[Dict[str, str]], prompt),
                client=client,
                overwrite_cache=overwrite_cache,
                return_response=return_response,
                **kwargs,
            )
        return self._run(
            prompt=cast(Union[str, List[str]], prompt),
            client=client,
            overwrite_cache=overwrite_cache,
            stop_token=stop_token,
            return_response=return_response,
            **kwargs,
        )

    def _run(
        self,
        prompt: Union[str, List[str]],
        client: Client,
>>>>>>> upstream/main
        overwrite_cache: bool = False,
        stop_token: Optional[str] = None,
        return_response: bool = False,
        **kwargs: Any,
    ) -> Union[str, List[str], np.ndarray, List[np.ndarray], Response]:
        """
        Run the prompt.

        Args:
            prompt: prompt(s) to run.
<<<<<<< HEAD
=======
            client: client to run.
>>>>>>> upstream/main
            overwrite_cache: whether to overwrite cache.
            stop_token: stop token for prompt generation.
                        Default is self.stop_token.
                        "" for no stop token.
            return_response: whether to return Response object.

        Returns:
            response from prompt.
        """
        is_batch = isinstance(prompt, list)
<<<<<<< HEAD

        stop_token = stop_token if stop_token is not None else self.stop_token
        # Must pass kwargs as dict for client "pop" methods removed used arguments
        request_params = self.client.get_request(prompt, kwargs)
=======
        stop_token = stop_token if stop_token is not None else self.stop_token
        # Must pass kwargs as dict for client "pop" methods removed used arguments
        request_params = client.get_request(prompt, kwargs)
>>>>>>> upstream/main
        # Avoid nested list of results - enforce n = 1 for batch
        if is_batch and request_params.n > 1:
            raise ValueError("Batch mode does not support n > 1.")
        self._validate_kwargs(kwargs, request_params)

        cached_idx_to_response, request_params = self._split_cached_requests(
<<<<<<< HEAD
            request_params, overwrite_cache
        )
        # If not None value or empty list - run new request
        if request_params.prompt:
            response = self.client.run_request(request_params)
=======
            request_params, client, overwrite_cache
        )
        # If not None value or empty list - run new request
        if request_params.prompt:
            # Start timing metrics
            self.client_pool.start_timer()
            response = client.run_request(request_params)
            self.client_pool.end_timer()
>>>>>>> upstream/main
        else:
            # Nothing to run
            response = None

        final_response = self._stitch_responses_and_cache(
            request=request_params,
<<<<<<< HEAD
            response=response,
            cached_idx_to_response=cached_idx_to_response,
        )

=======
            client=client,
            response=response,
            cached_idx_to_response=cached_idx_to_response,
        )
>>>>>>> upstream/main
        # Extract text results
        if return_response:
            return final_response
        else:
            return final_response.get_response(stop_token, is_batch)

<<<<<<< HEAD
=======
    def _run_chat(
        self,
        prompt: List[Dict[str, str]],
        client: Client,
        overwrite_cache: bool = False,
        return_response: bool = False,
        **kwargs: Any,
    ) -> Union[str, Response]:
        """
        Run the prompt.

        Args:
            prompt: prompt dictionary to run.
            client: client to run.
            overwrite_cache: whether to overwrite cache.
            stop_token: stop token for prompt generation.
                        Default is self.stop_token.
                        "" for no stop token.
            return_response: whether to return Response object.

        Returns:
            response from prompt.
        """
        is_batch = False
        # Get a request for an empty prompt to handle all kwargs
        request_params = client.get_request("", kwargs)
        # Add prompt and cast as chat request
        request_params_dict = request_params.to_dict()
        request_params_dict["prompt"] = prompt
        request_params_as_chat = LMChatRequest(**request_params_dict)
        # Avoid nested list of results - enforce n = 1 for batch
        if request_params_as_chat.n > 1:
            raise ValueError("Chat mode does not support n > 1.")
        self._validate_kwargs(kwargs, request_params_as_chat)

        cached_idx_to_response, request_params_as_chat = self._split_cached_requests(  # type: ignore # noqa: E501
            request_params_as_chat, client, overwrite_cache
        )
        # If not None value or empty list - run new request
        if request_params_as_chat.prompt:
            # Start timing metrics
            self.client_pool.start_timer()
            response = client.run_chat_request(request_params_as_chat)
            self.client_pool.end_timer()
        else:
            # Nothing to run
            response = None

        final_response = self._stitch_responses_and_cache(
            request=request_params_as_chat,
            client=client,
            response=response,
            cached_idx_to_response=cached_idx_to_response,
        )

        # Extract text results
        if return_response:
            return final_response
        else:
            return cast(str, final_response.get_response("", is_batch))

    def _run_stream(
        self,
        prompt: str,
        client: Client,
        overwrite_cache: bool = False,
        stop_token: Optional[str] = None,
        return_response: bool = False,
        **kwargs: Any,
    ) -> Union[Generator[str, None, None], Generator[Response, None, None]]:
        """
        Run the prompt in a stream.

        Args:
            prompt: prompt(s) to run.
            client: client to run.
            overwrite_cache: whether to overwrite cache.
            stop_token: stop token for prompt generation.
                        Default is self.stop_token.
                        "" for no stop token.
            return_response: whether to return Response object.

        Returns:
            response from prompt.
        """
        is_batch = False
        stop_token = stop_token if stop_token is not None else self.stop_token
        # Must pass kwargs as dict for client "pop" methods removed used arguments
        request_params = client.get_request(prompt, kwargs)
        # Avoid nested list of results - enforce n = 1 for batch
        if request_params.n > 1:
            raise ValueError("Stream mode does not support n > 1.")
        self._validate_kwargs(kwargs, request_params)

        cached_idx_to_response, request_params = self._split_cached_requests(
            request_params, client, overwrite_cache
        )
        if request_params.prompt:
            # Because we are streaming, we should have either a cached response
            # a prompt to run
            assert len(cached_idx_to_response) == 0
            response_iter = client.run_streaming_request(request_params)
            is_cached = False
        else:
            assert len(cached_idx_to_response) == 1
            response_iter = cached_idx_to_response[0].as_iter()
            is_cached = True

        saved_responses = []
        # Start timing metrics
        self.client_pool.start_timer()
        for response_token in response_iter:
            saved_responses.append(response_token)
            if return_response:
                yield response_token
            else:
                yield cast(
                    Union[str, Response], response_token.get_response("", is_batch)
                )
        self.client_pool.end_timer()

        if not is_cached:
            final_response = Response.union_all(
                saved_responses, as_single_lmchoice=True
            )
            self._stitch_responses_and_cache(
                request=request_params,
                client=client,
                response=final_response,
                cached_idx_to_response=cached_idx_to_response,
            )

>>>>>>> upstream/main
    async def arun_batch(
        self,
        prompts: List[str],
        overwrite_cache: bool = False,
        stop_token: Optional[str] = None,
        return_response: bool = False,
<<<<<<< HEAD
=======
        chunk_size: int = -1,
        verbose: bool = False,
>>>>>>> upstream/main
        **kwargs: Any,
    ) -> Union[List[str], List[np.ndarray], Response]:
        """
        Run a batch of prompts with async.

<<<<<<< HEAD
=======
        If the client pool is a single client, all prompts will be sent
        to one client and batch_size (which is passed it as kwargs) will
        determine how the prompts are split.

        If the client pool is a pool of clients, the prompts will be split
        into chunks and sent to the clients. Each client will split the
        chunk into batch_size prompts to send to the model.

>>>>>>> upstream/main
        Args:
            prompts: prompts to run.
            overwrite_cache: whether to overwrite cache.
            stop_token: stop token for prompt generation.
<<<<<<< HEAD
                        Default is self.stop_token.
                        "" for no stop token.
            return_response: whether to return Response object.
=======
                Default is self.stop_token.
                "" for no stop token.
            return_response: whether to return Response object.
            chunk_size: number of prompts to send to a client in chunks.
                For each chunk, the client will split the chunk into
                batch_sized prompts to send to the model.
                For a single manifest client, there is no impact to
                setting chunk_size. For a client pool, chunk_size
                can be used to distribute the load across the clients.
            verbose: whether to print progress of async tasks.
>>>>>>> upstream/main

        Returns:
            response from prompt.
        """
<<<<<<< HEAD
        stop_token = stop_token if stop_token is not None else self.stop_token
        # Must pass kwargs as dict for client "pop" methods removed used arguments
        request_params = self.client.get_request(prompts, kwargs)
=======
        if not isinstance(prompts, list):
            raise ValueError("Prompts must be a list of strings.")
        if not prompts:
            raise ValueError("Prompts must not be empty.")
        if not isinstance(prompts[0], str):
            raise ValueError("Prompts must be a list of strings.")

        # Split the prompts into chunks for connection pool
        prompt_chunks: List[Tuple[Client, List[str]]] = []
        if chunk_size > 0:
            for i in range(0, len(prompts), chunk_size):
                prompt_chunks.append(
                    (self.client_pool.get_next_client(), prompts[i : i + chunk_size])
                )
        else:
            prompt_chunks = [(self.client_pool.get_next_client(), prompts)]

        # Run the chunks
        tasks = []
        for client, chunk in prompt_chunks:
            tasks.append(
                asyncio.create_task(
                    self._arun_batch_client(
                        prompts=chunk,
                        client=client,
                        overwrite_cache=overwrite_cache,
                        verbose=verbose,
                        **kwargs,
                    )
                )
            )
        logger.info(f"Running {len(tasks)} tasks across all clients.")
        responses = await asyncio.gather(*tasks)
        final_response = Response.union_all(responses)
        stop_token = stop_token if stop_token is not None else self.stop_token

        # Extract text results
        if return_response:
            return final_response
        else:
            return cast(
                Union[List[str], List[np.ndarray]],
                final_response.get_response(stop_token, True),
            )

    async def _arun_batch_client(
        self,
        prompts: List[str],
        client: Client,
        overwrite_cache: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Response:
        """
        Run a batch of prompts with async for single client.

        Args:
            prompts: prompts to run.
            client: client to run.
            overwrite_cache: whether to overwrite cache.
            verbose: whether to print progress of async tasks.

        Returns:
            response from prompt.
        """
        # Must pass kwargs as dict for client "pop" methods removed used arguments
        request_params = client.get_request(prompts, kwargs)
>>>>>>> upstream/main
        # Avoid nested list of results - enforce n = 1 for batch
        if request_params.n > 1:
            raise ValueError("Batch mode does not support n > 1.")
        self._validate_kwargs(kwargs, request_params)

        cached_idx_to_response, request_params = self._split_cached_requests(
<<<<<<< HEAD
            request_params, overwrite_cache
        )
        # If not None value or empty list - run new request
        if request_params.prompt:
            response = await self.client.arun_batch_request(request_params)
=======
            request_params, client, overwrite_cache
        )
        # If not None value or empty list - run new request
        if request_params.prompt:
            self.client_pool.start_timer()
            response = await client.arun_batch_request(request_params, verbose=verbose)
            self.client_pool.end_timer()
>>>>>>> upstream/main
        else:
            # Nothing to run
            response = None

        final_response = self._stitch_responses_and_cache(
            request=request_params,
<<<<<<< HEAD
            response=response,
            cached_idx_to_response=cached_idx_to_response,
        )

        # Extract text results
        if return_response:
            return final_response
        else:
            return cast(
                Union[List[str], List[np.ndarray]],
                final_response.get_response(stop_token, True),
            )
=======
            client=client,
            response=response,
            cached_idx_to_response=cached_idx_to_response,
        )
        return final_response
>>>>>>> upstream/main

    def score_prompt(
        self,
        prompt: Union[str, List[str]],
        overwrite_cache: bool = False,
        **kwargs: Any,
    ) -> Dict:
        """
        Score the prompt via forward pass of the model - no sampling or generation.

        Returns the response object with logits of the prompt.

        Args:
            prompt: prompt(s) to run.
            overwrite_cache: whether to overwrite cache.

        Returns:
            response from prompt.
        """
<<<<<<< HEAD
        # Must pass kwargs as dict for client "pop" methods removed used arguments
        request_params = self.client.get_request(prompt, kwargs)
        request_params.request_type = "score_prompt"

        if request_params.n > 1:
            raise ValueError("Sequence scoring does not support n > 1.")
        self._validate_kwargs(kwargs, request_params)

        cached_idx_to_response, request_params = self._split_cached_requests(
            request_params, overwrite_cache
        )
        # If not None value or empty list - run new request
        if request_params.prompt:
            try:
                response = cast(
                    HuggingFaceClient, self.client
                ).get_score_prompt_request(request_params)
=======
        client = self.client_pool.get_next_client()
        # Must pass kwargs as dict for client "pop" methods removed used arguments
        request_params = client.get_request(prompt, kwargs)
        request_params_as_score = LMScoreRequest(**request_params.to_dict())

        if request_params_as_score.n > 1:
            raise ValueError("Sequence scoring does not support n > 1.")
        self._validate_kwargs(kwargs, request_params_as_score)

        cached_idx_to_response, request_params_as_score = self._split_cached_requests(  # type: ignore # noqa: E501
            request_params_as_score, client, overwrite_cache
        )
        # If not None value or empty list - run new request
        if request_params_as_score.prompt:
            try:
                response = cast(HuggingFaceClient, client).run_score_prompt_request(
                    request_params_as_score
                )
>>>>>>> upstream/main
            except AttributeError:
                raise ValueError("`score_prompt` only supported for HF models.")
        else:
            # Nothing to run
            response = None

        final_response = self._stitch_responses_and_cache(
<<<<<<< HEAD
            request=request_params,
=======
            request=request_params_as_score,
            client=client,
>>>>>>> upstream/main
            response=response,
            cached_idx_to_response=cached_idx_to_response,
        )
        return final_response.to_dict()
