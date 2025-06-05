import os
import queue
from collections import defaultdict
from typing import Any, List

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM

from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils import PythonExecutor,excute_codes

from .utils import ray_noset_visible_devices
import copy
logger = init_logger(__name__)


@ray.remote
def get_all_env_variables():
    import os

    return os.environ


@ray.remote
class LLMRayActor:

    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        noset_visible_devices = kwargs.pop("noset_visible_devices")
        if kwargs.get("distributed_executor_backend") == "ray":
            # a hack to make the script work.
            # stop ray from manipulating CUDA_VISIBLE_DEVICES
            # at the top-level when the distributed_executor_backend is ray.
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        elif noset_visible_devices:
            # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
            # when the distributed_executor_backend is not ray and
            # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

        num_gpus = kwargs.pop("num_gpus")
        if bundle_indices is not None:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            print(f"creating LLM with bundle_indices={bundle_indices}")

        # Number of actors that will send prompt to this engine
        self.num_actors = kwargs.pop("num_actors")
        self.actor_counter = 0
        self.requests = {}
        self.response_queues = defaultdict(queue.Queue)

        self.llm = LLM(*args, **kwargs)

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray):
        return self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return self.llm.collective_rpc("update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache))

    def reset_prefix_cache(self):
        self.llm.llm_engine.reset_prefix_cache()

    def sleep(self, level=1):
        self.llm.sleep(level=level)

    def wake_up(self):
        self.llm.wake_up()

    def add_requests(self, actor_rank, *, sampling_params, vllm_vision_input):
        """
        Save the requests from actors and generate responses when all actors have sent their requests
        """
        self.requests[actor_rank] = vllm_vision_input
        self.actor_counter += 1
        if self.actor_counter == self.num_actors:
            assert len(self.requests) == self.num_actors
            num_requests = []
            requests = []
            for actor_rank, request in self.requests.items():
                num_requests.append((actor_rank, len(request)))
                requests.extend(request)
            if len(requests) > 0:
                # For now we assume that all requests have the same sampling params
                # responses = self.llm.generate(requests, sampling_params=sampling_params)
                if len(sampling_params.stop)==0:
                    responses = self.llm.generate(requests, sampling_params=sampling_params)
                else:
                    responses = self.generate_code_exec_batch(request, sampling_params)
                    # responses=self.generate_code_exec(request, sampling_params)
            else:
                responses = []
            
            offset = 0
            self.responses = {}
            for actor_rank, num in num_requests:
                self.response_queues[actor_rank].put(responses[offset : offset + num])
                offset += num

            self.actor_counter = 0
            self.requests = {}

    def get_responses(self, actor_rank):
        """
        Return the responses for the actor with the given rank
        """
        return self.response_queues[actor_rank].get()
    def update_response(self,new_response, fini_response,code_res):
        old_response_text = fini_response.outputs[0].text
        fini_response.outputs[0].text += code_res + new_response.outputs[0].text
        tmp_tokenizer = self.llm.get_tokenizer()
        # fini_response.outputs[0].token_ids += new_response.outputs[0].token_ids

        # fini_response.outputs[0].token_ids = tmp_tokenizer.encode(fini_response.outputs[0].text)

        old_response_tokenids = tmp_tokenizer.encode(old_response_text)
        new_response_tokenids = tmp_tokenizer.encode(new_response.outputs[0].text)

        strat_interpreter = "<interpreter>"
        end_interpreter = "</interpreter>"
        code_res_list = code_res.split(strat_interpreter)
        precode_res = code_res_list[0]
        midcode_res, endcode_res = code_res_list[-1].split(end_interpreter)

        code_res_lst = [precode_res,strat_interpreter,midcode_res,end_interpreter,endcode_res]
        code_res_tokenids = []
        for tmp in code_res_lst:
            code_res_tokenids += tmp_tokenizer.encode(tmp)
        fini_response_tokenids = old_response_tokenids + code_res_tokenids + new_response_tokenids
        fini_response.outputs[0].token_ids = fini_response_tokenids

        max_model_len_llm = self.llm.llm_engine.model_config.max_model_len
        new_prompts_length = len(fini_response.outputs[0].token_ids) + len(fini_response.prompt_token_ids)
        return fini_response, new_prompts_length >= max_model_len_llm
    def generate_code_exec_batch(self, requests, sampling_params):
        executor = PythonExecutor()
        responses=self.llm.generate(requests, sampling_params=sampling_params)
        final_responses = copy.deepcopy(responses)
        
        pred_stop_reason_lst = [response.outputs[0].stop_reason for response in responses]
        first_time = True
        max_code_exec_times = 5
        code_exec_times = 0
        while (first_time or any(
            [
                pred_stop_reason is not None for pred_stop_reason in pred_stop_reason_lst
            ]) and code_exec_times <max_code_exec_times
        ):
            code_exec_times+=1
            first_time = False
            # 批量处理请求
            new_requests = []
            code_to_execute_lst = []
            for idx_res, request in enumerate(requests):
                if pred_stop_reason_lst[idx_res] is not None: # 最近的输出中有代码
                    response = responses[idx_res]
                    response_text = response.outputs[0].text
                    
                    code_to_execute = response_text.replace("</code>","").split("```python")[-1].replace("```", "").strip()

                    tmp_tokenizer = self.llm.get_tokenizer()
                    max_model_len_llm = self.llm.llm_engine.model_config.max_model_len
                    # prompt_len = len(tmp_tokenizer.encode(request["prompt"]))
                    # code_lenth = len(tmp_tokenizer.encode())
                    prompt_len = len(response.prompt_token_ids)
                    new_prompts_length = len(response.outputs[0].token_ids) + prompt_len + 150 # 代码执行结果留35 tokensbuffer
                    if new_prompts_length >= max_model_len_llm:
                        # import pdb;pdb.set_trace()
                        pred_stop_reason_lst[idx_res] = None
                        continue
                    new_requests.append(request)
                    new_requests[-1]["prompt"] += response_text
                    code_to_execute_lst.append(code_to_execute)
                else:  # 输出到eos终止，不需要处理，最终响应已存在final_responses中
                    pass
            batch_results_include_none = []
            batch_results, no_code_idx = excute_codes(
                    code_to_execute_lst, executor=executor)
            # import pdb;pdb.set_trace()
            batch_execu_res_txt = []
            assert len(new_requests) == len(code_to_execute_lst), f"length new_requests:{len(new_requests)} not equal to length code_to_execute_lst: {len(code_to_execute_lst)}"
            prompt_length_skip=[]
            
            for idx_code in range(len(code_to_execute_lst)):
                if idx_code in no_code_idx:
                    batch_results_include_none.append(None)
                    execu_content = "None"
                else:
                    batch_results_include_none.append(batch_results.pop(0))
                    exe_result=batch_results_include_none[-1]
                    output, report = exe_result
                    if report == "Done":
                        execu_content = output
                    else:
                        execu_content = report
                tmp_code_res = "\n" + "<interpreter>\n" + execu_content + "</interpreter>\n"
                new_requests[idx_code]["prompt"] +=  tmp_code_res
                batch_execu_res_txt.append(tmp_code_res)

                tmp_tokenizer = self.llm.get_tokenizer()
                tmp_code_res_ids_length = len(tmp_tokenizer.encode(tmp_code_res))
                prompt_tokenids_len = len(tmp_tokenizer.encode(new_requests[idx_code]["prompt"]))
                max_model_len_llm = self.llm.llm_engine.model_config.max_model_len
                if prompt_tokenids_len >= max_model_len_llm: # 理论上只有第一次可能会发生，即一次generate后长度过长无法进行第二次generate
                    logger.info("text info: %r, length ress %d, length prompt %d, max_model_len: %d, ", 
                                    new_requests[idx_code]["prompt"], 
                                    tmp_code_res_ids_length,
                                    prompt_tokenids_len, 
                                    max_model_len_llm)
                #     prompt_length_skip.append(idx_code)
                #     import pdb;pdb.set_trace()
                #     new_requests.pop(idx_code)

            #每一轮后，new_requests中请求的数量都会减少，只对有代码生成的部分继续进行推理
            # sampling_params_new = copy.deepcopy(sampling_params)
            
            new_responses = self.llm.generate(new_requests, sampling_params=sampling_params)
            tmp_new_res_cnt = 0
            for idx, stop_reason in enumerate(pred_stop_reason_lst):
                if stop_reason is not None:  #上次输出被</code>终止，将本次response和原来的concat做为最终输出
                    if tmp_new_res_cnt in prompt_length_skip:  # 理论上只有第一次可能会发生
                        pred_stop_reason_lst[idx]=None
                        continue
                    tmp_response = new_responses[tmp_new_res_cnt]
                    tmp_exec_text = batch_execu_res_txt[tmp_new_res_cnt]
                    tmp_new_res_cnt += 1
                    pred_stop_reason_lst[idx] = tmp_response.outputs[0].stop_reason # 更新stop_reason状态

                    tmp_fini_response = final_responses[idx]
                    #TODO 用新的输出更新原有输出  
                    #0528 TODO 将代码执行结果加入
                    final_responses[idx],prompt_over_length = self.update_response(tmp_response, tmp_fini_response,tmp_exec_text)
                    # 更新stop_reason
                    pred_stop_reason_lst[idx] = tmp_response.outputs[0].stop_reason
                    if prompt_over_length:
                        pred_stop_reason_lst[idx]=None
                    #更新原有response和request
                    responses[idx] = tmp_response
                    requests[idx] = new_requests[tmp_new_res_cnt-1]
            
        return final_responses
    def generate_code_exec(self, request, sampling_params):
        """
        Generate a response for the given request
        """
        executor = PythonExecutor()
        responses=self.llm.generate(request, sampling_params=sampling_params)
        final_responses = []
        final_code_num_lst = []
        response_idx = 0
        for response, prompt in zip(responses, request): #针对多个prompt
            response_idx += 1
            code_num_lst = [0 for _ in range(len(response.outputs))]
            intermediate_responses = [prompt for _ in range(len(response.outputs))]
            fini_responses = []
            pred_stop_reason_lst = [
                [output.text, output.stop_reason, output.token_ids] for output in response.outputs
            ]
            inter_responses_tmp = response
            first_time = True # 确保必进入一次
            while first_time or any(
                [
                    pred_stop_reason is not None
                    for pred_stop_reason in pred_stop_reason_lst
                ]
            ):  #每个prompt可能会因为num_return_sequences等出现一个prompt多个response，这样response.ouputs长度>1的list
                first_time = False
                code_to_execute_lst = []
                assert len(pred_stop_reason_lst) == len(intermediate_responses)
                for res_idx in range(len(pred_stop_reason_lst)):
                    pred_stop_reason = pred_stop_reason_lst[res_idx]
                    inter_response = intermediate_responses[res_idx]
                    if inter_response is None:
                        continue
                    pred, stop_reason = pred_stop_reason[:2]
                    if stop_reason != "</code>":
                        inter_response["prompt"] = inter_response["prompt"] + pred
                        #TODO fini_response需要修改
                        # import pdb;pdb.set_trace()
                        if inter_responses_tmp.outputs[0].text != pred: # 第一次不做处理
                            tmp_text = inter_responses_tmp.outputs[0].text + pred
                            inter_responses_tmp.outputs[0].text = tmp_text
                            inter_responses_tmp.outputs[0].token_ids = self.llm.get_tokenizer().encode(tmp_text)

                        # fini_responses.append(inter_response)
                        fini_responses.append(inter_responses_tmp)
                        pred_stop_reason_lst[res_idx] = None
                        intermediate_responses[res_idx] = None
                        continue
                    else:
                        code_to_execute_lst.append(pred.replace("</code>","").split("```python")[-1].replace("```", "").strip())
                        inter_response["prompt"] = inter_response["prompt"] + pred
                        intermediate_responses[res_idx] = inter_response

                new_intermediate_responses = None
                if len(code_to_execute_lst) == 0:
                    break
                batch_results, no_code_idx = excute_codes(
                    code_to_execute_lst, executor=executor
                )
                batch_results_include_none = []
                for i in range(len(code_to_execute_lst)):
                    if i in no_code_idx:
                        batch_results_include_none.append(None)
                    else:
                        batch_results_include_none.append(batch_results.pop(0))
                for i, inter_response in enumerate(intermediate_responses):
                    if inter_response is None:
                        continue
                    exe_result = batch_results_include_none.pop(0)
                    if exe_result is None:
                        excu_content = "None"
                    else:
                        output, report = exe_result
                        if report == "Done":
                            excu_content = output
                        else:
                            excu_content = report
                    inter_response["prompt"] = inter_response["prompt"] + "\n" + "<interpreter>\n" + excu_content + "</interpreter>\n\n"
                    intermediate_responses[i] = inter_response

                    tmp_text = inter_responses_tmp.outputs[0].text + "\n" + "<interpreter>\n" + excu_content + "</interpreter>\n\n"
                    inter_responses_tmp.outputs[0].text = tmp_text
                    tmp_token_ids = self.llm.get_tokenizer().encode(tmp_text)
                    inter_responses_tmp.outputs[0].token_ids = tmp_token_ids
                    
                    max_model_len_llm = self.llm.llm_engine.model_config.max_model_len
                    prompt_token_ids = self.llm.get_tokenizer().encode(inter_response["prompt"])
                    if len(prompt_token_ids) > max_model_len_llm:
                        intermediate_responses[i] = None
                        # logger.info(f"text info: {inter_response['prompt']}, length {len(prompt_token_ids)}, max_model_len: {max_model_len_llm}")
                        logger.info("text info: %r, length %d, max_model_len: %d", 
                                    inter_response['prompt'], 
                                    len(prompt_token_ids), 
                                    max_model_len_llm)

                    # import pdb;pdb.set_trace()
                    '''
                    max_model_len_llm = self.llm.llm_engine.model_config.max_model_len
                    inter_responses_tmp.outputs[0].token_ids = tmp_token_ids
                    if len(tmp_token_ids) > max_model_len_llm:
                        logger.info(f"text info: {tmp_text}, length {len(tmp_token_ids)}, max_model_len: {max_model_len_llm}")
                        tmp_token_ids = tmp_token_ids[:max_model_len_llm]
                        inter_responses_tmp.outputs[0].token_ids = tmp_token_ids
                        tmp_text=self.llm.get_tokenizer().decode(tmp_token_ids)
                        inter_responses_tmp.outputs[0].text = tmp_text
                    '''    
                   
                intermediate_responses_to_gen = [
                    inter_response
                    for inter_response in intermediate_responses
                    if inter_response is not None
                ]
                tmp_sampling_params = copy.deepcopy(sampling_params)
                # tmp_sampling_params["
                new_intermediate_responses = self.llm.generate(
                    intermediate_responses_to_gen, sampling_params=sampling_params
                )
                tmp_cnt = 0
                for new_i, pred_stop_reason in enumerate(pred_stop_reason_lst):
                    if pred_stop_reason is not None:
                        tmp_output = new_intermediate_responses[tmp_cnt].outputs.pop(0)
                        tmp_cnt += 1
                        pred_stop_reason_lst[new_i] = [
                            tmp_output.text,
                            tmp_output.stop_reason,
                            tmp_output.token_ids,
                        ]
                        code_num_lst[new_i] += 1
            if len(fini_responses) == 1:
                final_responses.append(fini_responses[-1])
                final_code_num_lst.append(code_num_lst)
            else:
                final_responses.append(fini_responses)
        return final_responses


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    num_total_actors: int,
    shared_pg=None,
    gpu_memory_utilization=None,
    vllm_enable_sleep=False,
):
    import vllm

    assert vllm.__version__ >= "0.7.2", "OpenRLHF only supports vllm >= 0.7.2"

    vllm_engines = []

    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    use_hybrid_engine = shared_pg is not None
    num_gpus = int(tensor_parallel_size == 1)
    if use_hybrid_engine and tensor_parallel_size == 1:
        # every worker will use 0.2 GPU, so that we can schedule
        # 2 instances on the same GPUs.
        num_gpus = 0.2

    if not use_hybrid_engine:
        # Create a big placement group to ensure that all engines are packed
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_engines * tensor_parallel_size)]
        shared_pg = placement_group(bundles, strategy="PACK")
        ray.get(shared_pg.ready())

    for i in range(num_engines):
        bundle_indices = None
        if tensor_parallel_size > 1:
            bundle_indices = list(range(i * tensor_parallel_size, (i + 1) * tensor_parallel_size))

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=shared_pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=i * tensor_parallel_size,
        )

        if num_engines >= num_total_actors:
            num_actors = 1
        else:
            num_actors = num_total_actors // num_engines + int(i < num_total_actors % num_engines)

        vllm_engines.append(
            LLMRayActor.options(
                num_cpus=num_gpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                model=pretrain,
                enforce_eager=enforce_eager,
                worker_cls="openrlhf.trainer.ray.vllm_worker_wrap.WorkerWrap",
                tensor_parallel_size=tensor_parallel_size,
                seed=seed + i,
                distributed_executor_backend=distributed_executor_backend,
                max_model_len=max_model_len,
                enable_prefix_caching=enable_prefix_caching,
                dtype="bfloat16",
                trust_remote_code=True,
                num_actors=num_actors,
                gpu_memory_utilization=gpu_memory_utilization,
                bundle_indices=bundle_indices,
                num_gpus=0.2 if use_hybrid_engine else 1,
                enable_sleep_mode=vllm_enable_sleep,
                noset_visible_devices=ray_noset_visible_devices(),
            )
        )

    if vllm_enable_sleep:
        batch_vllm_engine_call(vllm_engines, "sleep", rank_0_only=False)

    return vllm_engines


def batch_vllm_engine_call(engines: List[Any], method_name: str, *args, rank_0_only: bool = True, **kwargs):
    """
    Batch call a method on multiple vLLM engines.
    Args:
        engines: List of vLLM engine instances
        method_name: Name of the method to call
        rank_0_only: Only execute on rank 0 if True
        *args: Positional arguments to pass to the method
        **kwargs: Keyword arguments to pass to the method
    Returns:
        List of results from ray.get() if on rank 0, None otherwise
    """
    import torch

    if rank_0_only and torch.distributed.get_rank() != 0:
        return None

    refs = []
    for engine in engines:
        method = getattr(engine, method_name)
        refs.append(method.remote(*args, **kwargs))

    return ray.get(refs)
