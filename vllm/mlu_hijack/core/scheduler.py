import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union
from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.policy import Policy
from vllm.core.scheduler import (SchedulingBudget, SchedulerPrefillOutputs,
                                 SchedulerRunningOutputs, SchedulerSwappedInOutputs,
                                 SchedulerOutputs, Scheduler)
from vllm.sequence import SequenceGroup, Sequence
from vllm.mlu_hijack.mlu_hijack_utils import MluHijackObject
from vllm.logger import init_logger

logger = init_logger(__name__)


vllm__core__scheduler__Scheduler____init____org = Scheduler.__init__
vllm__core__scheduler__Scheduler___schedule_prefills__org = Scheduler._schedule_prefills
vllm__core__scheduler__Scheduler___schedule_running__org = Scheduler._schedule_running
vllm__core__scheduler__Scheduler___schedule__org = Scheduler._schedule


def vllm__core__scheduler__Scheduler__init_scheduler_view(self):
    logger.info(f"vLLM scheduler profiling start...")

    self.df = pd.DataFrame(
        data={
            'waiting': [], 'running': [], 'swapped': [], 'finished': [],
            'wait_to_run_reqs': [], 'run_to_wait_reqs': [], 'wait_to_run_tokens': [],
            'batch_utils': [], 'block_utils': [], 'preempt_ratio': []
        },
        dtype=np.float32
    )
    self.sched_step = 0
    self.running_seqs = 0
    self.waiting_seqs = 0
    self.swapped_seqs = 0
    self.finished_seqs = 0
    self.total_seqs = 0
    self.running_to_waiting_seqs = 0
    self.waiting_to_running_seqs = 0
    self.wait_to_run_tokens = 0
    self.batch_utils = 0
    self.block_utils = 0
    self.preempt_ratio = 0

    self.finished_seq_groups = []


def summary_finished_seq_groups(seq_groups: List[SequenceGroup]):
    df = pd.DataFrame(
        data={
            'ttft/s': [], 'time_in_queue/s': [], 'context_latency/s': [], 'decoder_latency/s': []
        },
        dtype=np.float32
    )
    for seq_group in seq_groups:
        ttft = seq_group.metrics.first_token_time - seq_group.metrics.arrival_time
        time_in_queue = seq_group.metrics.time_in_queue
        context_latency = seq_group.metrics.first_token_time - seq_group.metrics.first_scheduled_time
        decoder_latency = seq_group.metrics.finished_time - seq_group.metrics.first_token_time
        decoder_token_num = seq_group.get_seqs()[0].get_output_len() - 1
        per_token_latency = decoder_latency if decoder_token_num == 0 \
                                            else decoder_latency / decoder_token_num
        df_ = pd.DataFrame(
            [[ttft, time_in_queue, context_latency, decoder_latency, per_token_latency, decoder_token_num]],
            columns=['ttft/s', 'time_in_queue/s', 'context_latency/s', 'decoder_latency/s', 'per_token_latency/s', 'decoder_tokens'],
            index=[str(seq_group.request_id)]
        )
        df = pd.concat([df, df_])
    sum_, max_, mean_, min_, p99_ = df.sum(), df.max(), df.mean(), df.min(), df.quantile(0.99)
    df.loc['Sum'] = sum_
    df.loc['Max'] = max_
    df.loc['Mean'] = mean_
    df.loc['Min'] = min_
    df.loc['P99'] = p99_
    return df


def vllm__core__scheduler__Scheduler__save_scheduler_view(self, scheduler_idx=0):
    logger.info(f"vLLM scheduler profiling save...")
    plt.rcParams.update({'font.size': 8})
    figure = plt.figure(figsize=(6.4, 5.6))
    gs = figure.add_gridspec(3, hspace=0)
    axes = gs.subplots(sharex=True, sharey=False)
    figure.suptitle("Cambricon vLLM Scheduler View")
    # scheduler queue view
    self.df.plot(ax=axes[0], y=['waiting', 'running', 'swapped', 'finished'])
    axes[0].set_xlabel('X-LLMEngineStep', loc='left')
    axes[0].set_ylabel('Y-ReqNum', loc='top')
    # utilization
    self.df.plot(ax=axes[1], y=['batch_utils', 'block_utils', 'preempt_ratio'])
    axes[1].set_xlabel('X-LLMEngineStep', loc='left')
    axes[1].set_ylabel('Y-Utilization(%)', loc='top')
    # token view
    self.df.plot(ax=axes[2], y=['wait_to_run_tokens'])
    axes[2].set_xlabel('X-LLMEngineStep', loc='left')
    axes[2].set_ylabel('Y-TokenNum', loc='top')
    for ax in axes:
        ax.label_outer()
        ax.legend(loc='upper right')
    figure.tight_layout()
    figure.savefig(f"vllm_scheduler{scheduler_idx}_view.svg", dpi=300, format='svg')
    plt.close(figure)

    time_df = summary_finished_seq_groups(self.finished_seq_groups)

    sched_df = self.df.copy(deep=True)
    max_, mean_, min_ = sched_df.max(), sched_df.mean(), sched_df.min()
    sched_df.loc["Max"] = max_
    sched_df.loc["Mean"] = mean_
    sched_df.loc["Min"] = min_
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.max_colwidth', None,
                           'display.float_format', '{:^6,.2f}'.format,
                           'expand_frame_repr', False):
        logger.info(sched_df.loc[["Max", "Mean", "Min"]])
        logger.info(time_df.loc[["Sum", "Max", "Mean", "Min", "P99"]])
    sched_df.astype(str).to_csv(f"vllm_scheduler{scheduler_idx}_step_view.csv", mode="w")
    time_df.astype(str).to_csv(f"vllm_scheduler{scheduler_idx}_reqs_view.csv", mode="w")


def vllm__core__scheduler__Scheduler____init__(
    self,
    scheduler_config: SchedulerConfig,
    cache_config: CacheConfig,
    lora_config: Optional[LoRAConfig],
    pipeline_parallel_size: int = 1,
) -> None:
    vllm__core__scheduler__Scheduler____init____org(
        self=self,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        lora_config=lora_config,
        pipeline_parallel_size=pipeline_parallel_size
    )
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: add for scheduler profiling
    '''
    self.init_scheduler_view()
    '''
    ==================
    End of MLU Hijack
    ==================
    '''


def vllm__core__scheduler__Scheduler___schedule_prefills(
    self,
    waiting_queue: deque,
    budget: SchedulingBudget,
    curr_loras: Optional[Set[int]],
    enable_chunking: bool = False,
) -> Tuple[deque, SchedulerPrefillOutputs]:
    remaining_waiting, prefills = vllm__core__scheduler__Scheduler___schedule_prefills__org(
        self=self,
        waiting_queue=waiting_queue,
        budget=budget,
        curr_loras=curr_loras,
        enable_chunking=enable_chunking
    )
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: add for scheduler profiling
    '''
    sched_to_run = [s.seq_group.request_id for s in prefills.seq_groups]
    self.waiting_to_running_seqs = len(sched_to_run)
    sched_seq_tokens = [s.seq_group.get_seqs()[0].get_len() for s in prefills.seq_groups]
    self.wait_to_run_tokens = sum(sched_seq_tokens)
    '''
    ==================
    End of MLU Hijack
    ==================
    '''
    return remaining_waiting, prefills


def vllm__core__scheduler__Scheduler___schedule_running(
    self,
    running_queue: deque,
    budget: SchedulingBudget,
    curr_loras: Optional[Set[int]],
    policy: Policy,
    enable_chunking: bool = False,
) -> Tuple[deque, SchedulerRunningOutputs]:
    remaining_running, running_scheduled = vllm__core__scheduler__Scheduler___schedule_running__org(
        self=self,
        running_queue=running_queue,
        budget=budget,
        curr_loras=curr_loras,
        policy=policy,
        enable_chunking=enable_chunking
    )
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: add for scheduler profiling
    '''
    sched_to_preempt = [s.request_id for s in running_scheduled.preempted]
    self.running_to_waiting_seqs += len(sched_to_preempt)
    '''
    ==================
    End of MLU Hijack
    ==================
    '''
    return remaining_running, running_scheduled


def vllm__core__scheduler__Scheduler___schedule(self) -> SchedulerOutputs:
    scheduler_outputs = vllm__core__scheduler__Scheduler___schedule__org(self)
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: add for scheduler profiling
    '''
    self.sched_step += 1
    '''
    ==================
    End of MLU Hijack
    ==================
    '''
    return scheduler_outputs


def vllm__core__scheduler__Scheduler__free_finished_seq_groups(self) -> None:
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: add for scheduler profiling
    '''
    finished_seq_groups_ = [seq_group for seq_group in self.running
                                    if seq_group.is_finished()]

    for queue in [self.running, self.swapped, self.waiting]:
        self._finished_requests_ids += [
            seq_group.request_id for seq_group in queue
            if seq_group.is_finished()
        ]
    self.running = deque(seq_group for seq_group in self.running
                            if not seq_group.is_finished())

    self.running_seqs = len(self.running)
    self.waiting_seqs = len(self.waiting)
    self.swapped_seqs = len(self.swapped)
    self.finished_seqs += len(finished_seq_groups_)
    self.finished_seq_groups += finished_seq_groups_

    total_seqs_ = self.running_seqs + self.waiting_seqs + self.swapped_seqs + self.finished_seqs
    if total_seqs_ == 0:
        return

    if total_seqs_ > self.total_seqs:
        self.total_seqs = total_seqs_

    self.batch_utils = self.running_seqs / self.scheduler_config.max_num_seqs
    self.block_utils = (self.block_manager.num_total_gpu_blocks -
                        self.block_manager.get_num_free_gpu_blocks()) / self.block_manager.num_total_gpu_blocks
    self.preempt_ratio = self.running_to_waiting_seqs / self.total_seqs

    df_ = pd.DataFrame(
        [[self.waiting_seqs, self.running_seqs, self.swapped_seqs, len(finished_seq_groups_),
          self.waiting_to_running_seqs, self.running_to_waiting_seqs, self.wait_to_run_tokens,
          self.batch_utils, self.block_utils, self.preempt_ratio]],
        columns=['waiting', 'running', 'swapped', 'finished',
                 'wait_to_run_reqs', 'run_to_wait_reqs', 'wait_to_run_tokens',
                 'batch_utils', 'block_utils', 'preempt_ratio'],
        index=[str(self.sched_step)])
    self.df = pd.concat([self.df, df_])
    '''
    ==================
    End of MLU Hijack
    ==================
    '''


def vllm__core__scheduler__Scheduler____del__(self):
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: add for scheduler profiling
    '''
    self.save_scheduler_view()
    '''
    ==================
    End of MLU Hijack
    ==================
    '''


MluHijackObject.add_hijack_object(Scheduler,
                                  Scheduler.__init__,
                                  vllm__core__scheduler__Scheduler____init__)
MluHijackObject.add_hijack_object(Scheduler,
                                  Scheduler._schedule_prefills,
                                  vllm__core__scheduler__Scheduler___schedule_prefills)
MluHijackObject.add_hijack_object(Scheduler,
                                  Scheduler._schedule_running,
                                  vllm__core__scheduler__Scheduler___schedule_running)
MluHijackObject.add_hijack_object(Scheduler,
                                  Scheduler._schedule,
                                  vllm__core__scheduler__Scheduler___schedule)
MluHijackObject.add_hijack_object(Scheduler,
                                  Scheduler.free_finished_seq_groups,
                                  vllm__core__scheduler__Scheduler__free_finished_seq_groups)
MluHijackObject.add_hijack_object(Scheduler,
                                  "__del__",
                                  vllm__core__scheduler__Scheduler____del__)
MluHijackObject.add_hijack_object(Scheduler,
                                  "init_scheduler_view",
                                  vllm__core__scheduler__Scheduler__init_scheduler_view)
MluHijackObject.add_hijack_object(Scheduler,
                                  "save_scheduler_view",
                                  vllm__core__scheduler__Scheduler__save_scheduler_view)
