from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.mlu_hijack.mlu_hijack_utils import MluHijackObject
from vllm.logger import init_logger

logger = init_logger(__name__)


# for client init/reset server scheduler profile data
def vllm__engine__async_llm_engine__AsyncLLMEngine__init_scheduler_view(self):
    for scheduler in self.engine.scheduler:
        if hasattr(scheduler, "init_scheduler_view"):
            scheduler.init_scheduler_view()
        else:
            logger.warning("Can not find any scheduler view, " +
                           "please 'export VLLM_SCHEDULER_PROFILE=true' first.")


# for client pulling server scheduler profile data
def vllm__engine__async_llm_engine__AsyncLLMEngine__save_scheduler_view(self):
    for idx, scheduler in enumerate(self.engine.scheduler):
        if hasattr(scheduler, "save_scheduler_view"):
            scheduler.save_scheduler_view(idx)
        else:
            logger.warning("Can not find any scheduler view, " +
                           "please 'export VLLM_SCHEDULER_PROFILE=true' first.")


MluHijackObject.add_hijack_object(AsyncLLMEngine,
                                  "init_scheduler_view",
                                  vllm__engine__async_llm_engine__AsyncLLMEngine__init_scheduler_view)
MluHijackObject.add_hijack_object(AsyncLLMEngine,
                                  "save_scheduler_view",
                                  vllm__engine__async_llm_engine__AsyncLLMEngine__save_scheduler_view)
